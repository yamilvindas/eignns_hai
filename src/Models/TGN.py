import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter
from torch_geometric.utils._scatter import scatter_argmax
from torch_geometric.nn.models.tgn import TGNMemory as TGNMemoryTorchGeo
from src.Models.modules import MLP
from typing import Union, Dict, Any, Optional, Callable, Tuple, Iterable, List
from copy import deepcopy


# MESSAGE MODULES
class IdentityMessage(nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor):
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)
 
class MLPMessage(nn.Module):
    def __init__(
                    self,
                    raw_msg_dim: int,
                    memory_dim: int,
                    time_dim: int,
                    hidden_channels: int,
                    dropout: float = 0.5,
                    device : Optional[str]='cpu'
                ):
        super().__init__()
        self.out_channels = hidden_channels
        self.mlp = MLP(in_channels = raw_msg_dim + 2 * memory_dim + time_dim, 
                    out_channels = self.out_channels,
                    hidden_sizes = [hidden_channels], 
                    dropout=dropout, 
                    batch_norm=False,
                    device=device
                    )

    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor):
        return self.mlp(torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1))


# MEMORY AGGREGATORS
class LastAggregator(nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        argmax = scatter_argmax(t, index, dim=0, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out
    

class MeanAggregator(nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        x = scatter(msg, index, dim=0, dim_size=dim_size, reduce='mean').to(msg.device)
        return x
    

class SumAggregator(nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        x = scatter(msg, index, dim=0, dim_size=dim_size, reduce='sum').to(msg.device)
        return x
    

#======================================================================#
#======================================================================#
# HOMOGENEOUS TGNMemory strongly inspired from Pytorch Geometric Implementation
#======================================================================#
#======================================================================#
# model definition
import copy
from collections import defaultdict
from torch.nn import GRUCell, Linear
TGNMessageStoreType = Dict[int, Tuple[Tensor, Tensor, Tensor, Tensor]]


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int, device: Optional[str] = 'cpu'):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels, device=device)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.view(-1, 1)).cos()
    

class TGNMemory(torch.nn.Module):
    r"""The Temporal Graph Network (TGN) memory model from the
    `"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    <https://arxiv.org/abs/2006.10637>`_ paper.

    .. note::

        For an example of using TGN, see `examples/tgn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        tgn.py>`_.

    Args:
        num_nodes (int): The number of nodes to save memories for.
        raw_msg_dim (int): The raw message dimensionality.
        memory_dim (int): The hidden memory dimensionality.
        time_dim (int): The time encoding dimensionality.
        message_module (torch.nn.Module): The message function which
            combines source and destination node memory embeddings, the raw
            message and the time encoding.
        aggregator_module (torch.nn.Module): The message aggregator function
            which aggregates messages to the same destination into a single
            representation.
    """
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 time_dim: int, message_module: Callable,
                 aggregator_module: Callable,
                 device: Optional[str] = 'cpu'):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim, device=device)
        self.gru = GRUCell(message_module.out_channels, memory_dim, device=device)

        self.register_buffer('memory', torch.empty(num_nodes, memory_dim, device=device))
        last_update = torch.empty(self.num_nodes, dtype=torch.long, device=device)
        self.register_buffer('last_update', last_update)
        self.register_buffer('_assoc', torch.empty(num_nodes, dtype=torch.long, device=device))

        self.msg_s_store = {}
        self.msg_d_store = {}

        self.to(device)
        self.reset_parameters()

    def to(self, device):
        self.device = device
        self.memory = self.memory.to(device)
        self.last_update = self.last_update.to(device)
        self._assoc = self._assoc.to(device)
        self.msg_s_module = self.msg_s_module.to(device)
        self.msg_d_module = self.msg_d_module.to(device)
        self.aggr_module = self.aggr_module.to(device)
        self.time_enc = self.time_enc.to(device)
        self.gru = self.gru.to(device)
        return super().to(device)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if hasattr(self.msg_s_module, 'reset_parameters'):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, 'reset_parameters'):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, 'reset_parameters'):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.gru.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        self.memory.zero_()
        self.last_update.zero_()
        self._reset_message_store()

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp.
        """
        if self.training:
            memory, last_update = self._get_updated_memory(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]

        return memory, last_update

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
        """Updates the memory with newly encountered interactions
        :obj:`(src, dst, t, raw_msg)`.
        """
        n_id = torch.cat([src, dst]).unique().to(self.device)

        if self.training:
            self._update_memory(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id)

    def _reset_message_store(self):
        i = self.memory.new_empty((0, ), dtype=torch.long).to(self.device)
        msg = self.memory.new_empty((0, self.raw_msg_dim)).to(self.device)
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_memory(self, n_id: Tensor):
        memory, last_update = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, dst_s = self._compute_msg(n_id, self.msg_s_store,
                                                     self.msg_s_module)

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self._compute_msg(n_id, self.msg_d_store,
                                                     self.msg_d_module)

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Get local copy of updated memory.
        memory = self.gru(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce='max')[n_id].long()

        return memory, last_update

    def _update_msg_store(self, src: Tensor, dst: Tensor, t: Tensor,
                          raw_msg: Tensor, msg_store: TGNMessageStoreType):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id, perm.split(count.tolist())):
            msg_store[i.item()] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _compute_msg(self, n_id: Tensor, msg_store: TGNMessageStoreType,
                     msg_module: Callable):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        
        src = torch.cat([s.to(self.device) for s in src], dim=0)
        dst = torch.cat([d.to(self.device) for d in dst], dim=0)
        t   = torch.cat([tt.to(self.device) for tt in t], dim=0)

        raw_msg = [m.to(self.device) for i, m in enumerate(raw_msg) if m.numel() > 0 or i == 0]
        raw_msg = torch.cat(raw_msg, dim=0).to(self.device)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)

        return msg, t, src, dst

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self._update_memory(
                torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        super().train(mode)

#======================================================================#
#======================================================================#
# Final Temporal Graph Network
#======================================================================#
#======================================================================#
class TGN(nn.Module):
    def __init__(self, in_channels : int, hidden_channels : int, time_dim : int, out_channels : int, num_nodes :int, 
                 message_module : Callable, aggregator_module: Callable, metadata : Tuple=None,
                dim_enc_nodes_feaures: int=None, input_proj_homo: bool=False, device : Optional[str]='cpu'):
        """
        Temporal Graph Networks for Deep Learning on Dynamic Graphs by Rossi et al. (arXiv:2006.10637)
        """
        super(TGN, self).__init__()
        # Some attributes
        self.num_nodes = num_nodes
        self.message_dim = in_channels
        self.hidden_channels = hidden_channels
        self.time_dim = time_dim
        self.hetero_mode = metadata is not None
        self.input_proj_homo = input_proj_homo
        self.device = torch.device(device)

        # Total number of nodes for all types combined
        if (type(self.num_nodes) == dict):
            self.total_num_nodes = 0
            for node_type in self.num_nodes:
                self.total_num_nodes += self.num_nodes[node_type]
        else:
            self.total_num_nodes = self.num_nodes

        # Attribute to conver local node types indices into global ones
        if (self.hetero_mode):
            self.local_to_global_nodes_ids = {}
            self.inv_local_to_global_nodes_ids = {}
            self.current_global_id = 0

        # Node input encoders
        if (self.hetero_mode):
            self.nodes_input_encoders = torch.nn.ModuleDict(
                                                    {
                                                        node_type: nn.Linear(in_dim, dim_enc_nodes_feaures)
                                                        for node_type, in_dim in in_channels.items()
                                                    }
                                                )
        else:
            self.nodes_input_encoders = nn.Linear(in_channels, dim_enc_nodes_feaures)

        # TGN MEMORY
        if (self.hetero_mode):
            self.tgn = TGNMemory(
                                    num_nodes=self.total_num_nodes,
                                    raw_msg_dim=dim_enc_nodes_feaures,
                                    memory_dim=hidden_channels,
                                    time_dim=time_dim,
                                    message_module=message_module,
                                    aggregator_module=aggregator_module,
                                ).to(device)

        else:
            self.tgn = TGNMemory(
                                    num_nodes=num_nodes,
                                    raw_msg_dim=in_channels,
                                    memory_dim=hidden_channels,
                                    time_dim=time_dim,
                                    message_module=message_module,
                                    aggregator_module=aggregator_module,
                                ).to(device)

        # Output layers
        if self.hetero_mode :
            self.node_types = metadata[0]
            self.layer_out = nn.ModuleDict({key: nn.Sequential(nn.Linear(self.hidden_channels, self.hidden_channels), nn.GELU(), nn.Linear(self.hidden_channels, out_channels)) for key in metadata[0]})
            self.thresholds = {key: torch.tensor(0.5, device=device) for key in metadata[0]}
        else:
            self.layer_out = nn.Sequential(nn.Linear(self.hidden_channels, self.hidden_channels), nn.GELU(), nn.Linear(self.hidden_channels, out_channels))
            self.thresholds = torch.tensor(0.5, device=device)

        self.to(device)

    def to(self, device):
        """Move model and memories to the specified device."""
        self.device = device
        self.tgn.to(device)
        self.layer_out = self.layer_out.to(device)
        if self.hetero_mode:
            self.thresholds = {k: v.to(device) for k, v in self.thresholds.items()}
        else:
            self.thresholds = self.thresholds.to(device)
        return super().to(device)

    @torch.no_grad()
    def clear(self):
        """Clears the memory buffers."""
        self.tgn.reset_state()

    @torch.no_grad()
    def detach(self):
        """Detaches memory state from computation graph."""
        self.tgn.detach()

    def to_temporal_data(self, x, ids, edge_index, edge_attr, timestamp):
        """ Convert graph (x, ids, edge_index, edge_attr, timestamp) into temporal datas (src, dst, timestamp, features)."""
        if self.hetero_mode:
            # Getting the nodes keys
            node_keys = list(ids.keys())

            # Prepare self-loops 
            node_ids = torch.cat([ids[k] for k in node_keys])
            timestamps = torch.cat([timestamp[k] for k in node_keys])
            raw_msg = torch.cat([x[k] for k in node_keys])

            sources = destinations = node_ids  # Self-loops

            # Prepare edges
            edge_sources = []
            edge_destinations = []
            edge_messages = []
            edge_timestamps = []

            for edge_key in edge_index:
                # Getting the keys
                k0, _, k1 = edge_key

                # Getting the edges
                edges = edge_index[edge_key]

                # Getting the edges sources and destinations nodes, as well as the messages and timestamps
                edge_sources.append(edges[0])
                edge_destinations.append(ids[k1][edges[1]])
                edge_messages.append(x[k0][edges[0]])
                edge_timestamps.append(timestamp[k0][edges[0]])

            if edge_sources:
                edge_sources = torch.cat(edge_sources)
                edge_destinations = torch.cat(edge_destinations)
                edge_messages = torch.cat(edge_messages)
                edge_timestamps = torch.cat(edge_timestamps)

                sources = torch.cat([sources, edge_sources])
                destinations = torch.cat([destinations, edge_destinations])
                raw_msg = torch.cat([raw_msg, edge_messages])
                timestamps = torch.cat([timestamps, edge_timestamps])

        else: 
            # VERY IMPORTANT: IN THIS CASE WE SUPPOSE THAT THE NUMBER OF NODES AND THEIR IDS
            # VEY IMPORTANT: IS FIXED OVER TIME, AND THEY ARE ORDERED (ids[0] = 0, ids[1] = 1, etc.)
            sources = destinations = ids
            timestamps = timestamp
            raw_msg = x

            # Append edges
            if edge_index.numel() > 0:
                sources = torch.cat([sources, ids[edge_index[0]]])
                destinations = torch.cat([destinations, ids[edge_index[1]]])
                raw_msg = torch.cat([raw_msg, x[edge_index[0]]])
                timestamps = torch.cat([timestamps, timestamp[edge_index[0]]])
            

        return sources.long(), destinations.long(), timestamps.float(), raw_msg.float()

    def compute_node_embeddings(self, x, ids, edge_index, edge_attr, timestamps):
        """Computes node embeddings for both homogeneous and heterogeneous graphs."""
        # Get temporal data
        src, dst, t, raw_msg = self.to_temporal_data(x, ids, edge_index, edge_attr, timestamps)

        # Update state
        self.tgn.update_state(src, dst, t, raw_msg.detach())

        # Get memory
        n_id = torch.cat([ids[k] for k in ids.keys() if ids[k].numel() > 0]).long().to(self.device) if self.hetero_mode else ids.long().to(self.device)
        n_id_to_idx = {nid.item(): i for i, nid in enumerate(n_id)}

        memory, _ = self.tgn(n_id)  

        # Get output
        if (self.hetero_mode):
            out = {}
            for k in ids.keys():
                if ids[k].numel() > 0:
                    node_ids_k = ids[k].long()
                    out[k] = memory[[n_id_to_idx[n.item()] for n in node_ids_k]]
                else:
                    out[k] = torch.empty(0, self.hidden_channels, device=self.device)
            # out = {k: memory[torch.isin(n_id, nodes[k])] if nodes[k].numel() > 0 else torch.empty(0, self.hidden_channels).to(nodes[k].device) for k in nodes.keys()}
        else:
            # Get the output
            out = memory if n_id.numel() > 0 else torch.empty(0, self.hidden_channels).to(n_id.device)

        return out

    def forward(self, x, ids, edge_index, edge_attr, timestamps):
        """Forward pass."""
        # Convert to local node types indices to global indices (homogenous for all the node types)
        if (self.hetero_mode):
            # Transforming node IDs into global IDs
            global_ids = {}
            local_node_ids_within_graph = {}
            for node_type in list(ids.keys()):
                # If necessary, adding the current local node ID for the current type to the global IDs dict mapping
                # AND creating the global_ids for the current node type to use later on this forward function
                global_ids[node_type] = torch.empty(ids[node_type].shape)
                for tmp_id in range(len(ids[node_type])):
                    # Add to attribute mapping local node types IDs to global one
                    if ( (node_type, int(ids[node_type][tmp_id])) not in self.local_to_global_nodes_ids):
                        self.local_to_global_nodes_ids[(node_type, int(ids[node_type][tmp_id]))] = self.current_global_id
                        self.inv_local_to_global_nodes_ids[self.current_global_id] = (node_type, int(ids[node_type][tmp_id]))
                        self.current_global_id += 1

                    # Updating the local node IDs within the current graph (needed for remapping edge_index as it is what Pytorch Geometric needs)
                    local_node_ids_within_graph[(node_type, int(ids[node_type][tmp_id]))] = tmp_id
                    
                    # Updating the global IDs for this forward function (using current samples)
                    global_ids[node_type][tmp_id] = self.local_to_global_nodes_ids[(node_type, int(ids[node_type][tmp_id]))]

            # Transforming the nodes IDs in the edges to global IDs
            remapped_edge_index = {}
            for edge_type in list(edge_index.keys()):
                # Getting the types of the source and destination nodes of the current edges
                src_type, _, dst_type = edge_type

                # Updating the edges indices to the global ones
                remapped_edge_index[edge_type] = torch.empty(edge_index[edge_type].shape)
                src_ids = edge_index[edge_type][0, :]
                dst_ids = edge_index[edge_type][1, :]
                for tmp_id in range(edge_index[edge_type].shape[1]): # edge_index[edge_type] is of shape (2, n_edges)
                    remapped_edge_index[edge_type][0, tmp_id] = local_node_ids_within_graph[(src_type, int(src_ids[tmp_id]))] 
                    remapped_edge_index[edge_type][1, tmp_id] = local_node_ids_within_graph[(dst_type, int(dst_ids[tmp_id]))] 
                remapped_edge_index[edge_type] = remapped_edge_index[edge_type].long()

        else:
            global_ids = ids
            remapped_edge_index = edge_index

        # Input projection
        if self.hetero_mode:
            x_proj = {k: self.nodes_input_encoders[k](v) for k, v in x.items()}
        else:
            if (self.input_proj_homo):
                x_proj = self.nodes_input_encoders(x)
            else:
                x_proj = x

        # Computing the output
        out = self.compute_node_embeddings(x_proj, global_ids, remapped_edge_index, edge_attr, timestamps)
        if self.hetero_mode:
            return {k: self.layer_out[k](v) for k, v in out.items()}
        return self.layer_out(out)
