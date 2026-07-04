import torch
import torch.nn as nn
from torch_geometric.nn.encoding import TemporalEncoding
from src.Models.modules import MLP, ReplayBuffer, Identity, AttentionAggregator, LSTMAggregator, GraphSageModule, GATModule

def init_temporal_module(module, **kwargs):

    if module.lower() == 'identity':
        return Identity()
    elif module.lower() == 'lstm':
        return LSTMAggregator(**kwargs)
    elif module.lower() == 'attention':
        return AttentionAggregator(**kwargs) 
    else :
        raise NotImplementedError(f'init_temporal_module : {module} is not implemented')

def init_spatial_module(module, **kwargs):

    if module.lower() == 'identity':
        return Identity()
    elif module.lower() == 'graphsage':
        return GraphSageModule(**kwargs)
    elif module.lower() == 'gat':
        return GATModule(**kwargs)
    else :
        raise NotImplementedError(f'init_spatial_module : {module} is not implemented')

class STM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 temporal_module_params, spatial_module_params,  
                 temporal_augments=True, spatial_augments=True,
                 time_dim=0, dropout=0.5, skip_connection_spatial=False,
                 metadata=None,
                 device='cpu'):
        """
        Implementation of the Spatio-Temporal Memory (STM) model
        """
        super().__init__()

        #============================================================================#
        #============================================================================#
        self.apply_temporal_augments = temporal_augments
        self.apply_spatial_augments = spatial_augments
        self.device = device
        self.hetero_mode = metadata is not None
        self.skip_connection_spatial = skip_connection_spatial

        self.d_t = time_dim
        self.d_h = hidden_channels

        self.temporal_module = init_temporal_module(
                                                        temporal_module_params['Type'],
                                                        **temporal_module_params['Args'],
                                                        metadata=temporal_module_params['Metadata']
                                                    )
        self.temporal_module = self.temporal_module.to(device)
        self.spatial_module = init_spatial_module(
                                                    spatial_module_params['Type'],
                                                    **spatial_module_params['Args'],
                                                    metadata=spatial_module_params['Metadata']
                                                )
        self.spatial_module = self.spatial_module.to(device)
        self.time_encoder = TemporalEncoding(out_channels=time_dim).to(device) if time_dim > 0 else None

        #============================================================================#
        #============================================================================#
        def init_module(init_fn, layer_type):
            """Helper function to initialize modules with optional heterogeneity"""
            if (layer_type.lower() in ['proj_layer']):
                return nn.ModuleDict({key: init_fn(key) for key in metadata[0]}) if self.hetero_mode else init_fn()
            else:
                return nn.ModuleDict({key: init_fn() for key in metadata[0]}) if self.hetero_mode else init_fn()
        # Initializing some modules
        #============================================================================#
        # Replay buffers for augmentations
        self.last_spatial = init_module(lambda: ReplayBuffer(1, (spatial_module_params['Args']['out_channels'],), device=device), layer_type='last_spatial')
        self.last_temporal = init_module(lambda: ReplayBuffer(1, (temporal_module_params['Args']['out_channels'],), device=device), layer_type='last_temporal')
        #============================================================================#
        # MLP Layers
        if self.hetero_mode:
            def init_proj_layer_hetero_mode(key):
                return MLP(in_channels[key]+time_dim, self.d_h, hidden_sizes=[self.d_h], dropout=dropout, device=device)
            self.proj_layer = init_module(init_proj_layer_hetero_mode, layer_type='proj_layer') 
        else:
            self.proj_layer = init_module(lambda: MLP(in_channels+time_dim, self.d_h, hidden_sizes=[self.d_h], dropout=dropout, device=device), layer_type='proj_layer')
        #============================================================================#
        # Spatial and temporal projection layers
        # IMPORTANT: the input of self.spatial_in is self.last_temporal[k] and the output should be the same shape as the output of self.proj_layer
        self.spatial_in = init_module(lambda: MLP(temporal_module_params['Args']['out_channels'], self.d_h, hidden_sizes=[self.d_h], dropout=dropout, device=device), layer_type='spatial_in')
        # IMPORTANT: the input of self.temporal_in is self.last_spatial[k] and the output should be the same shape as the output of self.proj_layer 
        self.temporal_in = init_module(lambda: MLP(spatial_module_params['Args']['out_channels'], self.d_h, hidden_sizes=[self.d_h], dropout=dropout, device=device), layer_type='temporal_in')
        #============================================================================#
        # Output layer
        self.out = init_module(lambda: MLP(self.d_h, out_channels, hidden_sizes=[self.d_h], dropout=dropout, device=device), layer_type='out')

        self.to(device)

    def to(self, device):
        """Move model and memories to the specified device."""
        self.device = device
        return super().to(device)

    @torch.no_grad()
    def clear(self):
        """Clears the memory buffers."""
        if hasattr(self.temporal_module, 'clear') and callable(self.temporal_module.clear):
            self.temporal_module.clear()

        for mem in (self.last_temporal.values() if self.hetero_mode else [self.last_temporal]):
            mem.clear()
        for mem in (self.last_spatial.values() if self.hetero_mode else [self.last_spatial]):
            mem.clear()

    @torch.no_grad()
    def detach(self):
        """Detaches memory state from computation graph."""
        if hasattr(self.temporal_module, 'detach') and callable(self.temporal_module.detach):
            self.temporal_module.detach()

        for mem in (self.last_temporal.values() if self.hetero_mode else [self.last_temporal]):
            mem.detach()
        for mem in (self.last_spatial.values() if self.hetero_mode else [self.last_spatial]):
            mem.detach()
        
    def add_time_encoding_(self, x, timestamps):
        """Adds temporal encoding to the input tensor."""
        if self.time_encoder is None:
            return x
        return {k: torch.cat([x[k], self.time_encoder(timestamps[k])], dim=-1) for k in x} if self.hetero_mode else torch.cat([x, self.time_encoder(timestamps)], dim=-1)

    def augments_spatial_(self, x, ids):
        """Applies spatial augmentation."""
        if not self.apply_spatial_augments:
            return x
        
        if (self.hetero_mode):
            spat_augment = {} 
            for k in x:
                if (x[k].size(0) > 0):
                    spat_augment[k] = x[k] + self.spatial_in[k](self.last_temporal[k].batch_retrieve(ids[k])[0].squeeze(1))
                else:
                    spat_augment[k] = torch.empty(0, self.d_h, device=self.device)
        else:
            spat_augment = x + self.spatial_in(self.last_temporal.batch_retrieve(ids)[0].squeeze(1))

        return spat_augment

    def augments_temporal_(self, x, ids):
        """Applies temporal augmentation."""
        if not self.apply_temporal_augments:
            return x
        
        if (self.hetero_mode):
            temp_augment = {} 
            for k in x:
                if (x[k].size(0) > 0):
                    temp_augment[k] = x[k] + self.temporal_in[k](self.last_spatial[k].batch_retrieve(ids[k])[0].squeeze(1))
                else:
                    temp_augment[k] = torch.empty(0, self.d_h, device=self.device)
        else:
            temp_augment = x + self.temporal_in(self.last_spatial.batch_retrieve(ids)[0].squeeze(1))

        return temp_augment

    def mem_spatial_(self, x, ids):
        """Memorizes spatial embeddings for temporal augments."""
        if self.hetero_mode:
            for k in x:
                self.last_spatial[k].batch_add(ids[k], x[k])
        else:
            self.last_spatial.batch_add(ids, x)

    def mem_temporal_(self, x, ids):
        """Memorizes temporal embeddings for spatial augments."""
        if self.hetero_mode:
            for k in x:
                self.last_temporal[k].batch_add(ids[k], x[k])
        else:
            self.last_temporal.batch_add(ids, x)

    def embed_(self, xt, spatial_embeddings, temporal_embeddings, xs=None):
        """Aggregates spatial and temporal embeddings."""
        if (not self.skip_connection_spatial):
            if self.hetero_mode:
                out = {}
                for k in xt:
                    out[k] = self.out[k](xt[k] + spatial_embeddings[k] + temporal_embeddings[k])
            else:
                out = self.out(xt + spatial_embeddings + temporal_embeddings)
        else:
            if self.hetero_mode:
                out = {}
                for k in xt:
                    out[k] = self.out[k](xt[k] + xs[k] + spatial_embeddings[k] + temporal_embeddings[k])
            else:
                out = self.out(xt + xs + spatial_embeddings + temporal_embeddings) 
    
        return out

    def forward(self, xt, ids, edge_index, edge_attr, timestamps):
        """Forward pass of the STM model."""
        if xt is None or len(xt) == 0:
            return torch.empty(0, self.d_h, device=self.device) if not self.hetero_mode else {k: torch.empty(0, self.d_h, device=self.device) for k in xt}

        # Temporal encoding
        x = self.add_time_encoding_(xt, timestamps)
        x = {k: self.proj_layer[k](x[k]) for k in x} if self.hetero_mode else self.proj_layer(x)

        # Spatial embeddings
        xs = self.augments_spatial_(x, ids)
        spatial_embeddings = self.spatial_module(xs, ids, edge_index, edge_attr, timestamps) 
        self.mem_spatial_(spatial_embeddings, ids) 

        # Temporal embeddings
        xt = self.augments_temporal_(x, ids)
        temporal_embeddings = self.temporal_module(xt, ids, edge_index, edge_attr, timestamps)
        self.mem_temporal_(temporal_embeddings, ids)

        # Final output
        if (not self.skip_connection_spatial):
            out = self.embed_(
                                xt=xt,
                                spatial_embeddings=spatial_embeddings,
                                temporal_embeddings=temporal_embeddings,
                                xs=None
                            ) 
        else:
            out = self.embed_(
                                xt=xt,
                                spatial_embeddings=spatial_embeddings,
                                temporal_embeddings=temporal_embeddings,
                                xs=xs
                             ) 
        return out

    def __repr__(self):
        return f'Initialized STM on device {self.device}\n{super().__repr__()}'
