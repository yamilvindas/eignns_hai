import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Sequence, Dict

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData


# https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/signal/dynamic_graph_temporal_signal.html
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/signal/dynamic_hetero_graph_temporal_signal.html#DynamicHeteroGraphTemporalSignal
Edge_Indices = Sequence[Union[Dict[Tuple[str, str, str], np.ndarray], np.ndarray, None]]
Edge_Weights = Sequence[Union[Dict[Tuple[str, str, str], np.ndarray], np.ndarray, None]]
Node_Features = Sequence[Union[Dict[str, np.ndarray], np.ndarray, None]] 
Targets = Sequence[Union[Dict[str, np.ndarray], np.ndarray, None]]
Additional_Features = Sequence[Union[Dict[str, np.ndarray], np.ndarray, None]]

class DynamicGraphTemporalSignal(object):
    r"""A data iterator object to contain a dynamic graph with a
    changing edge set and weights . The feature set and node labels
    (target) are also dynamic. The iterator returns a single discrete temporal
    snapshot for a time period (e.g. day or week). This single snapshot is a
    Pytorch Geometric Data object. Between two temporal snapshots the edges,
    edge weights, target matrices and optionally passed attributes might change.

    Args:
        edge_indices (Sequence of Numpy arrays): Sequence of edge index tensors.
        edge_weights (Sequence of Numpy arrays): Sequence of edge weight tensors.
        features (Sequence of Numpy arrays): Sequence of node feature tensors.
        targets (Sequence of Numpy arrays): Sequence of node label (target) tensors.
        **kwargs (optional Sequence of Numpy arrays): Sequence of additional attributes.
    """

    def __init__(
        self,
        edge_indices: Edge_Indices,
        edge_weights: Edge_Weights,
        features: Node_Features,
        targets: Targets,
        **kwargs: Additional_Features
    ):
        self.edge_indices = edge_indices
        self.edge_weights = edge_weights
        self.features = features
        self.targets = targets
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()
        self.to(torch.device("cpu"))

    def _check_temporal_consistency(self):
        assert len(self.features) == len(
            self.targets
        ), "Temporal dimension inconsistency."
        assert len(self.edge_indices) == len(
            self.edge_weights
        ), "Temporal dimension inconsistency."
        assert len(self.features) == len(
            self.edge_weights
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.targets) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.features)

    def __len__(self):
        return self.snapshot_count
    
    def to(self, device):
        self.device = device

    def _get_edge_index(self, time_index: int):
        if self.edge_indices[time_index] is None:
            return self.edge_indices[time_index]
        else:
            return torch.LongTensor(self.edge_indices[time_index]).to(self.device)

    def _get_edge_weight(self, time_index: int):
        if self.edge_weights[time_index] is None:
            return self.edge_weights[time_index]
        else:
            return torch.FloatTensor(self.edge_weights[time_index]).to(self.device)

    def _get_features(self, time_index: int):
        if self.features[time_index] is None:
            return self.features[time_index]
        else:
            return torch.FloatTensor(self.features[time_index]).to(self.device)

    def _get_target(self, time_index: int):
        if self.targets[time_index] is None:
            return self.targets[time_index]
        else:
            if self.targets[time_index].dtype.kind == "i":
                return torch.LongTensor(self.targets[time_index]).to(self.device)
            elif self.targets[time_index].dtype.kind == "f":
                return torch.FloatTensor(self.targets[time_index]).to(self.device)

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if feature.dtype.kind == "i":
            return torch.LongTensor(feature).to(self.device)
        elif feature.dtype.kind == "f":
            return torch.FloatTensor(feature).to(self.device)

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = DynamicGraphTemporalSignal(
                self.edge_indices[time_index],
                self.edge_weights[time_index],
                self.features[time_index],
                self.targets[time_index],
                **{key: getattr(self, key)[time_index] for key in self.additional_feature_keys}
            )
        else:
            x = self._get_features(time_index)
            edge_index = self._get_edge_index(time_index)
            edge_weight = self._get_edge_weight(time_index)
            y = self._get_target(time_index)
            additional_features = self._get_additional_features(time_index)

            snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                            y=y, **additional_features)
        return snapshot

    def __next__(self):
        if self.t < len(self.features):
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self


class DynamicHeteroGraphTemporalSignal(object):
    r"""A data iterator object to contain a dynamic heterogeneous graph with a
    changing edge set and weights. The feature set and node labels
    (target) are also dynamic. The iterator returns a single discrete temporal
    snapshot for a time period (e.g. day or week). This single snapshot is a
    Pytorch Geometric HeteroData object. Between two temporal snapshots the edges,
    edge weights, target matrices and optionally passed attributes might change.

    Args:
        edge_index_dicts (Sequence of dictionaries where keys=Tuples and values=Numpy arrays):
         Sequence of relation type tuples and their edge index tensors.
        edge_weight_dicts (Sequence of dictionaries where keys=Tuples and values=Numpy arrays):
         Sequence of relation type tuples and their edge weight tensors.
        feature_dicts (Sequence of dictionaries where keys=Strings and values=Numpy arrays): Sequence of node
         types and their feature tensors.
        target_dicts (Sequence of dictionaries where keys=Strings and values=Numpy arrays): Sequence of node
         types and their label (target) tensors.
        **kwargs (optional Sequence of dictionaries where keys=Strings and values=Numpy arrays): Sequence
         of node types and their additional attributes.
    """

    def __init__(
        self,
        edge_index_dicts: Edge_Indices,
        edge_weight_dicts: Edge_Weights,
        feature_dicts: Node_Features,
        target_dicts: Targets,
        **kwargs: Additional_Features
    ):
        self.edge_index_dicts = edge_index_dicts
        self.edge_weight_dicts = edge_weight_dicts
        self.feature_dicts = feature_dicts
        self.target_dicts = target_dicts
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()
        self.to(torch.device("cpu"))

    def _check_temporal_consistency(self):
        assert len(self.feature_dicts) == len(
            self.target_dicts
        ), "Temporal dimension inconsistency."
        assert len(self.edge_index_dicts) == len(
            self.edge_weight_dicts
        ), "Temporal dimension inconsistency."
        assert len(self.feature_dicts) == len(
            self.edge_weight_dicts
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.target_dicts) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.feature_dicts)

    def __len__(self):
        return self.snapshot_count
    
    def to(self, device):
        self.device = device

    def _get_edge_index(self, time_index: int):
        if self.edge_index_dicts[time_index] is None:
            return self.edge_index_dicts[time_index]
        else:
            return {key: torch.LongTensor(value).to(self.device) for key, value in self.edge_index_dicts[time_index].items()
                    if value is not None}

    def _get_edge_weight(self, time_index: int):
        if self.edge_weight_dicts[time_index] is None:
            return self.edge_weight_dicts[time_index]
        else:
            return {key: torch.FloatTensor(value).to(self.device) for key, value in self.edge_weight_dicts[time_index].items()
                    if value is not None}

    def _get_features(self, time_index: int):
        if self.feature_dicts[time_index] is None:
            return self.feature_dicts[time_index]
        else:
            return {key: torch.FloatTensor(value).to(self.device) for key, value in self.feature_dicts[time_index].items()
                    if value is not None}

    def _get_target(self, time_index: int):
        if self.target_dicts[time_index] is None:
            return self.target_dicts[time_index]
        else:
            return {key: torch.FloatTensor(value).to(self.device) if value.dtype.kind == "f" else torch.LongTensor(value).to(self.device)
                    if value.dtype.kind == "i" else value for key, value in self.target_dicts[time_index].items()
                    if value is not None}

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if feature is None:
            return feature
        else:
            return {key: torch.FloatTensor(value).to(self.device) if value.dtype.kind == "f" else torch.LongTensor(value).to(self.device)
                    if value.dtype.kind == "i" else value for key, value in feature.items()
                    if value is not None}

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = DynamicHeteroGraphTemporalSignal(
                self.edge_index_dicts[time_index],
                self.edge_weight_dicts[time_index],
                self.feature_dicts[time_index],
                self.target_dicts[time_index],
                **{key: getattr(self, key)[time_index] for key in self.additional_feature_keys}
            )
        else:
            x_dict = self._get_features(time_index)
            edge_index_dict = self._get_edge_index(time_index)
            edge_weight_dict = self._get_edge_weight(time_index)
            y_dict = self._get_target(time_index)
            additional_features = self._get_additional_features(time_index)

            snapshot = HeteroData()
            if x_dict:
                for key, value in x_dict.items():
                    snapshot[key].x = value
            if edge_index_dict:
                for key, value in edge_index_dict.items():
                    snapshot[key].edge_index = value
            if edge_weight_dict:
                for key, value in edge_weight_dict.items():
                    snapshot[key].edge_attr = value
            if y_dict:
                for key, value in y_dict.items():
                    snapshot[key].y = value
            if additional_features:
                for feature_name, feature_dict in additional_features.items():
                    if feature_dict:
                        for key, value in feature_dict.items():
                            snapshot[key][feature_name] = value
        return snapshot

    def __next__(self):
        if self.t < len(self.feature_dicts):
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self


TemporalGraph = Union[DynamicGraphTemporalSignal, DynamicHeteroGraphTemporalSignal]


class TemporalGraphDataset(Dataset):
    """
    Collection of temporal graphs as a collection of :
        - Nodes events : dataframe (node_events_id, trial_id, timestamp, node_id) 
        - Nodes features : tensor (node_events_id, node_features) 
        - Nodes targets : tensor (node_events_id, node_targets)
        - Edges events : dataframe (edge_events_id, trial_id, timestamp, src_id, dst_id)
        - Edges features : tensor (edge_events_id, d_edge_features)
        - Edges targets : tensor (edge_events_id, edge_targets)
    """
    def __init__(self, *args, **kwargs):
        self._download(*args, **kwargs)


    @property
    def n_trials(self) -> int:
        return self.nodes_events['trial_id'].nunique()
    
    @property
    def n_nodes(self) -> int:
        return self.nodes_events['node_id'].nunique()
    
    @property
    def n_fts_nodes(self) -> int:
        return self.nodes_features.shape[-1]
    
    @property
    def n_trg_nodes(self) -> int:
        return self.nodes_targets.shape[-1]


    def timestamps(self, trial_idx : int = None, time_idxs : np.array = None) -> np.array:

        # Bolean masking
        if trial_idx is not None:
                nodes_trial_mask = self.nodes_events.trial_id == trial_idx
                edges_trial_mask = self.edges_events.trial_id == trial_idx
        else:
            nodes_trial_mask = slice(None)  # Faster than np.ones(dtype=bool)
            edges_trial_mask = slice(None)

        # Extract unique timestamps efficiently
        timestamps = np.unique(np.concatenate([
            self.nodes_events.timestamp[nodes_trial_mask],
            self.edges_events.timestamp[edges_trial_mask]
        ]))

        # Sort timestamps
        timestamps = np.sort(timestamps)

        # Apply time index filtering efficiently
        return timestamps if time_idxs is None else timestamps[time_idxs]


    def __repr__(self):
        txt = super().__repr__() + '\n'
        txt += 'Number of trials : {}\n'.format(self.n_trials)
        txt += 'Nodes events : {} {}\n'.format(type(self.nodes_events), self.nodes_events.shape)
        txt += 'Nodes features : {} {}\n'.format(type(self.nodes_features), self.nodes_features.shape)
        txt += 'Nodes targets : {} {}\n'.format(type(self.nodes_targets), self.nodes_targets.shape)
        txt += 'Edges events : {} {}\n'.format(type(self.edges_events), self.edges_events.shape)
        txt += 'Edges features : {} {}\n'.format(type(self.edges_features), self.edges_features.shape)
        txt += 'Edges targets : {} {}\n'.format(type(self.edges_targets), self.edges_targets.shape)
        return  txt


    def _download(self, *args, **kwargs) -> None:
        """ Download the dataset and store it in the object
        """
        self.nodes_events = pd.DataFrame(columns=['trial_id', 'timestamp', 'node_id'])
        self.nodes_features = np.empty((0,0))
        self.nodes_targets = np.empty((0,0))

        self.edges_events = pd.DataFrame(columns=['trial_id', 'timestamp', 'src_id', 'dst_id'])
        self.edges_features = np.empty((0,0))
        self.edges_targets = np.empty((0,0))

        
    def _snapshots(self, trial_idx : int = 0, time_idxs : np.array = None) -> TemporalGraph:
        """ Return the sequence of snapshots for a given trial idx at time indices
        """
        raise NotImplementedError


    def __getitem__(self, idx: Tuple[Union[int, List[int], np.ndarray], Union[int, List[Union[int, np.ndarray]]]]) -> Union[TemporalGraph, List[TemporalGraph]]:
        """ Return the sequence of snapshots for trial_idx at time_idxs
        """
        trial_idx, time_idxs = idx

        if trial_idx is None: # default : return all trials
            trial_idx = range(self.n_trials)

        if time_idxs is None: # default : return all time indices
            time_idxs = [None] * len(trial_idx) if hasattr(trial_idx, '__len__') else None

        if isinstance(trial_idx, int):
            return self._snapshots(trial_idx=trial_idx, time_idxs=time_idxs)
        elif isinstance(trial_idx, list) or isinstance(trial_idx, range):
            return [self._snapshots(trial_idx=i, time_idxs=j) for i, j in zip(trial_idx, time_idxs)]
        else:
            raise ValueError('Not recognized type for trial_idx :', (type(trial_idx), type(time_idxs)))
