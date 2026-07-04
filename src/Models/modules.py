from copy import deepcopy
from collections import defaultdict
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Union, Any, Dict, List, Tuple, Callable
from torch_geometric.nn import MLP, GraphSAGE, GCNConv, GATConv, GCN, SAGEConv
from edl_pytorch import Dirichlet, NormalInvGamma
from torch.nn import MultiheadAttention, TransformerEncoder, TransformerEncoderLayer
from torch.nn import LSTM
from torch_geometric.nn import to_hetero
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.nn import GAT

#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
class Identity(nn.Module):
    def __init__(self):
        """Identity module."""
        super(Identity, self).__init__()

    @torch.no_grad()
    def clear(self):
        """Clears the memory buffers. (Placeholder)"""
        pass

    @torch.no_grad()
    def detach(self):
        """Detaches memory state from computation graph. (Placeholder)"""
        pass

    def forward(self, x, *args, **kwargs):
        return x
    

class MLP(nn.Module):    
    def __init__(
        self,
        in_channels: int, out_channels: int, hidden_sizes: List[int] = [],
        dropout: float = 0.5,
        activation: Union[str, Callable] = 'gelu',
        batch_norm: bool = False,
        device: Optional[Union[str, torch.device]] = 'cpu',
        init_method: Optional[str] = None,
        residual: bool = False
    ):
        """ 
        Multi-layer perceptron with activation, dropout, and residuals connenctions.
        """
        super().__init__()
        
        # Process initialization inputs
        if dropout < 0 or dropout >= 1: raise ValueError("Dropout must be in [0, 1)")
        if isinstance(activation, str): activation = self._get_activation_fn(activation.lower())
        
        # Store config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.residual = residual and (in_channels == out_channels)  # Only allow residual if dimensions match
        
        # Build layers
        layers = []
        prev_dim = in_channels
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm: layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
            if dropout > 0: layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(prev_dim, out_channels))
        
        # Store layers
        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        if init_method:
            self._init_weights(init_method)
        
        # Move to device
        self.device = torch.device(device) if isinstance(device, str) else device
        self.to(self.device)
    
    def _get_activation_fn(self, name: str) -> Callable:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU,
            'leakyrelu': nn.LeakyReLU,
            'gelu': nn.GELU,
            'selu': nn.SELU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
        return activations[name]
    
    def _init_weights(self, method: str):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif method == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Forward pass with optional debugging and residual connection."""
        if debug:
            self._debug_input(x)
        
        # Store original input for residual connection if needed
        identity = x if self.residual else None
        
        # Forward pass
        x = self.mlp(x)
        
        # Add residual connection if enabled
        if identity is not None:
            x = x + identity
        
        if debug:
            self._debug_output(x)
        
        return x
    
    def _debug_input(self, x: torch.Tensor):
        """Debug input tensor."""
        if x.numel() == 0:
            return torch.empty((*x.shape[:-1], self.out_channels), device=self.device)
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values")
    
    def _debug_output(self, x: torch.Tensor):
        """Debug output tensor."""
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Output contains NaN or Inf values")
    
    def to(self, device):
        """Move model to device and update device reference."""
        self.device = torch.device(device) if isinstance(device, str) else device
        return super().to(self.device)


#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
class ThresholdedClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_sizes=[], threshold=0.5, activation=nn.Sigmoid, metadata=None, dropout=0.5, device='cpu'):
        """ 
        Multi-layer perceptron with thresholded binary classification head.
        """
        super().__init__()
        self.hetero_mode = metadata is not None

        def init_module(init_fn):
            """Initialize module conditionally based on metadata."""
            return nn.ModuleDict({key: init_fn() for key in metadata[0]}) if self.hetero_mode else init_fn()

        self.linear = init_module(lambda: MLP(in_channels, out_channels, hidden_sizes, dropout=dropout).to(device))
        self.activation = activation().to(device)
        self.threshold = {key: torch.tensor(threshold, device=device) for key in metadata[0]} if metadata is not None else torch.tensor(threshold, device=device)

    def forward(self, x):
        """Forward pass."""
        return {key: self.linear[key](x[key]) for key in x.keys()} if isinstance(x, dict) \
               else self.linear(x)
    
    def get_logits(self, x):
        """Get logits."""
        return self.forward(x)

    def get_probability(self, x):
        """Get probability."""
        x = self.get_logits(x)
        return {key: self.activation(x[key]) for key in x.keys()} if isinstance(x, dict) \
               else self.activation(x)
    
    def get_prediction(self, x):
        """Get binary prediction from thresholded probability."""
        proba = self.get_probability(x)
        return {key: proba[key].ge(self.threshold[key]).float() for key in proba.keys()} if isinstance(proba, dict) \
               else proba.ge(self.threshold).float()


# Create class for multi-class classifier
class MulticlassClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_sizes=[], metadata=None, dropout=0.5, device='cpu'):
        """ 
            FC model.
        """
        super().__init__()
        self.hetero_mode = metadata is not None

        def init_module(init_fn):
            """Initialize module conditionally based on metadata."""
            return torch.nn.ModuleDict({key: init_fn() for key in metadata[0]}) if self.hetero_mode else init_fn()

        self.linear = init_module(lambda: MLP(in_channels, out_channels, hidden_sizes, dropout=dropout).to(device))
        if (out_channels>2):
            activation = torch.nn.Softmax # Better suited for multi-class classification
        else:
            activation = torch.nn.Sigmoid # Better suited for binary classification
        self.activation = activation().to(device)

    def forward(self, x):
        """Forward pass."""
        return {key: self.linear[key](x[key]) for key in x.keys()} if isinstance(x, dict) \
               else self.linear(x)
    
    def get_logits(self, x):
        """Get logits."""
        return self.forward(x)

    def get_probability(self, x):
        """Get probability."""
        x = self.get_logits(x)
        return {key: self.activation(x[key]) for key in x.keys()} if isinstance(x, dict) \
               else self.activation(x)
    
    def get_prediction(self, x):
        """Get final prediction by taking the max value."""
        proba = self.get_probability(x)
        return {key: torch.argmax(proba[key], axis=1).ge(self.threshold[key]).float() for key in proba.keys()} if isinstance(proba, dict) \
               else torch.argmax(proba, axis=1)

# Create class for multi-class classifier for evidential learning
class MulticlassClassifierEvidentialLearning(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_sizes=[], metadata=None, dropout=0.5, device='cpu'):
        """ 
            FC model.
        """
        super().__init__()
        self.hetero_mode = metadata is not None

        def init_module(init_fn):
            """Initialize module conditionally based on metadata."""
            return torch.nn.ModuleDict({key: init_fn() for key in metadata[0]}) if self.hetero_mode else init_fn()

        self.linear = init_module(lambda: MLP(in_channels, out_channels, hidden_sizes, dropout=dropout).to(device))
        self.Dirichlet = init_module(lambda: Dirichlet(out_channels, out_channels).to(device))
        self.relu = nn.ReLU() # IMPORTANT: NECESSARY TO ENSURE NON-NEGATIVE EVIDENCE
        # IMPORTANT: In evidential learning we should not use softmax or sigmoid after the last layer that produces the parameters for the Dirichlet distribution.
        
    def forward(self, x):
        """Forward pass."""
        if isinstance(x, dict):
            out = {}
            for key in x.keys():
                out[key] = self.relu(self.linear[key](x[key]))
                out[key] = self.Dirichlet[key](out[key])
        else:
            out = self.relu(self.linear(x))
            out = self.Dirichlet(out)
        return out
    
    def get_logits(self, x):
        """Get logits."""
        return self.forward(x)

    def get_probability(self, x):
        """Get probability."""
        x = self.get_logits(x)
        return {key: x[key]/x[key].sum() for key in x.keys()} if isinstance(x, dict) \
               else x/x.sum()
    
    def get_prediction(self, x):
        """Get final prediction by taking the max value."""
        proba = self.get_probability(x)
        return {key: torch.argmax(proba[key], axis=1).ge(self.threshold[key]).float() for key in proba.keys()} if isinstance(proba, dict) \
               else torch.argmax(proba, axis=1)


# Create class for multi-class classifier
class MultistepClassifier(torch.nn.Module):
    def __init__(self, in_channels, num_classes, forecast_horizon, hidden_sizes=[], metadata=None, dropout=0.5, device='cpu'):
        """ 
            FC model.
        """
        super().__init__()
        self.hetero_mode = metadata is not None
        self.num_classes = num_classes
        self.forecast_horizon = forecast_horizon

        def init_module(init_fn):
            """Initialize module conditionally based on metadata."""
            return torch.nn.ModuleDict({key: init_fn() for key in metadata[0]}) if self.hetero_mode else init_fn()

        self.linear = init_module(lambda: MLP(in_channels, self.num_classes*self.forecast_horizon, hidden_sizes, dropout=dropout).to(device))
        activation = torch.nn.Softmax # Better suited for multi-class classification
        self.activation = activation(dim=1).to(device) # dim=1 BECAUSE IN THIS PARTICULAR CASE WE HAVE OUTPUTS OF SHAPE (batch_size, num_classes, forecast_horizon)

    def forward(self, x):
        """Forward pass."""
        if isinstance(x, dict):
            out = {}
            for key in x.keys():
                N = x[key].shape[0]
                out[key] = self.linear[key](x[key]).view(N, self.num_classes, self.forecast_horizon)
        else:
            N = x.shape[0]
            out = self.linear(x).view(N, self.num_classes, self.forecast_horizon)
        return out

    def get_logits(self, x):
        """Get logits."""
        return self.forward(x)

    def get_probability(self, x):
        """Get probability."""
        x = self.get_logits(x)
        return {key: self.activation(x[key]) for key in x.keys()} if isinstance(x, dict) \
               else self.activation(x)
    
    def get_prediction(self, x):
        """Get final prediction by taking the max value."""
        proba = self.get_probability(x)
        return {key: torch.argmax(proba[key], axis=1).ge(self.threshold[key]).float() for key in proba.keys()} if isinstance(proba, dict) \
               else torch.argmax(proba, axis=1)

# Create class for multi-class classifier for evidential learning
class MultistepClassifierEvidentialLearning(torch.nn.Module):
    def __init__(self, in_channels,  num_classes, forecast_horizon, hidden_sizes=[], metadata=None, dropout=0.5, device='cpu'):
        """ 
            FC model.
        """
        super().__init__()
        self.hetero_mode = metadata is not None
        self.num_classes = num_classes
        self.forecast_horizon = forecast_horizon

        def init_module(init_fn):
            """Initialize module conditionally based on metadata."""
            return torch.nn.ModuleDict({key: init_fn() for key in metadata[0]}) if self.hetero_mode else init_fn()

        self.linear = init_module(lambda: MLP(in_channels, self.num_classes*self.forecast_horizon, hidden_sizes, dropout=dropout).to(device))
        self.Dirichlet = init_module(lambda: Dirichlet(self.num_classes*self.forecast_horizon, self.num_classes*self.forecast_horizon).to(device))
        self.relu = nn.ReLU() # IMPORTANT: NECESSARY TO ENSURE NON-NEGATIVE EVIDENCE
        # IMPORTANT: In evidential learning we should not use softmax or sigmoid after the last layer that produces the parameters for the Dirichlet distribution.
        
    def forward(self, x):
        """Forward pass."""
        if isinstance(x, dict):
            out = {}
            for key in x.keys():
                N = x[key].shape[0]
                out[key] = self.relu(self.linear[key](x[key]))
                out[key] = self.Dirichlet[key](out[key]).view(N, self.num_classes, self.forecast_horizon)
        else:
            N = x.shape[0]
            out = self.relu(self.linear(x))
            out = self.Dirichlet(out).view(N, self.num_classes, self.forecast_horizon)
        return out
    
    def get_logits(self, x):
        """Get logits."""
        return self.forward(x)

    def get_probability(self, x):
        """Get probability."""
        x = self.get_logits(x)
        return {key: x[key]/x[key].sum(axis=1, keepdim=True) for key in x.keys()} if isinstance(x, dict) \
               else x/x.sum(axis=1, keepdim=True) # axis=1 BECAUSE IN THIS PARTICULAR CASE WE HAVE OUTPUTS OF SHAPE (batch_size, num_classes, forecast_horizon)
    
    def get_prediction(self, x):
        """Get final prediction by taking the max value."""
        proba = self.get_probability(x)
        return {key: torch.argmax(proba[key], axis=1).ge(self.threshold[key]).float() for key in proba.keys()} if isinstance(proba, dict) \
               else torch.argmax(proba, axis=1)

#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
# Create class for multi-class classifier
class IncRecRatesRegressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_sizes=[], metadata=None, dropout=0.5, device='cpu'):
        """ 
            FC model.
        """
        super().__init__()
        self.hetero_mode = metadata is not None

        def init_module(init_fn):
            """Initialize module conditionally based on metadata."""
            return torch.nn.ModuleDict({key: init_fn() for key in metadata[0]}) if self.hetero_mode else init_fn()

        out_channels = 2 # one for the incubation rate and another one for the recovery rate
        self.linear = init_module(lambda: MLP(in_channels, out_channels, hidden_sizes, dropout=dropout).to(device))
        activation = torch.nn.Softmax # Better suited for multi-class classification
        self.activation = activation().to(device)

    def forward(self, x):
        """Forward pass."""
        return {key: self.activation(self.linear[key](x[key])) for key in x.keys()} if isinstance(x, dict) \
               else self.activation(self.linear(x))

class BoundedNormalInvGamma(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 4)

    def forward(self, x):
        mu_raw, log_nu, log_alpha, log_beta = torch.chunk(self.fc(x), 4, dim=-1)
        mu = torch.sigmoid(mu_raw)              # bound μ
        nu = F.softplus(log_nu) + 1e-6          # > 0
        alpha = F.softplus(log_alpha) + 1.0     # > 1 for stability
        beta = F.softplus(log_beta) + 1e-6      # > 0

        return mu, nu, alpha, beta

# Create class for multi-class classifier
class IncRecRatesRegressorEvidentialLearning(torch.nn.Module):
    def __init__(self, in_channels, hidden_sizes=[], metadata=None, bounded=True, dropout=0.5, device='cpu'):
        """ 
            FC model.
        """
        super().__init__()
        self.hetero_mode = metadata is not None

        def init_module(init_fn):
            """Initialize module conditionally based on metadata."""
            return torch.nn.ModuleDict({key: init_fn() for key in metadata[0]}) if self.hetero_mode else init_fn()

        out_channels_intermediate = 16
        self.linear = init_module(lambda: MLP(in_channels, out_channels_intermediate, hidden_sizes, dropout=dropout).to(device))
        activation = torch.nn.ReLU
        self.activation = activation().to(device)
        # Creating two output heads: one for the incubation rate and another one for the recovery rate
        Head = BoundedNormalInvGamma if bounded else NormalInvGamma
        self.head_inc_rate = Head(out_channels_intermediate, 1).to(device)
        self.head_rec_rate = Head(out_channels_intermediate, 1).to(device)

    def forward(self, x):
        """Forward pass"""        
        if isinstance(x, dict):
            out_inc_rate = {}
            out_rec_rate = {}
            for key in x.keys():
                h = self.activation(self.linear[key](x[key]))
                out_inc_rate[key] = self.head_inc_rate(h)
                out_rec_rate[key] = self.head_rec_rate(h)
        else:
            h = self.activation(self.linear(x))
            out_inc_rate = self.head_inc_rate(h)
            out_rec_rate = self.head_rec_rate(h)

        return out_inc_rate, out_rec_rate
    
    

#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
# Create class for infection rate regressor
class InfectionRateRegressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_sizes=[], metadata=None, dropout=0.5, device='cpu'):
        """ 
            FC model.
        """
        super().__init__()
        self.hetero_mode = metadata is not None

        def init_module(init_fn):
            """Initialize module conditionally based on metadata."""
            return torch.nn.ModuleDict({key: init_fn() for key in metadata[0]}) if self.hetero_mode else init_fn()

        # Linear layer taking as input the concatenation of the features of both individuals
        self.linear = init_module(lambda: MLP(2*in_channels, 1, hidden_sizes, dropout=dropout).to(device))
        self.activation = torch.nn.Sigmoid().to(device)

    def expand_to_pairwise_combinations(self, x):
        # Expand x to get all pairwise combinations
        N = x.shape[0]
        M = x.shape[1]
        xi = x.unsqueeze(1).expand(N, N, M)  # shape: (N, N, M)
        xj = x.unsqueeze(0).expand(N, N, M)  # shape: (N, N, M)

        return xi, xj
    
    def forward(self, x_in):
        """Forward pass."""
        if (isinstance(x_in, dict)):
            out = {}
            for key in x_in.keys():
                # Get pairwise combinations
                xi, xj = self.expand_to_pairwise_combinations(x_in[key])
                 # Concatenate along the feature dimension
                x_pair = torch.cat([xi, xj], axis=-1) # shape: (N, N, 2M)
                # Projection
                out[key] = self.activation(self.linear[key](x_pair)).squeeze(-1) # shape: (N, N)
        else:
            # Get pairwise combinations
            xi, xj = self.expand_to_pairwise_combinations(x_in)
            # Concatenate along the feature dimension
            x_pair = torch.cat([xi, xj], axis=-1) # shape: (N, N, 2M)
            # Projection
            out = self.activation(self.linear(x_pair)).squeeze(-1) # shape: (N, N)
        
        return out
    

class EdgeEvidentialMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # mu_raw, log_nu, log_alpha, log_beta
        )
    
    def forward(self, h_pair):
        out = self.mlp(h_pair)  # (..., 4)
        mu_raw = out[..., 0]
        log_nu = out[..., 1]
        log_alpha = out[..., 2]
        log_beta = out[..., 3]

        mu = torch.sigmoid(mu_raw)            # bound mu to [0,1]
        nu = F.softplus(log_nu) + 1e-6
        alpha = F.softplus(log_alpha) + 1.0
        beta = F.softplus(log_beta) + 1e-6

        return mu, nu, alpha, beta

# Create class for infection rate regressor
class InfectionRateRegressorEvidentialLearning(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim=64, metadata=None, device='cpu'):
        """ 
            FC model.
        """
        super().__init__()
        self.hetero_mode = metadata is not None

        def init_module(init_fn):
            """Initialize module conditionally based on metadata."""
            return torch.nn.ModuleDict({key: init_fn() for key in metadata[0]}) if self.hetero_mode else init_fn()

        # Lineqr layer taking as input the concatenation of the features of both individuals

        def init_evidential_regressor():
            return nn.Sequential(
                                    nn.Linear(in_features=2*in_channels, out_features=hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 4),  # mu_raw, log_nu, log_alpha, log_beta
                                ).to(device)

        self.evidential_regressor = init_module(init_evidential_regressor)
    
    def forward(self, x_in):
        """Forward pass."""
        if (isinstance(x_in, dict)):
            raise NotImplementedError()
        else:
            # Prepare pairwise features by broadcasting and concatenation
            B, N, _ = x_in.shape
            x_i = x_in.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, M)
            x_j = x_in.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, M)
            x_pair = torch.cat([x_i, x_j], dim=-1)            # (B, N, N, 2*M)
        
            # Get output
            out = self.evidential_regressor(x_pair)  # (..., 4)
            mu_raw = out[..., 0]
            log_nu = out[..., 1]
            log_alpha = out[..., 2]
            log_beta = out[..., 3]

            mu = torch.sigmoid(mu_raw)            # bound mu to [0,1]
            nu = F.softplus(log_nu) + 1e-6
            alpha = F.softplus(log_alpha) + 1.0
            beta = F.softplus(log_beta) + 1e-6

        return mu, nu, alpha, beta
        
        
    
#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
# Creation of model taking K embeddings (of K consecutive days) and computing a single output (infection risk)
class InfectionRiskPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, windows_size, hidden_sizes=[], threshold=0.5, activation=torch.nn.Sigmoid, metadata=None, dropout=0.5, device='cpu'):
        """ 
        Multi-layer perceptron with thresholded binary classification head.
        """
        super().__init__()
        self.hetero_mode = metadata is not None
        self.windows_size = windows_size
        
        def init_module(init_fn):
            """Initialize module conditionally based on metadata."""
            return torch.nn.ModuleDict({key: init_fn() for key in metadata[0]}) if self.hetero_mode else init_fn()

        # Learnable scalar weights of shape [K, 1] to allow Xavier init
        self.fusion_weights = torch.nn.Parameter(torch.empty(windows_size, 1))
        torch.nn.init.xavier_uniform_(self.fusion_weights)  # Xavier initialization

        # Final layers
        self.linear = init_module(lambda: MLP(in_channels, out_channels, hidden_sizes, dropout=dropout).to(device))
        self.activation = activation().to(device)
        self.threshold = {key: torch.tensor(threshold, device=device) for key in metadata[0]} if metadata is not None else torch.tensor(threshold, device=device)

    def forward(self, inputs):
        """Forward pass."""
        assert len(inputs) == self.windows_size, f"Expected {self.windows_size} inputs, got {len(inputs)}"
        
        if (isinstance(inputs[0], dict)):
            out = {}
            for key in inputs[0].keys():
                tmp_inputs = [inputs[i][key] for i in range(self.windows_size)]
            
                # Stack inputs: [windows_size, n_nodes, input_dim]
                stacked = torch.stack(tmp_inputs, dim=0)
                
                # Apply weights: reshape for broadcasting [windows_size, 1, 1]
                weights_reshaped = self.fusion_weights.view(self.windows_size, 1, 1)
                weighted = weights_reshaped * stacked
    
                # Sum 
                summed = weighted.sum(dim=0)
            
                # Projection
                out[key] = self.linear[key](summed)
        else:
            # Stack inputs: [windows_size, n_nodes, input_dim]
            stacked = torch.stack(inputs, dim=0)
            
            # Apply weights: reshape for broadcasting [windows_size, 1, 1]
            weights_reshaped = self.fusion_weights.view(self.windows_size, 1, 1)
            weighted = weights_reshaped * stacked

            # Sum 
            summed = weighted.sum(dim=0)

            # Projection
            out = self.linear(summed)
            
        return out
    
    def get_logits(self, x):
        """Get logits."""
        return self.forward(x)

    def get_probability(self, x):
        """Get probability."""
        x = self.get_logits(x)
        return {key: self.activation(x[key]) for key in x.keys()} if isinstance(x, dict) \
               else self.activation(x)
    
    def get_prediction(self, x):
        """Get binary prediction from thresholded probability."""
        proba = self.get_probability(x)
        return {key: proba[key].ge(self.threshold[key]).float() for key in proba.keys()} if isinstance(proba, dict) \
               else proba.ge(self.threshold).float()

#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
class GlobalEpidemioParamsRegressor(torch.nn.Module):
    def __init__(self, n_epi_params=3, count_method="Softmax", device='cpu'):
        """ 
            FC model.
        """
        super().__init__()
        self.n_epi_params = n_epi_params
        self.count_method = count_method
        self.epi_params = torch.nn.Parameter(torch.randn(n_epi_params))
    
    def forward(self, states_probs_over_time):
        """Forward pass."""
        #======================================================================#
        #======================================================================#
        # Parameters should be positive
        params = torch.nn.functional.softplus(self.epi_params)

        #======================================================================#
        #======================================================================#
        # Consistency between the number of states and number of epidemiological parameters
        batch_size, n_nodes, n_states, n_time_steps = states_probs_over_time.shape
        if (n_states == 4): # SEIR model
            if (self.n_epi_params != 3):
                raise RuntimeError(f"\nThe number of states correspond to a SEIR model, but the number of epidemiological parameters to learn is not consistent ({self.n_epi_params} instead of 3).\n")
        elif (n_states == 6):
            if (self.n_epi_params != 11): # SEIRD-NS model
                raise RuntimeError(f"\nThe number of states correspond to a SEIR-NS model, but the number of epidemiological parameters to learn is not consistent ({self.n_epi_params} instead of 11).\n")
        else:
            raise RuntimeError(f"\nThere are more states ({n_states}) than for a SEIR or SEIRD-NS epidemiological model.")
        #======================================================================#
        #======================================================================#
        # Counting the number of values per state
        # IMPORTANT: ARGMAX IS NOT DIFFERENTIABLE, SO TO COUNT THE NUMBER OF PATIENTS PER COMPARTMENT WE HAVE TO DO IT IN
        # IMPORTANT: ANOTHER (DIFFERENTIABLE) WAY. FOR INSTANCE, WE CAN USE SOFTMAX
        states_over_time = torch.zeros((batch_size, n_states, n_time_steps))
        if (self.count_method.lower() == "softmax"):
            expected_counts_per_compartment = torch.sum(states_probs_over_time, dim=1)
            if (n_states == 4): # SEIR model
                S = expected_counts_per_compartment[:, 0, :]
                E = expected_counts_per_compartment[:, 1, :]
                I = expected_counts_per_compartment[:, 2, :]
                R = expected_counts_per_compartment[:, 3, :]
                N = S + E + I + R
                beta = params[0]
                gamma = params[1]
                sigma = params[2]
                dS_dT = -beta*I*S/N
                dE_dT = -dS_dT - sigma*E
                dI_dT = sigma*E - gamma*I
                dR_dT = gamma*I 
                # Output
                out = torch.stack([dS_dT, dE_dT, dI_dT, dR_dT], dim=1)
            elif (n_states == 6):
                # States
                S = expected_counts_per_compartment[:, 0, :]
                E = expected_counts_per_compartment[:, 1, :]
                I = expected_counts_per_compartment[:, 2, :]
                R = expected_counts_per_compartment[:, 3, :]
                D = expected_counts_per_compartment[:, 4, :]
                NS = expected_counts_per_compartment[:, 5, :]
                N = S + E + I + R + D + NS
                # Params
                beta = params[0]
                A = params[1]
                A_S = params[2]
                A_E = params[3]
                A_I = params[4]
                A_R = params[5]
                A_NS = params[6]
                dis_rate = params[7]
                alpha = params[8]
                gamma = params[9]
                mu = params[10]
                # Derivatives
                dS_dT = -beta * S * I / N + A_S*A - dis_rate*S
                dE_dT = beta * S * I / N + A_E*A - (alpha + dis_rate) * E
                dI_dT = alpha * E + A_I*A - (gamma + mu + dis_rate) * I
                dR_dT = gamma * I + A_R*A - dis_rate*R
                dD_dT = mu * I - D
                dNS_dT = A_NS*A - D*NS
                # Output
                out = torch.stack([dS_dT, dE_dT, dI_dT, dR_dT, dD_dT, dNS_dT], dim=1)
        elif (self.count_method.lower() == "gumbelsoftmax"):
            raise NotImplementedError()
        else:
            raise ValueError("\nMethod {} to approximate argmax for compartment counting is no valid\n".format(self.count_method))
        

        return out

class PerIndEpidemioParamsRegressor(torch.nn.Module):
    def __init__(self, n_epi_params=3, device='cpu'):
        """ 
            FC model.
        """
        super().__init__()
        self.n_epi_params = n_epi_params
        self.epi_params = torch.nn.Parameter(torch.randn(n_epi_params))
    
    def forward_single_step(self, states_probs_t, adj_mat_t, normalize_adj_mat=False):
        """
            Does the forward pass when we have only one future step (i.e. forecast
            horizon of 1).

            IMPORTANT: We use the adjacency matrix at time t (so the interactions and
            connections between individuals at that time) to compute the force of
            infection, and get the right hand side of the ODEs. As interactions evolve
            over time, this makes that for increasing forecast horizons, we have a mismatch
            between what we assume to be the current interactions of individuals and the true ones.
        """
        #======================================================================#
        #======================================================================#
        # Parameters should be positive
        params = torch.nn.functional.softplus(self.epi_params)

        #======================================================================#
        #======================================================================#
        # Consistency between the number of states and number of epidemiological parameters
        batch_size, n_nodes, n_states = states_probs_t.shape
        if (n_states == 3): # SIR model with immigration/emmigration
            if (self.n_epi_params != 9):
                raise RuntimeError(f"\nThe number of states correspond to a SIR model with immigration/emmigration, but the number of epidemiological parameters to learn is not consisten ({self.n_epi_params} instead of 9).\n")
        
        
        elif (n_states == 4): # SEIR model
            if (self.n_epi_params != 3):
                raise RuntimeError(f"\nThe number of states correspond to a SEIR model, but the number of epidemiological parameters to learn is not consisten ({self.n_epi_params} instead of 3).\n")
        elif (n_states == 6):
            if (self.n_epi_params != 11): # SEIRD-NS model
                raise RuntimeError(f"\nThe number of states correspond to a SEIR-NS model, but the number of epidemiological parameters to learn is not consisten ({self.n_epi_params} instead of 11).\n")
        else:
            raise RuntimeError(f"\nThere are more states ({n_states}) than for a SEIR or SEIRD-NS epidemiological model.")
        
        #======================================================================#
        #======================================================================#
        # Getting the parameters of the epidemiological model
        if (n_states == 3): # SIR model with immigration/emmigration
            beta = params[0]
            gamma = params[1]
            lambda_S = params[2]
            lambda_I = params[3]
            lambda_R = params[4]
            mu_S = params[5]
            mu_I = params[6]
            mu_R = params[7]
            epsilon = params[8]
        
        elif (n_states == 4): # SEIR model
            beta = params[0]
            sigma = params[1]
            gamma = params[2]
        
        elif (n_states == 6): # SEIRD-NS model
            beta = params[0]
            A = params[1]
            A_S = params[2]
            A_E = params[3]
            A_I = params[4]
            A_R = params[5]
            A_NS = params[6]
            dis_rate = params[7]
            alpha = params[8]
            gamma = params[9]
            mu = params[10]

        #======================================================================#
        #======================================================================#
        # Getting the local infection hazard (force of infection)
        I = states_probs_t[:, :, 2]
        lambdas = (adj_mat_t @ I.unsqueeze(2)).squeeze(2)
        if (normalize_adj_mat):
            lambdas = lambdas/adj_mat_t.sum(dim=2)
        lambdas = beta*lambdas
        if (n_states == 3): # SIR model with immigration/emmigration
            lambdas = lambdas + epsilon 
        elif (n_states == 6): # SEIRD-NS model
            # Getting the fraction of arrivals per compartment in the current step
            w_t = 1/n_nodes

        #======================================================================#
        #======================================================================#
        # Getting the derivatives
        if (n_states == 3): # SIR model with immigration/emmigration
            # States
            S = states_probs_t[:, :, 0]
            I = states_probs_t[:, :, 1]
            R = states_probs_t[:, :, 2]

            # Derivatives
            dS_dT = lambda_S - (lambdas + mu_S)*S
            dI_dT = lambda_I + lambdas*S - (gamma + mu_I)*I
            dR_dT = lambda_R + gamma*I - mu_R*R

            # Output
            out = torch.stack([dS_dT, dI_dT, dR_dT], dim=1)
            
        
        elif (n_states == 4): # SEIR model
            # States
            S = states_probs_t[:, :, 0]
            E = states_probs_t[:, :, 1]
            I = states_probs_t[:, :, 2]
            R = states_probs_t[:, :, 3]

            # Derivatives
            dS_dT = -lambdas * S 
            dE_dT = lambdas * S - sigma * E
            dI_dT = sigma * E - gamma * I
            dR_dT = gamma * I

            # Output
            out = torch.stack([dS_dT, dE_dT, dI_dT, dR_dT], dim=1)
        
        elif (n_states == 6):
            # States
            S = states_probs_t[:, :, 0]
            E = states_probs_t[:, :, 1]
            I = states_probs_t[:, :, 2]
            R = states_probs_t[:, :, 3]
            D = states_probs_t[:, :, 4]
            NS = states_probs_t[:, :, 5]
            N = S + E + I + R + D + NS

            # Derivatives
            dS_dT = -(lambdas + dis_rate) * S + A_S*A*w_t
            dE_dT = lambdas * S + A_E*A*w_t - (alpha + dis_rate) * E
            dI_dT = alpha * E + A_I*A*w_t - (gamma + mu + dis_rate) * I
            dR_dT = gamma * I + A_R*A*w_t - dis_rate*R
            dD_dT = mu * I - D
            dNS_dT = A_NS*A*w_t - dis_rate*NS        

            # Output
            out = torch.stack([dS_dT, dE_dT, dI_dT, dR_dT, dD_dT, dNS_dT], dim=1)

        return out

    def forward(self, states_probs_t, adj_mat_t, normalize_adj_mat=False, use_future_adj_mats=False):
        """Forward pass."""
        if (len(states_probs_t.shape) == 3): # Forecast horizon of 1
            out = self.forward_single_step(states_probs_t, adj_mat_t, normalize_adj_mat)
        elif (len(states_probs_t.shape) == 4): # Forecast horizon greater than 1
            out = []
            bs, n_nodes, n_states, forecast_hor = states_probs_t.shape
            for hor in range(forecast_hor):
                if (use_future_adj_mats):
                    if (len(adj_mat_t.shape) != 3):
                        raise RuntimeError(f"\n When using the future adjacency matrices, adj_mat_t should be of shape (n_nodes, n_nodes, forecast horizon), whereas now it is of shape {adj_mat_t.shape}\n")
                    else:
                        single_out = self.forward_single_step(states_probs_t[:, :, :, hor], adj_mat_t[:, :, hor], normalize_adj_mat)
                        raise NotImplementedError(f"\nUse of future adjacency matrices is not implemented yet \n")
                else:
                    single_out = self.forward_single_step(states_probs_t[:, :, :, hor], adj_mat_t, normalize_adj_mat)
                    out.append(single_out)
            out = torch.stack(out, axis=-1)

        return out
        
    

class ODENextStatePredictor(torch.nn.Module):
    def __init__(self, n_epi_params=3, device='cpu'):
        """ 
            FC model.
        """
        super().__init__()
        self.n_epi_params = n_epi_params
        self.epi_params = torch.nn.Parameter(torch.randn(n_epi_params))

    def forward_single_step(self, states_probs_t, adj_mat_t, delta_t, normalize_adj_mat=False):
        """
            Does the forward pass when we have only one future step (i.e. forecast
            horizon of 1).

            IMPORTANT: We use the adjacency matrix at time t (so the interactions and
            connections between individuals at that time) to compute the force of
            infection, and get the right hand side of the ODEs. As interactions evolve
            over time, this makes that for increasing forecast horizons, we have a mismatch
            between what we assume to be the current interactions of individuals and the true ones.
        """
        #======================================================================#
        #======================================================================#
        # Parameters should be positive
        params = torch.nn.functional.softplus(self.epi_params)

        #======================================================================#
        #======================================================================#
        # Consistency between the number of states and number of epidemiological parameters
        batch_size, n_nodes, n_states = states_probs_t.shape
        if (n_states == 3): # SIR model with immigration/emmigration
            if (self.n_epi_params != 9):
                raise RuntimeError(f"\nThe number of states correspond to a SEIR model, but the number of epidemiological parameters to learn is not consisten ({self.n_epi_params} instead of 9).\n")
        
        elif (n_states == 4): # SEIR model
            if (self.n_epi_params != 3):
                raise RuntimeError(f"\nThe number of states correspond to a SEIR model, but the number of epidemiological parameters to learn is not consisten ({self.n_epi_params} instead of 3).\n")
        elif (n_states == 6):
            #if (self.n_epi_params != 11): # SEIRD-NS model
            if (self.n_epi_params != 5): # SEIRD-NS model
                raise RuntimeError(f"\nThe number of states correspond to a SEIR-NS model, but the number of epidemiological parameters to learn is not consisten ({self.n_epi_params} instead of 5).\n")
        else:
            raise RuntimeError(f"\nThere are more states ({n_states}) than for a SEIR or SEIRD-NS epidemiological model.")
        
        #======================================================================#
        #======================================================================#
        # Getting the parameters of the epidemiological model
        if (n_states == 3): # SIR model with immigration/emmigration
            beta = params[0]
            gamma = params[1]
            lambda_S = params[2]
            lambda_I = params[3]
            lambda_R = params[4]
            mu_S = params[5]
            mu_I = params[6]
            mu_R = params[7]
            epsilon = params[8]
        
        elif (n_states == 4): # SEIR model
            beta = params[0]
            sigma = params[1]
            gamma = params[2]

        elif (n_states == 6):
            # Params
            beta = params[0]
            dis_rate = params[1]
            alpha = params[2]
            gamma = params[3]
            mu = params[4]



        

        #======================================================================#
        #======================================================================#
        # Getting the local infection hazard (force of infection)
        I = states_probs_t[:, :, 2]
        lambdas = (adj_mat_t @ I.unsqueeze(2)).squeeze(2)
        if (normalize_adj_mat):
            lambdas = lambdas/adj_mat_t.sum(dim=2)
        lambdas = beta*lambdas
        if (n_states == 3): # SIR model with immigration/emmigration
            lambdas = lambdas + epsilon 
        elif (n_states == 6): # SEIRD-NS model
            # Getting the fraction of arrivals per compartment in the current step
            w_t = 1/n_nodes

        #======================================================================#
        #======================================================================#
        # Getting the probabilities that one node transition to another state
        # in the given delta_t interval
        if (n_states == 3): # SIR model with immigration/emmigration
            pass
        elif (n_states == 4): # SEIR model
            pass
        elif (n_states == 6):
            pass
        

        #======================================================================#
        #======================================================================#
        # Getting the derivatives
        if (n_states == 3): # SIR model with immigration/emmigration
            # States
            S = states_probs_t[:, :, 0]
            I = states_probs_t[:, :, 1]
            R = states_probs_t[:, :, 2]
            # Next state
            S_next = S * torch.exp(-(lambdas + mu_S)*delta_t) + ((lambda_S)/(lambdas + mu_S))*(1 - torch.exp(-(lambdas + mu_S)*delta_t))
            I_next = I * torch.exp(-(gamma + mu_I)*delta_t) + ((lambda_I)/(gamma + mu_I))*(1 - torch.exp(-(gamma + mu_I)*delta_t)) + S*((lambdas)/(lambdas + mu_S))*(1-torch.exp(-(lambdas + mu_S)*delta_t))
            R_next = R * torch.exp(-mu_R*delta_t) + (lambda_R/mu_R)*(1 - torch.exp(-mu_R*delta_t)) + I * (gamma/(gamma + mu_I)) * (1 - torch.exp(-(gamma + mu_I)*delta_t))

            # Output
            out = torch.stack([S_next, I_next, R_next], dim=1)
        
        elif (n_states == 4): # SEIR model
            # States
            S = states_probs_t[:, :, 0]
            E = states_probs_t[:, :, 1]
            I = states_probs_t[:, :, 2]
            R = states_probs_t[:, :, 3]
            # Next state
            S_next = S * torch.exp(-lambdas*delta_t)
            E_next = E * torch.exp(-sigma*delta_t) + (lambdas*S/sigma) * (1 - torch.exp(-sigma*delta_t))
            I_next = I * torch.exp(-gamma*delta_t) + (sigma*E/gamma) * (1-torch.exp(-gamma*delta_t))
            R_next = R + I * (1 - torch.exp(-gamma*delta_t)) + sigma * E * (delta_t - (1 - torch.exp(-gamma*delta_t))/gamma)

            # Output
            out = torch.stack([S_next, E_next, I_next, R_next], dim=1)
        
        elif (n_states == 6):
            # States
            S = states_probs_t[:, :, 0]
            E = states_probs_t[:, :, 1]
            I = states_probs_t[:, :, 2]
            R = states_probs_t[:, :, 3]
            D = states_probs_t[:, :, 4]
            NS = states_probs_t[:, :, 5]
            N = S + E + I + R + D + NS
            # Next state
            S_next = S * torch.exp(-(lambdas + dis_rate)*delta_t)

            E_next = E * torch.exp(-(alpha + dis_rate)*delta_t) +\
                     (lambdas * S)/(alpha + dis_rate)  * (1 - torch.exp(-(alpha + dis_rate)*delta_t))

            I_next = I * torch.exp(-(gamma + mu + dis_rate)*delta_t) +\
                    (alpha * E)/(gamma + mu + dis_rate) * (1 - torch.exp(-(gamma + mu + dis_rate)*delta_t))

            R_next = R * torch.exp(-dis_rate*delta_t) +\
                    (gamma * I * torch.exp(-dis_rate * delta_t))/(gamma + mu) * (1 - torch.exp(-(gamma + mu)*delta_t)) +\
                    gamma * (alpha * E)/(gamma + mu + dis_rate) * ( (1 - torch.exp(-dis_rate*delta_t))/dis_rate - (torch.exp(-dis_rate*delta_t) - torch.exp(-(gamma + mu + dis_rate)*delta_t))/(gamma + mu) )

            D_next = D * torch.exp(-torch.tensor(delta_t)) +\
                    (gamma * I * torch.exp(-delta_t))/(gamma + mu + dis_rate - 1) * (1 - torch.exp(-(gamma + mu + dis_rate - 1)*delta_t)) +\
                    (mu*(alpha * E))/(gamma + mu + dis_rate) * (1 - torch.exp(-delta_t) - (torch.exp(-delta_t) - torch.exp(-(gamma + mu + dis_rate)*delta_t))/(gamma + mu + dis_rate - 1))

            # NS_next = NS * torch.exp(-dis_rate*delta_t) + (A_NS*A*w_t/dis_rate) * (1 - torch.exp(-dis_rate * delta_t))
            NS_next = NS * torch.exp(-dis_rate*delta_t)

            # Output
            out = torch.stack([S_next, E_next, I_next, R_next, D_next, NS_next], dim=1)
        

        return out

    
    def forward(self, states_probs_t, adj_mat_t, delta_t, normalize_adj_mat=False, use_future_adj_mats=False):
        """Forward pass."""
        if (len(states_probs_t.shape) == 3): # Forecast horizon of 1
            out = self.forward_single_step(states_probs_t, adj_mat_t, delta_t, normalize_adj_mat)
        elif (len(states_probs_t.shape) == 4): # Forecast horizon greater than 1
            out = []
            bs, n_nodes, n_states, forecast_hor = states_probs_t.shape
            for hor in range(forecast_hor):
                if (use_future_adj_mats):
                    if (len(adj_mat_t.shape) != 3):
                        raise RuntimeError(f"\n When using the future adjacency matrices, adj_mat_t should be of shape (n_nodes, n_nodes, forecast horizon), whereas now it is of shape {adj_mat_t.shape}\n")
                    else:
                        single_out = self.forward_single_step(states_probs_t[:, :, :, hor], adj_mat_t[:, :, hor], delta_t, normalize_adj_mat)
                        raise NotImplementedError(f"\nUse of future adjacency matrices is not implemented yet \n")
                else:
                    single_out = self.forward_single_step(states_probs_t[:, :, :, hor], adj_mat_t, delta_t, normalize_adj_mat)
                    out.append(single_out)
            out = torch.stack(out, axis=-1)

        return out



#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
# REPLAY BUFFER
class ReplayBuffer(nn.Module):
    def __init__(self, max_capacity: int, default_shape: Tuple[int, ...], device: Optional[Union[str, torch.device]] = 'cpu') -> None:
        """
        Replay buffer for retrieving sequences of data for temporal models.
        
        Args:
            max_capacity: Maximum number of entries per key
            default_shape: Default shape of each entry (without batch dimension)
            device: Device to store memory (default: 'cpu')
        """
        super().__init__()
        self.max_capacity = max_capacity
        self.default_shape = default_shape
        self.device = torch.device(device) if isinstance(device, str) else device
        self.memory = defaultdict(self._create_empty_buffer)
        self._current_sizes = defaultdict(int)  # Track sizes for faster roll operations

    def _create_empty_buffer(self) -> Tensor:
        """Default buffer data."""
        return torch.zeros((self.max_capacity, *self.default_shape), dtype=torch.float32, device=self.device)

    def to(self, device: Union[str, torch.device]) -> 'ReplayBuffer':
        """Move memory to specified device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        for key in list(self.memory.keys()):
            self.memory[key] = self.memory[key].to(self.device)
        return self

    def add(self, key: Union[str, int, Tensor], value: Tensor) -> None:
        """
        Add a new entry to memory.
        
        Args:
            key: Key to identify the memory entry
            value: Tensor to store (shape: *default_shape)
        """
        key = self._normalize_key(key)
        value = value.unsqueeze(0).to(self.device, non_blocking=True)
        
        current_size = self._current_sizes[key]
        buffer = self.memory[key]
        
        if current_size < self.max_capacity:
            buffer[current_size] = value
            self._current_sizes[key] = current_size + 1
        else:
            # Overwriting oldest entry
            buffer[:-1] = buffer[1:].clone()
            buffer[-1] = value

    def batch_add(self, keys: List[Union[str, int, Tensor]], values: Tensor) -> None:
        """
        Batch addition using vectorized operations where possible.
        
        Args:
            keys: List of keys
            values: Batch tensor of shape (B, *default_shape)
        """
        if len(keys) != len(values):
            raise ValueError(f"Keys length {len(keys)} doesn't match values length {len(values)}")
            
        values = values.unsqueeze(1) if values.ndim == len(self.default_shape) + 1 else values
        values = values.to(self.device, non_blocking=True)
        
        for key, value in zip(keys, values):
            self.add(key, value)

    def retrieve(self, key: Union[str, int, Tensor]) -> Tensor:
        """
        Retrieve memory entry with optional subsampling.
        
        Args:
            key: Key to access memory
            max_samples: If provided, returns random subsample of entries
            
        Returns:
            Tensor containing the stored values (empty if key doesn't exist)
        """
        key = self._normalize_key(key)            
        buffer = self.memory[key]
        current_size = self._current_sizes[key] + 1
        data = buffer[:current_size]
        return data

    def batch_retrieve(self, 
                      keys: List[Union[str, int, Tensor]], 
                      padding_value: float = 0) -> Tuple[Tensor, List[int]]:
        """
        Batch retrieval with padding.
        
        Args:
            keys: List of keys to retrieve
            padding_value: Value for padding shorter sequences
            
        Returns:
            Tuple of (padded_batch, original_lengths)
        """
        memories = []
        lengths = []
        
        for key in keys:
            mem = self.retrieve(key)
            memories.append(mem)
            lengths.append(len(mem))

        if not memories:
            empty = torch.zeros((0, *self.default_shape), device=self.device)
            return empty.unsqueeze(1), [0]
            
        padded = torch.nn.utils.rnn.pad_sequence(memories, batch_first=True, padding_value=padding_value)
        return padded, lengths

    def clear(self, key: Optional[Union[str, int, Tensor]] = None) -> None:
        """Clear specific key or entire memory."""
        if key is not None:
            key = self._normalize_key(key)
            self.memory.pop(key, None)
            self._current_sizes.pop(key, None)
        else:
            self.memory.clear()
            self._current_sizes.clear()

    def detach(self) -> None:
        """Detach all tensors in memory."""
        for key in list(self.memory.keys()):
            self.memory[key] = self.memory[key].detach()

    def _normalize_key(self, key: Union[str, int, Tensor]) -> Union[str, int]:
        """Convert tensor keys to primitive types."""
        return key.item() if isinstance(key, Tensor) else key

    def __len__(self) -> int:
        """Return number of keys in buffer."""
        return len(self.memory)

    def size(self, key: Optional[Union[str, int, Tensor]] = None) -> int:
        """Return current size of buffer for specific key or total size."""
        if key is not None:
            key = self._normalize_key(key)
            return self._current_sizes.get(key, 0)
        return sum(self._current_sizes.values())


#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
# TEMPORAL MODULES
class AttentionAggregator(nn.Module):
    def __init__(self, in_channels, hidden_channels, memory_capacity, num_layers, heads, out_channels=None, dropout=0.5, metadata=None, device : Optional[str]='cpu'):
        """
        Sequence aggregator using attention mechanism.
        """
        super(AttentionAggregator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if  out_channels is not None else hidden_channels
        self.device = device
        self.hetero_mode = metadata is not None

        def init_module(init_fn):
            """Initialize module conditionally based on metadata."""
            return nn.ModuleDict({key: init_fn() for key in metadata[0]}) if self.hetero_mode else init_fn()
            
        self.memories = init_module(lambda: ReplayBuffer(memory_capacity, (in_channels,), device=device))
        self.query_proj = init_module(lambda: MLP(in_channels, hidden_channels, hidden_sizes=[], dropout=dropout))
        self.attn = init_module(lambda: TransformerEncoder(
            TransformerEncoderLayer(hidden_channels, heads, hidden_channels, dropout=dropout),
            num_layers
        ))
        self.out_layer = init_module(lambda: MLP(hidden_channels, self.out_channels, hidden_sizes=[], dropout=dropout))

        
        # cls token embedding
        self.cls = (
            nn.ParameterDict({key: nn.Parameter(torch.randn(1, 1, in_channels)) for key in metadata[0]})
            if self.hetero_mode
            else nn.Parameter(torch.randn(1, 1, in_channels))
        )

        self.to(device)

    def to(self, device):
        """Move model and memories to the specified device."""
        self.device = device
        if self.hetero_mode:
            for memory in self.memories.values():
                memory.to(device)
        else:
            self.memories.to(device)
        return super().to(device)

    @torch.no_grad()
    def clear(self):
        """Clears the memory buffers."""
        for memory in self.memories.values() if self.hetero_mode else [self.memories]:
            memory.clear()

    @torch.no_grad()
    def detach(self):
        """Detaches memory state from computation graph."""
        for memory in self.memories.values() if self.hetero_mode else [self.memories]:
            memory.detach()

    def _aggregate(self, previous, lengths, cls_token, query_proj, attn, out_layer):
        """Common logic for aggregation across homo- and hetero-mode."""
        # Prepend CLS token
        cls = cls_token.expand(previous.size(0), -1, -1)
        previous = torch.cat([cls, previous], dim=1)

        # Key Padding Mask
        lengths = torch.as_tensor(lengths, device=previous.device) + 1  # Account for CLS token
        mask = torch.arange(previous.size(1), device=previous.device).expand(lengths.size(0), -1) >= lengths.unsqueeze(1)

        # Query Projection
        query = query_proj(previous)

        # Attention Mechanism
        attn_output = attn(query, mask=mask, is_causal=True)  # (B, M, D)

        # Extract CLS Token Output
        return out_layer(attn_output[:, 0, :])

    def aggregate(self, x, ids):
        """Aggregates data with memory buffer."""
        self.memories.batch_add(ids, x)
        previous, lengths = self.memories.batch_retrieve(ids)  # (B, M, D)
        return self._aggregate(previous, lengths, self.cls, self.query_proj, self.attn, self.out_layer)

    def aggregate_hetero(self, x, ids):
        """Aggregates heterogeneous data with separate memory buffers."""
        out = {}
        for key, val in x.items():
            if x[key].size(0) == 0:
                out[key] = torch.empty((0, self.out_channels), device=x[key].device)
                continue

            # Retrieve memory entries
            self.memories[key].batch_add(ids[key], x[key])
            previous, lengths = self.memories[key].batch_retrieve(ids[key])

            # Aggregate
            out[key] = self._aggregate(previous, lengths, self.cls[key], self.query_proj[key], self.attn[key], self.out_layer[key])

        return out

    def forward(self, x, ids, edge_index, edge_attr, timestamps):
        """Forward pass."""
        return self.aggregate_hetero(x, ids) if self.hetero_mode else self.aggregate(x, ids)


class LSTMAggregator(nn.Module):
    def __init__(self, in_channels, hidden_channels, memory_capacity, num_layers, out_channels=None, dropout=0.5, metadata=None, device : Optional[str]='cpu'):
        """
        Sequence aggregator using Long Short-Term Memory (LSTM) cells.
        """
        super(LSTMAggregator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if  out_channels is not None else hidden_channels
        self.device = device
        self.hetero_mode = metadata is not None

        def init_module(init_fn):
            """Initialize module conditionally based on metadata."""
            return nn.ModuleDict({key: init_fn() for key in metadata[0]}) if self.hetero_mode else init_fn()
            
        self.memories = init_module(lambda: ReplayBuffer(memory_capacity, (in_channels,), device=device))
        self.lstm = init_module(lambda: LSTM(in_channels, hidden_channels, num_layers=num_layers, batch_first=True, dropout=dropout))
        self.out_layer = init_module(lambda: MLP(hidden_channels, self.out_channels, hidden_sizes=[], dropout=dropout))
        self.to(device)

    def to(self, device):
        """Move model and memories to the specified device."""
        self.device = device
        if self.hetero_mode:
            for memory in self.memories.values():
                memory.to(device)
        else:
            self.memories.to(device)
        return super().to(device)

    @torch.no_grad()
    def clear(self):
        """Clears the memory buffers."""
        for memory in self.memories.values() if self.hetero_mode else [self.memories]:
            memory.clear()

    @torch.no_grad()
    def detach(self):
        """Detaches memory state from computation graph."""
        for memory in self.memories.values() if self.hetero_mode else [self.memories]:
            memory.detach()

    def aggregate(self, x, ids):
        """Aggregates data with memory buffer."""
        self.memories.batch_add(ids, x)
        inputs, _ = self.memories.batch_retrieve(ids)
        x, _ = self.lstm(inputs)
        x = x[:, -1, :]
        x = self.out_layer(x)
        return x
    
    def aggregate_hetero(self, x, ids):
        """Aggregates heterogeneous data with separate memory buffers."""
        out = {}
        for key, val in x.items():

            if x[key].size(0) == 0:
                out[key] = torch.empty((0, self.out_channels)).to(x[key].device)
                continue

            # Retrieve memory entries
            self.memories[key].batch_add(ids[key], x[key])
            inputs, _ = self.memories[key].batch_retrieve(ids[key])
            
            # Aggregate
            out[key], _ = self.lstm[key](inputs)
            out[key] = out[key][:, -1, :]
            out[key] = self.out_layer[key](out[key])

        return out

    def forward(self, x, ids, edge_index, edge_attr, timestamps):
        """Forward pass."""
        return self.aggregate_hetero(x, ids) if self.hetero_mode else self.aggregate(x, ids)


#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
# SPATIAL MODULES
class GraphSageModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, act='ReLU', metadata=None, aggr='mean'):
        """
        Graph-based module using GraphSAGE.
        """
        super().__init__()

        self.hetero_mode = metadata is not None

        # Input projection layers (necessary for heterogeneous graphs where different types of nodes have different dimensions)
        #if (self.hetero_mode):
        if (isinstance(in_channels, dict)):
            self.in_proj_layer = nn.ModuleDict({
                                        node_type: nn.Linear(in_channels[node_type], hidden_channels)
                                        for node_type in in_channels
                                    })
        else:
            self.in_proj_layer = nn.Linear(in_channels, hidden_channels)

        # GNN
        self.gnn = GraphSAGE(in_channels=hidden_channels, hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout, act=act)
        if metadata is not None:
            self.gnn = to_hetero(self.gnn, metadata, aggr=aggr)

        # Output projection layers
        def init_module(init_fn):
            if metadata is not None:
                return nn.ModuleDict({key: init_fn() for key in metadata[0]})
            else:
                return init_fn()
            
        self.out_layers = init_module(lambda: MLP(hidden_channels, out_channels, hidden_sizes=[], dropout=dropout))
        

    @torch.no_grad()
    def clear(self):
        """Clears the memory buffers. (Placeholder)"""
        pass

    @torch.no_grad()
    def detach(self):
        """Detaches memory state from computation graph. (Placeholder)"""
        pass

    def forward(self, x, ids, edge_index, edge_attr, timestamps):
        """Forward pass."""
        # TRANSFORM edge_indices USING ids TO GET THE CORRECT INDICES WITHIN THE NODES IN THE CURRENT GRAPH
        # FOR EXAMPLE IF edge_indices[('Patient', 'to', 'Patient')] = [ [107, 574, 264], [744, 54, 9] ] it means that in the current graph
        # WE ONLY HAVE PATIENT NODES 9, 54, 107, 264, 574, 744, so ids['Patient'] can be [264, 744, 107, 9, 54, 574] (not necessarily in 
        # increasing order, it can be another order, they correspond to the ids of the nodes in xs). IN THAT CASE, WE NEED
        # TO REMAP edge_indices[('Patient', 'to', 'Patient')] TO HAVE INDICES FROM 0 TO LEN(xs.shape[0])-1 BECAUSE THAT IS WHAT PYTORCH GEOMETRIC NEEDS.
        # IN THAT CASE, remapped_edge_index[('Patient', 'to', 'Patient')] = [ [2, 5, 0], [1, 4, 3] ] BECAUSE 107 => 2, 574 => 5, 264 => 0, 744 => 1, 54 => 4, 9 => 3
        remapped_edge_index = deepcopy(edge_index)
        if (self.hetero_mode):
            node_ids_mapping = {}
            for key_nodes in ids:
                node_ids_mapping[key_nodes] = {int(ids[key_nodes][i]): i for i in range(len(ids[key_nodes]))}
            for key_edges in remapped_edge_index:
                node_1_type = key_edges[0]
                node_2_type = key_edges[2]
                for tmp_edge_ID in range(remapped_edge_index[key_edges].shape[1]):
                    remapped_edge_index[key_edges][0, tmp_edge_ID] = node_ids_mapping[node_1_type][ int(remapped_edge_index[key_edges][0, tmp_edge_ID]) ]
                    remapped_edge_index[key_edges][1, tmp_edge_ID] = node_ids_mapping[node_2_type][ int(remapped_edge_index[key_edges][1, tmp_edge_ID]) ]
        else:
            pass

        # Projecting the features of the different types of nodes into spaces of the same dimension
        if (isinstance(x, dict)):
            if (isinstance(self.in_proj_layer, nn.ModuleDict)):
                x = {
                        node_type: self.in_proj_layer[node_type](x[node_type])
                        for node_type in x
                    }
            else:
                x = {
                        node_type: self.in_proj_layer(x[node_type])
                        for node_type in x
                    }
        else:
            x = self.in_proj_layer(x)

        # Computing output
        out =  self.gnn(x, remapped_edge_index, edge_attr)

        if self.hetero_mode:
            for key in out.keys():
                out[key] = self.out_layers[key](out[key])
        else:
            out = self.out_layers(out)

        return out
  

class GATModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout, num_layers, act='ReLU', metadata=None, aggr='mean', add_self_loops=False):
        """
        Graph-based module using GraphSAGE.
        """
        super().__init__()

        self.hetero_mode = metadata is not None

        # Input projection layers (necessary for heterogeneous graphs where different types of nodes have different dimensions)
        if (isinstance(in_channels, dict)):
            self.in_proj_layer = nn.ModuleDict({
                                        node_type: nn.Linear(in_channels[node_type], hidden_channels)
                                        for node_type in in_channels
                                    })
        else:
            self.in_proj_layer = nn.Linear(in_channels, hidden_channels)

        # GNN
        self.gnn = GAT(in_channels=hidden_channels, hidden_channels=hidden_channels, heads=heads, dropout=dropout, num_layers=num_layers, act=act, add_self_loops=add_self_loops)
        if metadata is not None:
            self.gnn = to_hetero(self.gnn, metadata, aggr=aggr)

        # Output projection layers
        def init_module(init_fn):
            if metadata is not None:
                return nn.ModuleDict({key: init_fn() for key in metadata[0]})
            else:
                return init_fn()
            
        self.out_layers = init_module(lambda: MLP(hidden_channels, out_channels, hidden_sizes=[], dropout=dropout))


    @torch.no_grad()
    def clear(self):
        """Clears the memory buffers. (Placeholder)"""
        pass

    @torch.no_grad()
    def detach(self):
        """Detaches memory state from computation graph. (Placeholder)"""
        pass

    def forward(self, x, ids, edge_index, edge_attr, timestamps):
        """Forward pass."""
        # TRANSFORM edge_indices USING ids TO GET THE CORRECT INDICES WITHIN THE NODES IN THE CURRENT GRAPH
        # FOR EXAMPLE IF edge_indices[('Patient', 'to', 'Patient')] = [ [107, 574, 264], [744, 54, 9] ] it means that in the current graph
        # WE ONLY HAVE PATIENT NODES 9, 54, 107, 264, 574, 744, so ids['Patient'] can be [264, 744, 107, 9, 54, 574] (not necessarily in 
        # increasing order, it can be another order, they correspond to the ids of the nodes in xs). IN THAT CASE, WE NEED
        # TO REMAP edge_indices[('Patient', 'to', 'Patient')] TO HAVE INDICES FROM 0 TO LEN(xs.shape[0])-1 BECAUSE THAT IS WHAT PYTORCH GEOMETRIC NEEDS.
        # IN THAT CASE, remapped_edge_index[('Patient', 'to', 'Patient')] = [ [2, 5, 0], [1, 4, 3] ] BECAUSE 107 => 2, 574 => 5, 264 => 0, 744 => 1, 54 => 4, 9 => 3
        remapped_edge_index = deepcopy(edge_index)
        if (self.hetero_mode):
            node_ids_mapping = {}
            for key_nodes in ids:
                node_ids_mapping[key_nodes] = {int(ids[key_nodes][i]): i for i in range(len(ids[key_nodes]))}
            for key_edges in remapped_edge_index:
                node_1_type = key_edges[0]
                node_2_type = key_edges[2]
                for tmp_edge_ID in range(remapped_edge_index[key_edges].shape[1]):
                    remapped_edge_index[key_edges][0, tmp_edge_ID] = node_ids_mapping[node_1_type][ int(remapped_edge_index[key_edges][0, tmp_edge_ID]) ]
                    remapped_edge_index[key_edges][1, tmp_edge_ID] = node_ids_mapping[node_2_type][ int(remapped_edge_index[key_edges][1, tmp_edge_ID]) ]
        else:
            pass

        # Projecting the features of the different types of nodes into spaces of the same dimension
        if (isinstance(x, dict)):
            if (isinstance(self.in_proj_layer, nn.ModuleDict)):
                x = {
                        node_type: self.in_proj_layer[node_type](x[node_type])
                        for node_type in x
                    }
            else:
                x = {
                        node_type: self.in_proj_layer(x[node_type])
                        for node_type in x
                    }
        else:
            x = self.in_proj_layer(x)

        # Computing output
        out =  self.gnn(x, remapped_edge_index, edge_attr)

        if self.hetero_mode:
            for key in out.keys():
                out[key] = self.out_layers[key](out[key])
        else:
            out = self.out_layers(out)

        return out


class GCNModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, act='ReLU', metadata=None, aggr='mean'):
        """
        Graph-based module using GCN.
        """
        super().__init__()

        self.hetero_mode = metadata is not None

        # Input projection layers (necessary for heterogeneous graphs where different types of nodes have different dimensions)
        if (isinstance(in_channels, dict)):
            self.in_proj_layer = nn.ModuleDict({
                                        node_type: nn.Linear(in_channels[node_type], hidden_channels)
                                        for node_type in in_channels
                                    })
        else:
            self.in_proj_layer = nn.Linear(in_channels, hidden_channels)

        # GNN
        self.gnn = GCN(in_channels=hidden_channels, hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout, act=act)
        if metadata is not None:
            self.gnn = to_hetero(self.gnn, metadata, aggr=aggr)

        # Output projection layers
        def init_module(init_fn):
            if metadata is not None:
                return nn.ModuleDict({key: init_fn() for key in metadata[0]})
            else:
                return init_fn()
            
        self.out_layers = init_module(lambda: MLP(hidden_channels, out_channels, hidden_sizes=[], dropout=dropout))
        

    @torch.no_grad()
    def clear(self):
        """Clears the memory buffers. (Placeholder)"""
        pass

    @torch.no_grad()
    def detach(self):
        """Detaches memory state from computation graph. (Placeholder)"""
        pass

    def forward(self, x, ids, edge_index, edge_attr, timestamps):
        """Forward pass."""
        # TRANSFORM edge_indices USING ids TO GET THE CORRECT INDICES WITHIN THE NODES IN THE CURRENT GRAPH
        # FOR EXAMPLE IF edge_indices[('Patient', 'to', 'Patient')] = [ [107, 574, 264], [744, 54, 9] ] it means that in the current graph
        # WE ONLY HAVE PATIENT NODES 9, 54, 107, 264, 574, 744, so ids['Patient'] can be [264, 744, 107, 9, 54, 574] (not necessarily in 
        # increasing order, it can be another order, they correspond to the ids of the nodes in xs). IN THAT CASE, WE NEED
        # TO REMAP edge_indices[('Patient', 'to', 'Patient')] TO HAVE INDICES FROM 0 TO LEN(xs.shape[0])-1 BECAUSE THAT IS WHAT PYTORCH GEOMETRIC NEEDS.
        # IN THAT CASE, remapped_edge_index[('Patient', 'to', 'Patient')] = [ [2, 5, 0], [1, 4, 3] ] BECAUSE 107 => 2, 574 => 5, 264 => 0, 744 => 1, 54 => 4, 9 => 3
        remapped_edge_index = deepcopy(edge_index)
        if (self.hetero_mode):
            node_ids_mapping = {}
            for key_nodes in ids:
                node_ids_mapping[key_nodes] = {int(ids[key_nodes][i]): i for i in range(len(ids[key_nodes]))}
            for key_edges in remapped_edge_index:
                node_1_type = key_edges[0]
                node_2_type = key_edges[2]
                for tmp_edge_ID in range(remapped_edge_index[key_edges].shape[1]):
                    remapped_edge_index[key_edges][0, tmp_edge_ID] = node_ids_mapping[node_1_type][ int(remapped_edge_index[key_edges][0, tmp_edge_ID]) ]
                    remapped_edge_index[key_edges][1, tmp_edge_ID] = node_ids_mapping[node_2_type][ int(remapped_edge_index[key_edges][1, tmp_edge_ID]) ]
        else:
            pass

        # Projecting the features of the different types of nodes into spaces of the same dimension
        if (isinstance(x, dict)):
            if (isinstance(self.in_proj_layer, nn.ModuleDict)):
                x = {
                        node_type: self.in_proj_layer[node_type](x[node_type])
                        for node_type in x
                    }
            else:
                x = {
                        node_type: self.in_proj_layer(x[node_type])
                        for node_type in x
                    }
        else:
            x = self.in_proj_layer(x)

        # Computing output
        out =  self.gnn(x, remapped_edge_index, edge_attr)

        if self.hetero_mode:
            for key in out.keys():
                out[key] = self.out_layers[key](out[key])
        else:
            out = self.out_layers(out)

        return out


# Model wrapper
class ModelWrapper(torch.nn.Module):
    def __init__(self, model_encoder, model_predictor):
        super().__init__()
        self.encoder = model_encoder
        self.predictor = model_predictor

    def forward(self, x, edge_index=None, edge_attr=None, nodes_ids=None, timestamps=None):
        if (nodes_ids is not None) and (timestamps is not None):
            enc = self.encoder(x, edge_index, edge_attr, nodes_ids, timestamps)
        else:
            if (edge_index is None) and (edge_attr is None) and (nodes_ids is None) and (timestamps is None): # MLP
                enc = self.encoder(x)
            else:
                enc = self.encoder(x, edge_index, edge_attr)
        return self.predictor(enc) 


# Model weights initialization function
def init_weights(m):
    # Standard linear layers
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # GATConv (Graph Attention Network)
    elif isinstance(m, GATConv):
        if hasattr(m, 'lin_l'):
            nn.init.xavier_uniform_(m.lin_l.weight)
            if m.lin_l.bias is not None:
                nn.init.zeros_(m.lin_l.bias)
        if hasattr(m, 'lin_r'):
            nn.init.xavier_uniform_(m.lin_r.weight)
            if m.lin_r.bias is not None:
                nn.init.zeros_(m.lin_r.bias)
        if hasattr(m, 'att_src'):
            nn.init.xavier_uniform_(m.att_src)
        if hasattr(m, 'att_dst'):
            nn.init.xavier_uniform_(m.att_dst)

    # GCNConv
    elif isinstance(m, GCNConv):
        torch.nn.init.xavier_uniform_(m.lin.weight)
        if m.lin.bias is not None:
            torch.nn.init.zeros_(m.lin.bias)

    # SAGEConv
    elif isinstance(m, SAGEConv):
        torch.nn.init.xavier_uniform_(m.lin_l.weight)
        if m.lin_r is not None:
            torch.nn.init.xavier_uniform_(m.lin_r.weight)

    # MultiheadAttention
    elif isinstance(m, nn.MultiheadAttention):
        nn.init.xavier_uniform_(m.in_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)

    # Transformer Encoder Layer
    elif isinstance(m, nn.TransformerEncoderLayer):
        for pname, param in m.named_parameters():
            if ("norm" not in pname):
                if 'weight' in pname:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in pname:
                    nn.init.zeros_(param)

    # Transformer Encoder (just apply recursively)
    elif isinstance(m, nn.TransformerEncoder):
        for layer in m.layers:
            layer.apply(init_weights)


class SimpleGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metadata=None):
        super().__init__()
        self.hetero_mode = metadata is not None
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        if (self.hetero_mode):
            self.conv1 = to_hetero(self.conv1, metadata)
            self.conv2 = to_hetero(self.conv1, metadata)
            

    def forward(self, x, edge_index):
        out = self.conv1(x, edge_index).relu()
        out = self.conv2(out, edge_index)
        
        return out
    
# Simple predictor
class SEIRPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEIRPredictor, self).__init__()
        self.recurrent = EvolveGCNO(in_channels)
        self.linear = Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        h = self.recurrent(x, edge_index, edge_attr)
        out = self.linear(F.relu(h))
        return out
# Simple predictor
class SEIRPredictorDeeper(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=6):
        super(SEIRPredictorDeeper, self).__init__()
        self.recurrent_1 = EvolveGCNO(in_channels)
        self.recurrent_2 = EvolveGCNO(in_channels)
        self.recurrent_3 = EvolveGCNO(in_channels)
        self.recurrent_4 = EvolveGCNO(in_channels)
        self.recurrent_5 = EvolveGCNO(in_channels)
        self.linear_1 = Linear(in_channels, hidden_dim)
        self.linear_2 = Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index, edge_attr):
        h = self.recurrent_1(x, edge_index, edge_attr)
        h = self.recurrent_2(F.relu(h), edge_index, edge_attr)
        h = self.recurrent_3(F.relu(h), edge_index, edge_attr)
        h = self.recurrent_4(F.relu(h), edge_index, edge_attr)
        h = self.recurrent_5(F.relu(h), edge_index, edge_attr)
        h = self.linear_1(F.relu(h))
        out = self.linear_2(h)
        return out

