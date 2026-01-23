import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, NNConv, RGCNConv, RGATConv, GraphNorm
from egnn_pytorch import EGNN_Network
from torch.nn import Linear, ReLU, BatchNorm1d
from torch_geometric.nn.norm import BatchNorm, LayerNorm

from regnn_ensemble import REGCLEnsemble


class RGAT_affinity(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations, edge_dim, depth, dropout):
        super().__init__()
        self.depth = depth
        self.dropout = dropout
        self.act = nn.GELU()  # GELU activation

        # GNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()  # For residual projection

        self.pooling_layer = global_mean_pool

        # First layer
        self.layers.append(RGATConv(in_channels, hidden_channels, num_relations, edge_dim=edge_dim))
        self.norms.append(GraphNorm(hidden_channels))
        self.res_projs.append(nn.Linear(in_channels, hidden_channels))  # Project input for residual

        fc_hidden_channel = hidden_channels * 2

        # Hidden layers
        for _ in range(depth - 1):
            self.layers.append(RGATConv(hidden_channels, hidden_channels, num_relations, edge_dim=edge_dim))
            self.norms.append(GraphNorm(hidden_channels))
            self.res_projs.append(nn.Identity())  # No projection needed for hidden→hidden

        # Final MLP for scalar prediction
        self.fc_out = nn.Sequential(
            nn.Linear(fc_hidden_channel, fc_hidden_channel),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_channel, 1)  # output is a single scalar
        )

    def forward(self, x, edge_index, edge_type, edge_attr, batch):
        indices = x[:, -1].clone().long()
        # Pass through RGAT layers
        for conv, norm, res_proj in zip(self.layers, self.norms, self.res_projs):
            x_save = x
            x = conv(x, edge_index, edge_type, edge_attr)
            x = norm(x, batch)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + res_proj(x_save)  # Residual

        # Pooling
        x_prot = self.pooling_layer(x[indices == 0], batch=batch[indices == 0])
        x_lig = self.pooling_layer(x[indices == 1], batch=batch[indices == 1])

        # Concatenate protein and ligand representations
        x = torch.cat([x_prot, x_lig], dim=1)

        # Predict scalar binding affinity
        return self.fc_out(x).squeeze(-1)  # shape: [num_graphs]
    


class RGCN_affinity(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations, depth, dropout):
        super().__init__()
        self.depth = depth
        self.dropout = dropout
        self.act = nn.GELU()

        # GNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        self.pooling_layer = global_mean_pool

        # First layer
        self.layers.append(RGCNConv(in_channels, hidden_channels, num_relations))
        self.norms.append(GraphNorm(hidden_channels))
        self.res_projs.append(nn.Linear(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(depth - 1):
            self.layers.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
            self.norms.append(GraphNorm(hidden_channels))
            self.res_projs.append(nn.Identity())

        fc_hidden_channel = hidden_channels * 2

        # Final MLP for scalar regression
        self.fc_out = nn.Sequential(
            nn.Linear(fc_hidden_channel, fc_hidden_channel),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_channel, 1)  # single scalar
        )

    def forward(self, x, edge_index, edge_type, batch):
        indices = x[:, -1].clone().long()
        # Pass through GNN layers
        for conv, norm, res_proj in zip(self.layers, self.norms, self.res_projs):
            x_save = x
            x = conv(x, edge_index, edge_type)
            x = norm(x, batch)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + res_proj(x_save)

        # Pooling
        x_prot = self.pooling_layer(x[indices == 0], batch=batch[indices == 0])
        x_lig = self.pooling_layer(x[indices == 1], batch=batch[indices == 1])

        # Concatenate protein and ligand representations
        x = torch.cat([x_prot, x_lig], dim=1)

        # Predict scalar binding affinity
        return self.fc_out(x).squeeze(-1)  # [num_graphs]
    





class RGCN_affinity_bn(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations, depth, dropout):
        super().__init__()
        self.depth = depth
        self.dropout = dropout
        self.act = nn.GELU()

        # GNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        self.pooling_layer = global_mean_pool

        # First layer
        self.layers.append(RGCNConv(in_channels, hidden_channels, num_relations))
        self.norms.append(nn.BatchNorm1d(hidden_channels))
        self.res_projs.append(nn.Linear(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(depth - 1):
            self.layers.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.res_projs.append(nn.Identity())

        fc_hidden_channel = hidden_channels * 2

        # Final MLP for scalar regression
        self.fc_out = nn.Sequential(
            nn.Linear(fc_hidden_channel, fc_hidden_channel),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_channel, 1)
        )

    def forward(self, x, edge_index, edge_type, batch):
        indices = x[:, -1].clone().long()

        # Pass through GNN layers
        for conv, norm, res_proj in zip(self.layers, self.norms, self.res_projs):
            x_save = x
            x = conv(x, edge_index, edge_type)
            x = norm(x)  # BatchNorm1d does not use `batch`
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + res_proj(x_save)

        # Pooling
        x_prot = self.pooling_layer(x[indices == 0], batch=batch[indices == 0])
        x_lig = self.pooling_layer(x[indices == 1], batch=batch[indices == 1])

        # Concatenate protein and ligand representations
        x = torch.cat([x_prot, x_lig], dim=1)

        # Predict scalar binding affinity
        return self.fc_out(x).squeeze(-1)
    



class RGCN_affinity_no_residual(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations, depth, dropout):
        super().__init__()
        self.depth = depth
        self.dropout = dropout
        self.act = nn.GELU()

        # GNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.pooling_layer = global_mean_pool

        # First layer
        self.layers.append(RGCNConv(in_channels, hidden_channels, num_relations))
        self.norms.append(GraphNorm(hidden_channels))

        # Hidden layers
        for _ in range(depth - 1):
            self.layers.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
            self.norms.append(GraphNorm(hidden_channels))

        fc_hidden_channel = hidden_channels * 2

        # Final MLP for scalar regression
        self.fc_out = nn.Sequential(
            nn.Linear(fc_hidden_channel, fc_hidden_channel),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_channel, 1)  # single scalar
        )

    def forward(self, x, edge_index, edge_type, batch):
        indices = x[:, -1].clone().long()
        # Pass through GNN layers
        for conv, norm in zip(self.layers, self.norms):
            x = conv(x, edge_index, edge_type)
            x = norm(x, batch)  # GraphNorm needs batch
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling
        x_prot = self.pooling_layer(x[indices == 0], batch=batch[indices == 0])
        x_lig = self.pooling_layer(x[indices == 1], batch=batch[indices == 1])

        # Concatenate protein and ligand representations
        x = torch.cat([x_prot, x_lig], dim=1)

        # Predict scalar binding affinity
        return self.fc_out(x).squeeze(-1)
    



class RGAT_affinity_no_residual(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations, edge_dim, depth, dropout):
        super().__init__()
        self.depth = depth
        self.dropout = dropout
        self.act = nn.GELU()

        # GNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.pooling_layer = global_mean_pool

        # First layer
        self.layers.append(RGATConv(in_channels, hidden_channels, num_relations, edge_dim=edge_dim))
        self.norms.append(GraphNorm(hidden_channels))

        # Hidden layers
        for _ in range(depth - 1):
            self.layers.append(RGATConv(hidden_channels, hidden_channels, num_relations, edge_dim=edge_dim))
            self.norms.append(GraphNorm(hidden_channels))

        fc_hidden_channel = hidden_channels * 2

        # Final MLP for scalar regression
        self.fc_out = nn.Sequential(
            nn.Linear(fc_hidden_channel, fc_hidden_channel),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_channel, 1)  # single scalar
        )

    def forward(self, x, edge_index, edge_type, edge_attr, batch):
        indices = x[:, -1].clone().long()
        # Pass through GNN layers
        for conv, norm in zip(self.layers, self.norms):
            x = conv(x, edge_index, edge_type, edge_attr)
            x = norm(x, batch)  # GraphNorm needs batch
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling
        x_prot = self.pooling_layer(x[indices == 0], batch=batch[indices == 0])
        x_lig = self.pooling_layer(x[indices == 1], batch=batch[indices == 1])

        # Concatenate protein and ligand representations
        x = torch.cat([x_prot, x_lig], dim=1)

        # Predict scalar binding affinity
        return self.fc_out(x).squeeze(-1)
    




class New_RGCN_complex_affinity(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_relations, depth, dropout):
        super().__init__()
        

        """
        Initialization for RGAT model.

        in_dim: int - The dimensionality of input node features.
        out_dim: int - The dimensionality of output (default: 1).
        hidden_dim: int - The number of hidden units in the hidden layers.
        lr: float - learning rate for the optimizer.
        num_relations: int - The number of relation types.
        heads: int - Number of attention heads.
        edge_dim: int - Edge feature dimensionality (if using edge_attr).
        num_gnn_layers: int - Number of GNN layers.
        num_fc_layers: int - Number of fully connected layers.
        concat: bool - Whether to concatenate the heads.
        """

        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.num_relations = num_relations
        # self.heads = heads

        # self.concat = concat
        self.use_edge_attr = False #use_edge_attr=False because currently rgcnconv does not support edge_attr
        self.dropout = dropout
        # self.gnn_dropout = gnn_dropout

        # Calculate the effective hidden dimension considering heads and concat
        # effective_hidden_dim = hidden_dim * heads if concat else hidden_dim
        effective_hidden_dim = hidden_channels
        # RGAT layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Input layer
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations=num_relations, root_weight=False, aggr='mean'))
        self.bns.append(BatchNorm(effective_hidden_dim))

        # Hidden layers
        for _ in range(depth - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations, root_weight=False, aggr='mean'))

            self.bns.append(BatchNorm(effective_hidden_dim))

        # Fully connected layers
        self.fcs = torch.nn.ModuleList()
        fc_input_dim = effective_hidden_dim * 2  # Concatenated protein and ligand representations

        for _ in range(2 - 1):
            self.fcs.append(Linear(fc_input_dim, fc_input_dim))
            # self.fcs.append(ReLU())

        # Output layer
        self.fcs.append(Linear(fc_input_dim, 1))

        # Non-linear activation
        self.relu = ReLU()

        self.dropout_layer = torch.nn.Dropout(p=self.dropout)

        self.pooling_layer = global_mean_pool

    def forward(self, x, edge_index, edge_type, batch):
        """
        Forward pass for the RGAT model for a single complex.

        batch: Batch - The batch containing all the necessary attributes.
        """
        indices = x[:, -1].clone().long()

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_type=edge_type)
            x = F.relu(x)
            if self.training:
                x = self.dropout_layer(x)
            x = bn(x)

        # Pooling
        x_prot = self.pooling_layer(x[indices == 0], batch=batch[indices == 0])
        x_lig = self.pooling_layer(x[indices == 1], batch=batch[indices == 1])

        # Concatenate protein and ligand representations
        out = torch.cat([x_prot, x_lig], dim=1)

        # Fully connected layers
        for layer in self.fcs[:-1]:
            out = layer(out)
            out = self.relu(out)
        # Output layer
        out = self.fcs[-1](out).squeeze(-1)

        return out
    



class RGAT_complex_affinity(nn.Module):
    """
    Relational Graph Attention Network model class based on RGATConv for predicting single complex -logKd/Ki.
    """

    def __init__(self, in_channels, hidden_channels, num_relations, edge_dim, depth, dropout):

        """
        Initialization for RGAT model.

        in_dim: int - The dimensionality of input node features.
        out_dim: int - The dimensionality of output (default: 1).
        hidden_dim: int - The number of hidden units in the hidden layers.
        lr: float - learning rate for the optimizer.
        num_relations: int - The number of relation types.
        heads: int - Number of attention heads.
        edge_dim: int - Edge feature dimensionality (if using edge_attr).
        num_gnn_layers: int - Number of GNN layers.
        num_fc_layers: int - Number of fully connected layers.
        concat: bool - Whether to concatenate the heads.
        """
        super(RGAT_complex_affinity, self).__init__()

        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.num_relations = num_relations

        self.dropout = dropout

        # RGAT layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Input layer

        self.convs.append(RGATConv(in_channels, hidden_channels, num_relations=num_relations, edge_dim=edge_dim))
        
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(depth - 1):
            self.convs.append(RGATConv(hidden_channels, hidden_channels, num_relations=num_relations, edge_dim=edge_dim))
            self.bns.append(BatchNorm(hidden_channels))

        # Fully connected layers
        self.fcs = torch.nn.ModuleList()
        fc_input_dim = hidden_channels * 2  # Concatenated protein and ligand representations

        for _ in range(2 - 1):
            self.fcs.append(Linear(fc_input_dim, fc_input_dim))
            # self.fcs.append(ReLU())

        # Output layer
        self.fcs.append(Linear(fc_input_dim, 1))

        # Non-linear activation
        self.relu = ReLU()

        self.dropout_layer = torch.nn.Dropout(p=self.dropout)

        self.pooling_layer = global_mean_pool

    def forward(self, x, edge_index, edge_type, edge_attr, batch):
        """
        Forward pass for the RGAT model for a single complex.

        batch: Batch - The batch containing all the necessary attributes.
        """
        indices = x[:, -1].clone().long()

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_type=edge_type, edge_attr=edge_attr)
            x = F.relu(x)
            if self.training:
                x = self.dropout_layer(x)
            x = bn(x)

        # Pooling
        x_prot = self.pooling_layer(x[indices == 0], batch=batch[indices == 0])
        x_lig = self.pooling_layer(x[indices == 1], batch=batch[indices == 1])

        # Concatenate protein and ligand representations
        out = torch.cat([x_prot, x_lig], dim=1)

        # Fully connected layers
        for layer in self.fcs[:-1]:
            out = layer(out)
            out = self.relu(out)
        # Output layer
        out = self.fcs[-1](out).squeeze(-1)

        return out
    






class REGNN_complex_affinity(nn.Module):
    """
    Equivariant Relational Graph Neural Network for predicting protein-ligand binding affinity.
    Uses REGCLEnsemble to maintain equivariance while handling multiple relation types.
    """

    def __init__(self, in_channels, hidden_channels, num_relations,
                 edge_dim, depth, dropout, node_agg='sum', coords_agg='mean', attention=False,
                 normalize=False, tanh=False,
                 residual=True, update_coords=False):
        super(REGNN_complex_affinity, self).__init__()
        # Add this embedding layer
        self.embedding = Linear(in_channels, hidden_channels)
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        self.num_relations = num_relations
        self.dropout = dropout
        
        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(REGCLEnsemble(
            input_nf=hidden_channels,
            hidden_nf=hidden_channels,
            output_nf=hidden_channels,
            edges_in_d=edge_dim,
            num_relations=num_relations,
            act_fn=torch.nn.SiLU(),
            node_agg=node_agg,
            coords_agg=coords_agg,
            attention=attention,
            normalize=normalize,
            tanh=tanh,
            residual=residual,
            update_coords=update_coords
        ))
        
        self.bns.append(BatchNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(depth - 1):
            self.convs.append(REGCLEnsemble(
                input_nf=hidden_channels,
                hidden_nf=hidden_channels,
                output_nf=hidden_channels,
                edges_in_d=edge_dim,
                num_relations=num_relations,
                act_fn=torch.nn.SiLU(),
                node_agg=node_agg,
                coords_agg=coords_agg,
                attention=attention,
                normalize=normalize,
                tanh=tanh,
                residual=residual,
                update_coords=update_coords
            ))
            self.bns.append(BatchNorm(hidden_channels))
        
        # Rest of the code remains unchanged...
        # Fully connected layers
        self.fcs = torch.nn.ModuleList()
        fc_input_dim = hidden_channels * 2  # Concatenated protein and ligand representations

        # All hidden layers use the same width (fc_input_dim)
        for _ in range(2 - 1):
            self.fcs.append(Linear(fc_input_dim, fc_input_dim))

        # Only the output layer reduces to final dimension
        self.fcs.append(Linear(fc_input_dim, 1))
        
        # Non-linear activation
        self.relu = ReLU()
        
        # Dropout layer
        self.dropout_layer = torch.nn.Dropout(p=dropout)
        
        self.pooling_layer = global_mean_pool

    def forward(self, x, pos, edge_index, edge_type, edge_attr, batch):
        
        # Get protein/ligand indicator (last feature dimension)
        indices = x[:, -1].clone().long()
        # Apply embedding to input features
        x = self.embedding(x)
    
        # Process through GNN layers (use full feature vector)
        for conv, bn in zip(self.convs, self.bns):
            h, pos = conv(x, edge_index, pos, edge_attr, edge_type)
            h = F.relu(h)
            if self.training:
                h = self.dropout_layer(h)
            h = bn(h)
            x = h  # Update x for next layer
        
        # Pooling
        x_prot = self.pooling_layer(x[indices == 0], batch=batch[indices == 0])
        x_lig = self.pooling_layer(x[indices == 1], batch=batch[indices == 1])
        
        # Concatenate protein and ligand representations
        out = torch.cat([x_prot, x_lig], dim=1)
        
        # FC layers
        for i, layer in enumerate(self.fcs[:-1]):
            out = layer(out)
            out = self.relu(out)
            if self.training:
                out = self.dropout_layer(out)
        
        # Output layer
        out = self.fcs[-1](out).squeeze(-1)
        
        return out