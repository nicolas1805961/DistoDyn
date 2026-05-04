import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, NNConv, RGCNConv, RGATConv, GraphNorm
from egnn_pytorch import EGNN_Network
from torch.nn import Linear, ReLU, BatchNorm1d
from regnn_ensemble import REGCLEnsemble
from torch_geometric.nn.norm import BatchNorm


class RGATNodeClassifier(nn.Module):
    """
    Relational Graph Convolutional Network for node classification.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, edge_dim, depth, dropout):
        super(RGATNodeClassifier, self).__init__()
        self.num_relations = num_relations
        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = dropout

        # Calculate the effective hidden dimension considering heads and concat
        effective_hidden_dim = hidden_channels

        self.convs.append(RGATConv(
            in_channels, hidden_channels, num_relations=num_relations, heads=1,
            edge_dim=edge_dim, concat=False, dropout=0.0
        ))
        self.bns.append(BatchNorm(effective_hidden_dim))
 
        for _ in range(depth - 1):
            self.convs.append(RGATConv(
                effective_hidden_dim, hidden_channels, num_relations=num_relations, heads=1,
                edge_dim=edge_dim, concat=False, dropout=0.0
            ))
            self.bns.append(BatchNorm(effective_hidden_dim))

        # FC layers
        self.fcs = torch.nn.ModuleList()
        for _ in range(2 - 1):
            self.fcs.append(Linear(hidden_channels, hidden_channels))

        # Output layer
        self.fcs.append(Linear(hidden_channels, out_channels))

        self.dropout_layer = torch.nn.Dropout(p=self.dropout)

    def forward(self, x, edge_index, edge_type, edge_attr, batch):

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_type=edge_type, edge_attr=edge_attr)
            x = F.relu(x)
            if self.training:
                x = self.dropout_layer(x)
            x = bn(x)

        # FC layers
        for layer in self.fcs[:-1]:
            x = layer(x)         
            x = F.relu(x)
            if self.training:
                x = self.dropout_layer(x)

        out = self.fcs[-1](x)

        return out
    


class RGCNNodeClassifier(nn.Module):
    """
    Relational Graph Convolutional Network for node classification.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, depth, dropout):
        super().__init__()
        self.num_relations = num_relations
        self.dropout = dropout

        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Input layer
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations=num_relations, root_weight=True))
        self.bns.append(BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(depth - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations, root_weight=True))
            self.bns.append(BatchNorm1d(hidden_channels))

        # FC layers
        self.fcs = torch.nn.ModuleList()
        for _ in range(2 - 1):
            self.fcs.append(Linear(hidden_channels, hidden_channels))
            self.fcs.append(ReLU())
        # Output layer
        self.fcs.append(Linear(hidden_channels, out_channels))
        self.dropout_layer = torch.nn.Dropout(p=self.dropout)

    def forward(self, x, edge_index, edge_type, batch):
    
        # GNN layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = bn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # FC layers
        for layer in self.fcs[:-1]:
            x = layer(x)
            if self.training:
                x = self.dropout_layer(x)
        out = self.fcs[-1](x)

        return out


class GATModel(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=64, output_dim=1, num_layers=2, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.num_layers = num_layers
        self.heads = heads

        # First layer
        self.convs.append(
            GATv2Conv(
                in_channels=node_feat_dim,
                out_channels=hidden_dim,
                edge_dim=edge_feat_dim,
                heads=heads,
                concat=True,
                dropout=0.1
            )
        )

        # Additional layers
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim * heads,  # because concat=True multiplies hidden dim
                    out_channels=hidden_dim,
                    edge_dim=edge_feat_dim,
                    heads=heads,
                    concat=True,
                    dropout=0.1
                )
            )

        # Final linear layer
        self.fc_out = nn.Linear(hidden_dim * heads, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)

        # Global pooling to get graph-level embedding
        x = global_mean_pool(x, batch)
        return self.fc_out(x)




class NNConvModel(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=64, output_dim=1, num_layers=2, heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        hidden_total = hidden_dim * heads  # match GAT concat hidden size

        self.convs = nn.ModuleList()

        # Edge MLP for the first layer
        mlp1 = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feat_dim * hidden_total)  # edge_feat → weight matrix
        )
        self.convs.append(NNConv(node_feat_dim, hidden_total, mlp1, aggr='mean'))

        # Additional layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_total * hidden_total)
            )
            self.convs.append(NNConv(hidden_total, hidden_total, mlp, aggr='mean'))

        # Output layer
        self.fc_out = nn.Linear(hidden_total, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        return self.fc_out(x)
    



class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, edge_type)
        x = global_mean_pool(x, batch)
        return self.fc_out(x)
    


class RGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, edge_dim):
        super().__init__()
        self.conv1 = RGATConv(in_channels, hidden_channels, num_relations, edge_dim=edge_dim)
        self.conv2 = RGATConv(hidden_channels, hidden_channels, num_relations, edge_dim=edge_dim)
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type, edge_attr, batch):
        print(x.shape)
        print(edge_index.shape)
        print(edge_type.shape)
        print(edge_attr.shape)
        x = self.conv1(x, edge_index, edge_type, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_type, edge_attr).relu()
        x = global_mean_pool(x, batch)
        return self.fc_out(x)
    


class RGAT_bs(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, edge_dim, depth, dropout, return_attention=False):
        super().__init__()
        self.depth = depth
        self.dropout = dropout
        self.act = nn.GELU()  # GELU activation
        self.return_attention = return_attention

        # GNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()  # For residual projection

        # First layer
        self.layers.append(RGATConv(in_channels, hidden_channels, num_relations, edge_dim=edge_dim))
        self.norms.append(GraphNorm(hidden_channels))
        self.res_projs.append(nn.Linear(in_channels, hidden_channels))  # Project input for residual

        # Hidden layers
        for _ in range(depth - 1):
            self.layers.append(RGATConv(hidden_channels, hidden_channels, num_relations, edge_dim=edge_dim))
            self.norms.append(GraphNorm(hidden_channels))
            self.res_projs.append(nn.Identity())  # No projection needed for hidden→hidden

        # Final 2-layer MLP with dropout
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_type, edge_attr, batch):
        all_attn_weights = [] if self.return_attention else None

        for i, (conv, norm, res_proj) in enumerate(zip(self.layers, self.norms, self.res_projs)):
            x_save = x

            if self.return_attention:
                # RGATConv must support return_attention_weights
                x, attn_weights = conv(x, edge_index, edge_type, edge_attr, return_attention_weights=True)
                all_attn_weights.append(attn_weights)
            else:
                x = conv(x, edge_index, edge_type, edge_attr)

            x = norm(x, batch)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + res_proj(x_save)  # Add residual (with projection if needed)

        out = self.fc_out(x)

        if self.return_attention:
            return out, all_attn_weights
        else:
            return out
    



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

        # First layer
        self.layers.append(RGATConv(in_channels, hidden_channels, num_relations, edge_dim=edge_dim))
        self.norms.append(GraphNorm(hidden_channels))
        self.res_projs.append(nn.Linear(in_channels, hidden_channels))  # Project input for residual

        # Hidden layers
        for _ in range(depth - 1):
            self.layers.append(RGATConv(hidden_channels, hidden_channels, num_relations, edge_dim=edge_dim))
            self.norms.append(GraphNorm(hidden_channels))
            self.res_projs.append(nn.Identity())  # No projection needed for hidden→hidden

        # Final MLP for scalar prediction
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)  # output is a single scalar
        )

    def forward(self, x, edge_index, edge_type, edge_attr, batch):
        # Pass through RGAT layers
        for conv, norm, res_proj in zip(self.layers, self.norms, self.res_projs):
            x_save = x
            x = conv(x, edge_index, edge_type, edge_attr)
            x = norm(x, batch)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + res_proj(x_save)  # Residual

        # Pool node features to get graph-level representation
        x = global_mean_pool(x, batch)  # shape: [num_graphs, hidden_channels]

        # Predict scalar binding affinity
        return self.fc_out(x).squeeze(-1)  # shape: [num_graphs]
    


class RGCN_bs(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, depth, dropout):
        super().__init__()
        self.depth = depth
        self.dropout = dropout
        self.act = nn.GELU()  # GELU activation

        # GNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()  # For residual projection

        # First layer
        self.layers.append(RGCNConv(in_channels, hidden_channels, num_relations))
        self.norms.append(GraphNorm(hidden_channels))
        self.res_projs.append(nn.Linear(in_channels, hidden_channels))  # Project input for residual

        # Hidden layers
        for _ in range(depth - 1):
            self.layers.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
            self.norms.append(GraphNorm(hidden_channels))
            self.res_projs.append(nn.Identity())  # No projection needed for hidden→hidden

        # Final 2-layer MLP with dropout
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_type, batch):
        for i, (conv, norm, res_proj) in enumerate(zip(self.layers, self.norms, self.res_projs)):
            x_save = x
            x = conv(x, edge_index, edge_type)
            x = norm(x, batch)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + res_proj(x_save)  # Add residual (with projection if needed)
        return self.fc_out(x)
    




class RGCN_bs_paper(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, edge_dim, depth, dropout):
        super().__init__()
        self.depth = depth
        self.dropout = dropout
        self.act = nn.ReLU()  # ReLU activation

        # GNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()  # For residual projection

        # First layer
        self.layers.append(RGCNConv(in_channels, hidden_channels, num_relations))
        self.norms.append(BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(depth - 1):
            self.layers.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
            self.norms.append(BatchNorm1d(hidden_channels))

        # Final 2-layer MLP with dropout
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_type, batch=None):
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            x = conv(x, edge_index, edge_type)
            x = self.act(x)
            x = norm(x)
        return self.fc_out(x)

    




class Custom_EGNN(torch.nn.Module):
    def __init__(self, dim, depth, num_positions, edge_dim, num_tokens=None):
        super().__init__()
        self.model = EGNN_Network(
            num_tokens = num_tokens,
            num_positions = num_positions,           # unless what you are passing in is an unordered set, set this to the maximum sequence length
            dim = dim,
            depth = depth,
            edge_dim = edge_dim,
            num_nearest_neighbors = 8,
            coor_weights_clamp_value = 2.   # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
        )
        self.fc_out = nn.Linear(dim, 1)

    def forward(self, feats, coors, edges, mask, adj_mat=None):
        feats_out, coors_out = self.model(feats, coors, edges=edges, mask=mask, adj_mat=adj_mat)  # [B, L_max, dim]

        # Masked mean pooling over nodes
        mask = mask.unsqueeze(-1)  # [B, L_max, 1]
        feats_out = feats_out * mask.float()  # zero out padded positions
        pooled = feats_out.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)  # [B, dim]

        # Final linear layer to predict 1 value per protein
        out = self.fc_out(pooled)  # [B, 1]
        return out
    




class REGNNNodeClassifier(nn.Module):
    """
    Equivariant Relational Graph Neural Network for node classification.
    Uses REGCLEnsemble to maintain equivariance while handling multiple relation types.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,num_relations, edge_dim,
                 depth, dropout,
                 node_agg='sum', coords_agg='mean', attention=False, normalize=False,
                 tanh=False,
                 residual=True, update_coords=False):
        super(REGNNNodeClassifier, self).__init__()
        # Add this embedding layer
        self.embedding = Linear(in_channels, hidden_channels)
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
        
        # FC layers
        self.fcs = torch.nn.ModuleList()
        for i in range(2 - 1):
            self.fcs.append(Linear(hidden_channels, hidden_channels))
        
        # Output layer
        self.fcs.append(Linear(hidden_channels, out_channels))
        
        # Dropout layer
        self.dropout_layer = torch.nn.Dropout(p=dropout)
        
    def forward(self, x, pos, edge_index, edge_type, edge_attr, batch):
        
        # Apply embedding to input features
        x = self.embedding(x)
        
        # Process through GNN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h, pos = conv(x, edge_index, pos, edge_attr, edge_type)
            
            # Check for NaNs
            if torch.isnan(h).any():
                print(f"NaN detected after conv layer {i}")
                
                # Find which nodes have NaNs
                nan_nodes = torch.where(torch.isnan(h).any(dim=1))[0]
                print(f"Nodes with NaNs: {nan_nodes.tolist()}")
                
                # Find which graphs these nodes belong to
                if hasattr(batch, 'batch'):
                    problem_graphs = torch.unique(batch.batch[nan_nodes]).tolist()
                    print(f"Graphs with NaNs: {problem_graphs}")
                    
                    # If you have a batch.pdb_id list, you can print those too
                    if hasattr(batch, 'pdb_id') and len(batch.pdb_id) == len(torch.unique(batch.batch)):
                        problem_pdbs = [batch.pdb_id[i] for i in problem_graphs]
                        print(f"PDB IDs with NaNs: {problem_pdbs}")
                
                # Return dummy tensor to prevent further NaNs
                return torch.zeros((x.size(0), 2), device=x.device)  # Hard-coded out_dim=2
            
            h = F.relu(h)
            if self.training:
                h = self.dropout_layer(h)
            h = bn(h)
                
            x = h  # Update x for next layer
        
        # FC layers
        for layer in self.fcs[:-1]:
            x = layer(x)
            x = F.relu(x)
            if self.training:
                x = self.dropout_layer(x)
        
        # Check before final layer
        if torch.isnan(x).any():
            print("NaN detected before final layer")
            nan_nodes = torch.where(torch.isnan(x).any(dim=1))[0]
            print(f"Nodes with NaNs: {nan_nodes.tolist()}")
            if hasattr(batch, 'batch'):
                problem_graphs = torch.unique(batch.batch[nan_nodes]).tolist()
                print(f"Graphs with NaNs: {problem_graphs}")
            
        out = self.fcs[-1](x)
        return out