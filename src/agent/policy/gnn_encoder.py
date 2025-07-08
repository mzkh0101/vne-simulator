# src/agent/policy/gnn_encoder.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool


def get_conv_layer(conv_type, input_dim, hidden_dim):
    if conv_type == "gcn":
        return GCNConv(input_dim, hidden_dim)
    elif conv_type == "gat":
        return GATConv(input_dim, hidden_dim, heads=1, concat=False)
    elif conv_type == "gin":
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        return GINConv(mlp)
    else:
        raise ValueError(f"Unknown GNN conv_type: {conv_type}")


class GNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        num_layers=2,
        conv_type="gcn",
        use_global_pool=False
    ):
        super().__init__()
        self.use_global_pool = use_global_pool

        self.convs = nn.ModuleList()
        self.convs.append(get_conv_layer(conv_type, input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(get_conv_layer(conv_type, hidden_dim, hidden_dim))

        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = self.relu(conv(x, edge_index))

        if self.use_global_pool:
            batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = global_mean_pool(x, batch)  # [B, hidden_dim]

        return x
