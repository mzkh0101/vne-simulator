# src/agent/policy/gnn_encoder.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, use_global_pool=False):
        super(GNNEncoder, self).__init__()
        self.use_global_pool = use_global_pool

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = self.relu(conv(x, edge_index))

        if self.use_global_pool:
            batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = global_mean_pool(x, batch)  # [B, hidden_dim]

        return x  # [N, hidden_dim] または [B, hidden_dim]
