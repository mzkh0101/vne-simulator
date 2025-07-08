# tests/test_gnn_encoder.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch_geometric.data import Data
from src.agent.policy.gnn_encoder import GNNEncoder


def test_gnn_encoder():
    print("✅ Start: GNNEncoder Test")

    # ダミーグラフ：3ノード、双方向エッジ
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    model = GNNEncoder(input_dim=1, hidden_dim=64, conv_type="gcn", use_global_pool=True)
    out = model(data)
    print(f"Output shape: {out.shape}")  # [1, 64] expected
    assert out.shape == (1, 64)

    print("✅ Passed GNNEncoder test.\n")


if __name__ == "__main__":
    test_gnn_encoder()
