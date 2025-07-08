# src/agent/policy/action_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeMappingHead(nn.Module):
    """
    GNNからのノード表現を受け取り、各VNRノードに対して
    Substrateノードのマッピング先を確率分布で出力する層。
    """

    def __init__(self, hidden_dim, substrate_num_nodes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.substrate_num_nodes = substrate_num_nodes

        # 各VNRノードが substrate_num_nodes 個の候補から選ぶためのスコア計算
        self.classifier = nn.Linear(hidden_dim, substrate_num_nodes)

    def forward(self, vnr_node_embeddings):
        """
        Args:
            vnr_node_embeddings: Tensor [V, hidden_dim]
                V は VNRノード数

        Returns:
            logits: Tensor [V, substrate_num_nodes]
            probs: Tensor [V, substrate_num_nodes]
        """
        logits = self.classifier(vnr_node_embeddings)  # [V, N]
        probs = F.softmax(logits, dim=-1)
        return logits, probs
