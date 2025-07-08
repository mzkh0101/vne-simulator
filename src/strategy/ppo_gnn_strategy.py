# src/strategy/ppo_gnn_strategy.py

from src.strategy.base_strategy import EmbeddingStrategy
from src.agent.ppo import PPOAgent
from src.network.utils import nx_to_pyg
import torch
import copy


class PPOGNNStrategy(EmbeddingStrategy):
    def __init__(self, agent: PPOAgent, substrate_num_nodes: int, device="cuda"):
        self.agent = agent
        self.substrate_num_nodes = substrate_num_nodes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def embed(self, substrate_nx, vnr_nx):
        # NetworkX → PyG形式に変換
        vnr_data = nx_to_pyg(vnr_nx, is_vnr=True)
        vnr_data = vnr_data.to(self.device)

        # 行動選択（各VNRノードごとのマッピング先 substrate ノード）
        action, log_prob, value = self.agent.select_action(vnr_data)

        # Greedyと同様に substrate のコピーを使って埋め込みを試行
        substrate_copy = copy.deepcopy(substrate_nx)
        node_mapping = {}
        for vnr_node, substrate_node in enumerate(action.tolist()):
            demand = vnr_nx.nodes[vnr_node]["cpu_demand"]
            if substrate_copy.nodes[substrate_node]["cpu"] >= demand:
                node_mapping[vnr_node] = substrate_node
                substrate_copy.nodes[substrate_node]["cpu"] -= demand
            else:
                # 一つでも失敗したら全体を拒否（改良の余地あり）
                return False, {}

        # 成功 → 本体 substrate に反映（この層は外側が管理しても良い）
        for vnr_node, substrate_node in node_mapping.items():
            substrate_nx.nodes[substrate_node]["cpu"] -= vnr_nx.nodes[vnr_node]["cpu_demand"]

        # 学習用データを保存
        self.agent.store_transition(vnr_data, action, log_prob, reward=1.0, done=False, value=value)

        return True, {"node_mapping": node_mapping}
