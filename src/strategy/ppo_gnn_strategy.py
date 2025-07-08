# src/strategy/ppo_gnn_strategy.py

from src.strategy.base_strategy import EmbeddingStrategy
from src.agent.ppo import PPOAgent
from src.network.utils import nx_to_pyg
import torch
import copy
import networkx as nx


class PPOGNNStrategy(EmbeddingStrategy):
    def __init__(self, agent: PPOAgent, substrate_num_nodes: int, device="cuda"):
        self.agent = agent
        self.substrate_num_nodes = substrate_num_nodes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def embed(self, substrate_nx: nx.Graph, vnr_nx: nx.Graph) -> tuple[bool, dict]:
        # 1. PyGに変換
        vnr_data = nx_to_pyg(vnr_nx, is_vnr=True)
        vnr_data = vnr_data.to(self.device)

        # 2. 行動選択（ノードマッピング）
        action, log_prob, value = self.agent.select_action(vnr_data)
        action = action.tolist()

        # 3. ノードマッピング（無効チェックあり）
        substrate_copy = copy.deepcopy(substrate_nx)
        node_mapping = {}
        for vnr_node, substrate_node in enumerate(action):
            demand = vnr_nx.nodes[vnr_node]["cpu_demand"]
            available = substrate_copy.nodes[substrate_node]["cpu"]
            if substrate_node in node_mapping.values():
                return False, {}  # 重複マッピング禁止（1:1制約）
            if available >= demand:
                node_mapping[vnr_node] = substrate_node
                substrate_copy.nodes[substrate_node]["cpu"] -= demand
            else:
                return False, {}

        # 4. リンクマッピング（Greedy：最短経路 + BW制約）
        used_links = set()
        for (u, v) in vnr_nx.edges:
            su = node_mapping[u]
            sv = node_mapping[v]
            bw_demand = vnr_nx.edges[u, v]["bw_demand"]
            try:
                path = nx.shortest_path(substrate_copy, source=su, target=sv)
            except nx.NetworkXNoPath:
                return False, {}

            # 経路の帯域がすべて要求を満たすか確認
            valid = True
            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
                if substrate_copy[s][t]["bw"] < bw_demand:
                    valid = False
                    break
            if not valid:
                return False, {}

            # リンク資源を予約
            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
                substrate_copy[s][t]["bw"] -= bw_demand
                used_links.add((s, t))

        # 5. 実際に substrate に反映
        for v_node, s_node in node_mapping.items():
            substrate_nx.nodes[s_node]["cpu"] -= vnr_nx.nodes[v_node]["cpu_demand"]
        for (s, t) in used_links:
            substrate_nx[s][t]["bw"] -= vnr_nx.edges[0, 1]["bw_demand"]  # 仮：リンク全て同じbw_demand（正確にはVNRエッジ単位で保持する）

        # 6. 学習バッファ保存
        self.agent.store_transition(vnr_data, torch.tensor(action), log_prob, reward=1.0, done=False, value=value)

        return True, {
            "node_mapping": node_mapping,
            "used_links": list(used_links)
        }
