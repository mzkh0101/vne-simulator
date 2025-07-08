# tests/test_env_with_ppo.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.vne_env import VNEEnvironment
from src.network.substrate_generator import SubstrateGenerator
from src.network.vnr_generator import VNRGenerator
from src.network.vnr_scheduler import PoissonVNRGenerator
from src.agent.policy.gnn_encoder import GNNEncoder
from src.agent.policy.action_head import NodeMappingHead
from src.agent.ppo import PPOAgent
from src.strategy.ppo_gnn_strategy import PPOGNNStrategy

import torch.nn as nn


def test_env_with_ppo():
    print("✅ Start: VNEEnvironment + PPOGNNStrategy Test")

    # --- 設定値 ---
    substrate_num_nodes = 50
    vnr_node_range = (5, 8)
    cpu_range = (100, 200)
    bw_range = (50, 100)
    cpu_demand_range = (10, 30)
    bw_demand_range = (5, 20)

    # --- ネットワーク生成 ---
    substrate = SubstrateGenerator().generate(
        num_nodes=substrate_num_nodes,
        m=2,
        cpu_range=cpu_range,
        bw_range=bw_range
    )

    vnr_gen = VNRGenerator()
    scheduler = PoissonVNRGenerator(vnr_gen, lam=3)

    # --- PPO構成 ---
    gnn_config = {
        "input_dim": 1,
        "hidden_dim": 64,
        "num_layers": 2,
        "conv_type": "gcn",
        "use_global_pool": False,
    }
    actor_encoder = GNNEncoder(**gnn_config)
    critic_encoder = GNNEncoder(**{**gnn_config, "use_global_pool": True})
    actor_head = NodeMappingHead(hidden_dim=64, substrate_num_nodes=substrate_num_nodes)
    critic_head = nn.Linear(64, 1)

    agent_config = {
        "lr": 1e-3,
        "gamma": 0.99,
        "clip_ratio": 0.2,
        "update_epochs": 3
    }

    agent = PPOAgent(actor_encoder, actor_head, critic_encoder, critic_head, agent_config)
    strategy = PPOGNNStrategy(agent, substrate_num_nodes=substrate_num_nodes)

    # --- 環境 ---
    config = {
        "env": {
            "vnr_num_nodes": vnr_node_range,
            "vnr_cpu_demand_range": cpu_demand_range,
            "vnr_bw_demand_range": bw_demand_range,
        },
        "training": {
            "max_steps_per_episode": 10
        }
    }

    env = VNEEnvironment(config, substrate, scheduler, strategy)
    obs = env.reset()

    for _ in range(config["training"]["max_steps_per_episode"]):
        obs, reward, done, _ = env.step()
        print(f"Step reward: {reward}")
        if done:
            break

    metrics = env.get_metrics()
    print(f"Acceptance ratio: {metrics['acceptance_ratio']:.2f}")
    print("✅ Passed PPO strategy environment test.\n")


if __name__ == "__main__":
    test_env_with_ppo()
