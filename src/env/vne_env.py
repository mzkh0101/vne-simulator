# src/env/vne_env.py

import copy


class VNEEnvironment:
    def __init__(self, config, substrate_graph, vnr_scheduler, embedding_strategy):
        self.config = config
        self.original_substrate = substrate_graph  # 保存用
        self.substrate = None
        self.vnr_scheduler = vnr_scheduler
        self.strategy = embedding_strategy
        self.current_step = 0
        self.max_steps = config["training"]["max_steps_per_episode"]
        self.accepted = 0
        self.total = 0

    def reset(self):
        self.substrate = copy.deepcopy(self.original_substrate)
        self.current_step = 0
        self.accepted = 0
        self.total = 0
        return self.observe()

    def step(self):
        vnrs = self.vnr_scheduler.generate_batch(
            vnr_size_range=self.config["env"]["vnr_num_nodes"],
            cpu_demand_range=self.config["env"]["vnr_cpu_demand_range"],
            bw_demand_range=self.config["env"]["vnr_bw_demand_range"],
        )

        rewards = 0
        for vnr in vnrs:
            success, _ = self.strategy.embed(self.substrate, vnr)
            rewards += 1 if success else 0
            self.total += 1
            self.accepted += int(success)

        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self.observe(), rewards, done, {}

    def observe(self):
        return self.substrate  # 学習用には別の変換が必要（PyGなど）

    def get_metrics(self):
        if self.total == 0:
            return {"acceptance_ratio": 0.0}
        return {"acceptance_ratio": self.accepted / self.total}
