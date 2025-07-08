import numpy as np
from .vnr_generator import VNRGenerator


class PoissonVNRGenerator:
    def __init__(self, vnr_generator: VNRGenerator, lam: float):
        self.vnr_generator = vnr_generator
        self.lam = lam

    def generate_batch(self, vnr_size_range: tuple, cpu_demand_range: tuple, bw_demand_range: tuple) -> list:
        batch = []
        num_vnrs = np.random.poisson(self.lam)
        for _ in range(num_vnrs):
            n_nodes = np.random.randint(*vnr_size_range)
            vnr = self.vnr_generator.generate(n_nodes, cpu_demand_range, bw_demand_range)
            batch.append(vnr)
        return batch
