import networkx as nx
import random


class VNRGenerator:
    def __init__(self, seed: int = 42, alpha: float = 0.4, beta: float = 0.1):
        self.rng = random.Random(seed)
        self.alpha = alpha
        self.beta = beta

    def generate(self, num_nodes: int, cpu_demand_range: tuple, bw_demand_range: tuple) -> nx.Graph:
        G = nx.waxman_graph(n=num_nodes, alpha=self.alpha, beta=self.beta, domain=(0, 0, 1, 1))

        while not nx.is_connected(G):
            G = nx.waxman_graph(n=num_nodes, alpha=self.alpha, beta=self.beta, domain=(0, 0, 1, 1))

        for node in G.nodes:
            G.nodes[node]["cpu_demand"] = self.rng.randint(*cpu_demand_range)

        for u, v in G.edges:
            G.edges[u, v]["bw_demand"] = self.rng.randint(*bw_demand_range)

        return G
