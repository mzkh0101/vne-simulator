import networkx as nx
import random


class SubstrateGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate(self, num_nodes: int, m: int, cpu_range: tuple, bw_range: tuple) -> nx.Graph:
        G = nx.barabasi_albert_graph(n=num_nodes, m=m)

        for node in G.nodes:
            G.nodes[node]["cpu"] = self.rng.randint(*cpu_range)

        for u, v in G.edges:
            G.edges[u, v]["bw"] = self.rng.randint(*bw_range)

        return G
