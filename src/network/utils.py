import torch
from torch_geometric.data import Data
import networkx as nx


def nx_to_pyg(graph: nx.Graph, is_vnr: bool = False) -> Data:
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    if is_vnr:
        cpu_attr = [graph.nodes[n]["cpu_demand"] for n in graph.nodes]
        edge_attr = [[graph.edges[u, v]["bw_demand"]] for u, v in graph.edges]
    else:
        cpu_attr = [graph.nodes[n]["cpu"] for n in graph.nodes]
        edge_attr = [[graph.edges[u, v]["bw"]] for u, v in graph.edges]

    x = torch.tensor(cpu_attr, dtype=torch.float).view(-1, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
