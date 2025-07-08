# src/strategy/base_strategy.py

from abc import ABC, abstractmethod
import networkx as nx


class EmbeddingStrategy(ABC):
    @abstractmethod
    def embed(self, substrate: nx.Graph, vnr: nx.Graph) -> tuple[bool, dict]:
        """
        埋め込み戦略の共通インターフェース

        Returns:
            success (bool): 埋め込み成功したか
            info (dict): 任意の付加情報（例: ノードマッピング）
        """
        pass
