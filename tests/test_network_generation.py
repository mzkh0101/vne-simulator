# tests/test_network_generation.py

from src.network.substrate_generator import SubstrateGenerator
from src.network.vnr_generator import VNRGenerator
from src.network.vnr_scheduler import PoissonVNRGenerator
from src.network.utils import nx_to_pyg


def test_network_generation():
    print("✅ Start: Network Generation Test")

    # Substrate
    sub_gen = SubstrateGenerator(seed=42)
    substrate = sub_gen.generate(
        num_nodes=100,
        m=3,
        cpu_range=(100, 200),
        bw_range=(50, 100)
    )
    assert substrate.number_of_nodes() == 100
    print(f"Substrate nodes: {substrate.number_of_nodes()}, edges: {substrate.number_of_edges()}")

    # VNR
    vnr_gen = VNRGenerator(seed=42)
    scheduler = PoissonVNRGenerator(vnr_generator=vnr_gen, lam=3)
    vnrs = scheduler.generate_batch(
        vnr_size_range=(5, 15),
        cpu_demand_range=(10, 30),
        bw_demand_range=(5, 20)
    )
    print(f"Generated {len(vnrs)} VNRs in this step")

    # PyG変換テスト
    substrate_pyg = nx_to_pyg(substrate)
    vnr_pyg_list = [nx_to_pyg(vnr, is_vnr=True) for vnr in vnrs]

    assert substrate_pyg.x.shape[0] == substrate.number_of_nodes()
    print(f"Substrate PyG node shape: {substrate_pyg.x.shape}")

    if vnr_pyg_list:
        print(f"First VNR PyG node shape: {vnr_pyg_list[0].x.shape}")

    print("✅ Passed all tests.\n")


if __name__ == "__main__":
    test_network_generation()
