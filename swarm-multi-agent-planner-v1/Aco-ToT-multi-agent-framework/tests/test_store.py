from __future__ import annotations

from pathlib import Path

from aco_tot.store import create_initial_pheromone_state, load_pheromone_state, save_pheromone_state


def test_save_and_load_roundtrip(tmp_path: Path):
    state = create_initial_pheromone_state(alpha=1.0, beta=2.0, epsilon=0.001, rho=0.1, seed=42)
    out = tmp_path / "pheromones.json"
    save_pheromone_state(out, state)
    loaded = load_pheromone_state(out)
    assert loaded.metadata["alpha"] == 1.0
    assert loaded.metadata["beta"] == 2.0
    assert loaded.metadata["rho"] == 0.1
    assert len(loaded.pheromones) == len(state.pheromones)
