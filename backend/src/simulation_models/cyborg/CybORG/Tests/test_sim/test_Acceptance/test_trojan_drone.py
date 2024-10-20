import pytest

from simulation_models.cyborg.CybORG import CybORG
from simulation_models.cyborg.CybORG.Agents import SleepAgent
from simulation_models.cyborg.CybORG.Simulator.Scenarios import DroneSwarmScenarioGenerator


def test_red_spawns():
    sg = DroneSwarmScenarioGenerator(num_drones=25, starting_num_red=0, red_internal_only=False)
    cyborg = CybORG(scenario_generator=sg, seed=123)

    assert not any(['red' in i for i in cyborg.active_agents])
    for i in range(100):
        cyborg.step()
    assert any(['red' in i for i in cyborg.active_agents])
