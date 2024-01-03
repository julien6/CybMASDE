from typing import Dict, List, Any


class SimulationModel:

    def __init__(self, scenario_model: Any, max_episode: int = 1, max_iteration: int = 1) -> None:

        self.max_eps = max_episode
        self.max_its = max_iteration

        self.current_ep = 0
        self.current_it = 0

    def next(self, requested_info: List = ["episode_number", "iteration_number", "observation_spaces", "agents_actions",
                                           "agents_observations", "agents_rewards", "true_states"]) -> Dict:

        raise NotImplementedError()
