import gym
from gym.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import wrappers
import numpy as np
from typing import Dict, Tuple


def env(grid_size=(10, 10), agents_number=3, view_size=3):
    """
    Factory function to create the Warehouse Management environment.
    Args:
        grid_size (tuple): Size of the grid (rows, cols).
        agents_number (int): Number of agents in the environment.
        view_size (int): Observation range of each agent.
    """
    environment = WarehouseManagementEnv(
        grid_size=grid_size, agents_number=agents_number, view_size=view_size)
    # return wrappers.CaptureStdoutWrapper(environment)
    return environment


class WarehouseManagementEnv(ParallelEnv):
    metadata = {"render.modes": ["human"], "name": "warehouse_management_v1"}

    def __init__(self, grid_size=(10, 10), agents_number=3, view_size=3, seed=42):
        """
        Initialize the Warehouse Management environment.
        Args:
            grid_size (tuple): Dimensions of the warehouse grid.
            agents_number (int): Number of agents in the environment.
            view_size (int): Observation range for each agent.
        """
        super().__init__()

        # Environment setup
        self.grid_size = grid_size
        self.agents_number = agents_number
        self.view_size = view_size

        # Agent and action spaces
        self.agents = [f"agent_{i}" for i in range(agents_number)]
        self.possible_agents = self.agents[:]
        # 0: noop, 1: up, 2: down, 3: left, 4: right, 5: pick, 6: drop
        self.action_space = {agent: Discrete(
            7, seed=seed) for agent in self.agents}
        self.observation_space = {agent: Box(
            0, 14, (view_size * 2 + 1, view_size * 2 + 1, 1), np.int32, seed=seed) for agent in self.agents}

        # Cell types (adjusted based on descriptions)
        self.EMPTY = 1
        self.OBSTACLE = 0
        self.AGENT_WITHOUT_OBJECT = 2
        self.AGENT_WITH_PRIMARY = 3
        self.AGENT_WITH_SECONDARY = 4
        self.PRIMARY_OBJECT = 5
        self.SECONDARY_OBJECT = 6
        self.EMPTY_INPUT = 7
        self.INPUT_WITH_OBJECT = 8
        self.EMPTY_INPUT_CRAFT = 9
        self.INPUT_CRAFT_WITH_OBJECT = 10
        self.EMPTY_OUTPUT_CRAFT = 11
        self.OUTPUT_CRAFT_WITH_OBJECT = 12
        self.EMPTY_OUTPUT = 13
        self.OUTPUT_WITH_OBJECT = 14

        self.reset()

    def reset(self):
        """
        Resets the environment to its initial state.
        """
        self._initialize_grid()
        self._initialize_agents()
        self.terminated = {agent: False for agent in self.agents}
        self.truncated = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        return self.observe()

    def _initialize_grid(self):
        """
        Create the initial grid layout with zones, obstacles, and objects.
        """
        self.grid = np.full(self.grid_size, self.EMPTY)

        # Define obstacles
        self.grid[4, 4] = self.OBSTACLE
        self.grid[5, 4] = self.OBSTACLE

        # Define input zones
        self.grid[2, 0] = self.EMPTY_INPUT
        self.grid[3, 0] = self.INPUT_WITH_OBJECT
        self.grid[4, 0] = self.INPUT_WITH_OBJECT
        self.grid[5, 0] = self.EMPTY_INPUT
        self.grid[6, 0] = self.INPUT_WITH_OBJECT
        self.grid[7, 0] = self.INPUT_WITH_OBJECT

        # Define craft zones
        self.grid[3, 4] = self.INPUT_CRAFT_WITH_OBJECT # EMPTY_INPUT_CRAFT
        self.grid[6, 4] = self.EMPTY_INPUT_CRAFT
        self.grid[4, 5] = self.EMPTY_OUTPUT_CRAFT
        self.grid[5, 5] = self.OUTPUT_CRAFT_WITH_OBJECT

        # Define output zones
        self.grid[4, 9] = self.EMPTY_OUTPUT
        self.grid[5, 9] = self.OUTPUT_CRAFT_WITH_OBJECT

    def _initialize_agents(self):
        """
        Place agents in their initial positions.
        """
        self.agent_positions = {
            agent: (i + 1, 1) for i, agent in enumerate(self.agents)
        }
        self.agent_states = {
            agent: self.AGENT_WITHOUT_OBJECT for agent in self.agents}

    def observe(self) -> Dict[str, np.ndarray]:
        """
        Return observations for all agents.
        Each observation is a grid slice centered around the agent.
        """
        observations = {}
        for agent in self.agents:
            x, y = self.agent_positions[agent]
            slice_x = slice(max(0, x - self.view_size),
                            min(self.grid_size[0], x + self.view_size + 1))
            slice_y = slice(max(0, y - self.view_size),
                            min(self.grid_size[1], y + self.view_size + 1))
            obs = self.grid[slice_x, slice_y]
            padded_obs = np.full(
                (self.view_size * 2 + 1, self.view_size * 2 + 1), self.EMPTY)
            padded_obs[: obs.shape[0], : obs.shape[1]] = obs
            observations[agent] = padded_obs[..., np.newaxis]
        return observations

    def step(self, actions: Dict[str, int]):
        """
        Apply the actions taken by all agents.
        Args:
            actions (dict): A dictionary mapping agents to their actions.
        """
        for agent, action in actions.items():
            if self.terminated[agent]:
                continue

            if action in [1, 2, 3, 4]:  # Movement
                self._move_agent(agent, action)
            elif action == 5:  # Pick up object
                self._pick_object(agent)
            elif action == 6:  # Drop object
                self._drop_object(agent)

        # Compute rewards and check terminations
        self._update_rewards()
        self.terminated = {agent: self._check_termination(
            agent) for agent in self.agents}

        return self.observe(), self.rewards, self.terminated, self.truncated

    def _move_agent(self, agent: str, action: int):
        """
        Moves an agent if the move is valid.
        Args:
            agent (str): Agent identifier.
            action (int): Movement action (1: up, 2: down, 3: left, 4: right).
        """
        x, y = self.agent_positions[agent]
        new_x, new_y = x, y
        if action == 1 and x > 0:  # Up
            new_x -= 1
        elif action == 2 and x < self.grid_size[0] - 1:  # Down
            new_x += 1
        elif action == 3 and y > 0:  # Left
            new_y -= 1
        elif action == 4 and y < self.grid_size[1] - 1:  # Right
            new_y += 1

        if self.grid[new_x, new_y] in [self.EMPTY] and (new_x, new_y) not in self.agent_positions.values():
            self.agent_positions[agent] = (new_x, new_y)

    def _pick_object(self, agent: str):
        """
        Handles picking up objects from valid cells.
        """
        x, y = self.agent_positions[agent]
        if self.grid[x, y] in [self.PRIMARY_OBJECT, self.INPUT_WITH_OBJECT]:
            self.agent_states[agent] = self.AGENT_WITH_PRIMARY
            self.grid[x, y] = self.EMPTY

    def _drop_object(self, agent: str):
        """
        Handles dropping objects onto valid cells.
        """
        x, y = self.agent_positions[agent]
        if self.agent_states[agent] == self.AGENT_WITH_PRIMARY:
            if self.grid[x, y] in [self.EMPTY_INPUT_CRAFT]:
                self.grid[x, y] = self.INPUT_CRAFT_WITH_OBJECT
                self.agent_states[agent] = self.AGENT_WITHOUT_OBJECT

    def _update_rewards(self):
        """
        Update rewards based on the state of the environment.
        """
        for agent in self.agents:
            self.rewards[agent] = 0  # Default reward
            x, y = self.agent_positions[agent]
            if self.grid[x, y] == self.OUTPUT_WITH_OBJECT:
                # Reward for delivering a secondary object
                self.rewards[agent] += 10

    def _check_termination(self, agent: str) -> bool:
        """
        Check whether an agent has completed its task.
        """
        return False  # No termination conditions yet.

    def render(self, mode="human"):
        """
        Render the current state of the grid.
        """
        grid = self.grid.copy()
        for agent, (x, y) in self.agent_positions.items():
            grid[x, y] = self.AGENT_WITHOUT_OBJECT
        print(grid)

    def close(self):
        """
        Close the environment.
        """
        pass


if __name__ == "__main__":

    warehouse_env = env(grid_size=(10, 10), agents_number=3, view_size=3)

    number_to_action = {
        0: "nothing",
        1: "up",
        2: "down",
        3: "left",
        4: "right",
        5: "pick up",
        6: "drop"
    }

    # Réinitialiser l'environnement
    observations = warehouse_env.reset()
    print("=== État initial de la grille ===")
    warehouse_env.render()

    for step in range(10):  # Effectuer 10 étapes
        print(f"\n=== Étape {step + 1} ===")

        # Actions aléatoires pour tous les agents
        actions = {agent: warehouse_env.action_space[agent].sample() for agent in warehouse_env.agents}
        print(f"Actions des agents: ", {
              agent: number_to_action[action] for agent, action in actions.items()})

        # Effectuer une étape
        observations, rewards, terminated, truncated = warehouse_env.step(
            actions)

        # Afficher la grille après l'étape
        warehouse_env.render()

        # Afficher les récompenses
        print(f"Récompenses: {rewards}")

        # Vérifier si tous les agents ont terminé
        if all(terminated.values()):
            print("\nTous les agents ont terminé leurs tâches.")
            break

    warehouse_env.close()
