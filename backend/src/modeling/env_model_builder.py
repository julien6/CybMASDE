import copy
import functools
import json
from typing import Dict, Union
import gymnasium
import numpy as np
import random
from gym.spaces import Discrete, MultiDiscrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium.utils import EzPickle
from PIL import Image
from pprint import pprint
from pettingzoo.utils.conversions import parallel_wrapper_fn, to_parallel
from movingcompany.env.renderer import GridRenderer
from pettingzoo.utils.wrappers import BaseWrapper

FPS = 20

__all__ = ["env", "parallel_env", "raw_env"]


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class AECEnvBuilder:

    aecEnv

    def __ini__t(self):
        pass

    def set_observation_transition_function(self, otf):
        pass

    def set_reward_function(self, rf):
        pass

    def set_constraints(self, cm):
        pass


class AECEnv(Generic[AgentID, ObsType, ActionType]):
    """The AECEnv steps agents one at a time.

    If you are unsure if you have implemented a AECEnv correctly, try running
    the `api_test` documented in the Developer documentation on the website.
    """

    metadata: dict[str, Any]  # Metadata for the environment

    # All agents that may appear in the environment
    possible_agents: list[AgentID]
    agents: list[AgentID]  # Agents active at any given time

    observation_spaces: dict[
        AgentID, gymnasium.spaces.Space
    ]  # Observation space for each agent
    # Action space for each agent
    action_spaces: dict[AgentID, gymnasium.spaces.Space]

    # Whether each agent has just reached a terminal state
    terminations: dict[AgentID, bool]
    truncations: dict[AgentID, bool]
    rewards: dict[AgentID, float]  # Reward from the last step for each agent
    # Cumulative rewards for each agent
    _cumulative_rewards: dict[AgentID, float]
    infos: dict[
        AgentID, dict[str, Any]
    ]  # Additional information from the last step for each agent

    agent_selection: AgentID  # The agent currently being stepped

    def __init__(self):
        pass

    def step(self, action: ActionType) -> None:
        """Accepts and executes the action of the current agent_selection in the environment.

        Automatically switches control to the next agent.
        """
        raise NotImplementedError

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        """Resets the environment to a starting state."""
        raise NotImplementedError

    # TODO: Remove `Optional` type below
    def observe(self, agent: AgentID) -> ObsType | None:
        """Returns the observation an agent currently can make.

        `last()` calls this function.
        """
        raise NotImplementedError

    def render(self) -> None | np.ndarray | str | list:
        """Renders the environment as specified by self.render_mode.

        Render mode can be `human` to display a window.
        Other render modes in the default environments are `'rgb_array'`
        which returns a numpy array and is supported by all environments outside of classic,
        and `'ansi'` which returns the strings printed (specific to classic environments).
        """
        raise NotImplementedError

    def state(self) -> np.ndarray:
        """State returns a global view of the environment.

        It is appropriate for centralized training decentralized execution methods like QMIX
        """
        raise NotImplementedError(
            "state() method has not been implemented in the environment {}.".format(
                self.metadata.get("name", self.__class__.__name__)
            )
        )

    def close(self):
        """Closes any resources that should be released.

        Closes the rendering window, subprocesses, network connections,
        or any other resources that should be released.
        """
        pass

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        """
        warnings.warn(
            "Your environment should override the observation_space function. Attempting to use the observation_spaces dict attribute."
        )
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        """
        warnings.warn(
            "Your environment should override the action_space function. Attempting to use the action_spaces dict attribute."
        )
        return self.action_spaces[agent]

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)

    def _deads_step_first(self) -> AgentID:
        """Makes .agent_selection point to first terminated agent.

        Stores old value of agent_selection so that _was_dead_step can restore the variable after the dead agent steps.
        """
        _deads_order = [
            agent
            for agent in self.agents
            if (self.terminations[agent] or self.truncations[agent])
        ]
        if _deads_order:
            self._skip_agent_selection = self.agent_selection
            self.agent_selection = _deads_order[0]
        return self.agent_selection

    def _clear_rewards(self) -> None:
        """Clears all items in .rewards."""
        for agent in self.rewards:
            self.rewards[agent] = 0

    def _accumulate_rewards(self) -> None:
        """Adds .rewards dictionary to ._cumulative_rewards dictionary.

        Typically called near the end of a step() method
        """
        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

    def agent_iter(self, max_iter: int = 2**63) -> AECIterable:
        """Yields the current agent (self.agent_selection).

        Needs to be used in a loop where you step() each iteration.
        """
        return AECIterable(self, max_iter)

    def last(
        self, observe: bool = True
    ) -> tuple[ObsType | None, float, bool, bool, dict[str, Any]]:
        """Returns observation, cumulative reward, terminated, truncated, info for the current agent (specified by self.agent_selection)."""
        agent = self.agent_selection
        assert agent is not None
        observation = self.observe(agent) if observe else None
        return (
            observation,
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def _was_dead_step(self, action: ActionType) -> None:
        """Helper function that performs step() for dead agents.

        Does the following:

        1. Removes dead agent from .agents, .terminations, .truncations, .rewards, ._cumulative_rewards, and .infos
        2. Loads next agent into .agent_selection: if another agent is dead, loads that one, otherwise load next live agent
        3. Clear the rewards dict

        Examples:
            Highly recommended to use at the beginning of step as follows:

        def step(self, action):
            if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
                self._was_dead_step()
                return
            # main contents of step
        """
        if action is not None:
            raise ValueError(
                "when an agent is dead, the only valid action is None")

        # removes dead agent
        agent = self.agent_selection
        assert (
            self.terminations[agent] or self.truncations[agent]
        ), "an agent that was not dead as attempted to be removed"
        del self.terminations[agent]
        del self.truncations[agent]
        del self.rewards[agent]
        del self._cumulative_rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)

        # finds next dead agent or loads next live agent (Stored in _skip_agent_selection)
        _deads_order = [
            agent
            for agent in self.agents
            if (self.terminations[agent] or self.truncations[agent])
        ]
        if _deads_order:
            if getattr(self, "_skip_agent_selection", None) is None:
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = _deads_order[0]
        else:
            if getattr(self, "_skip_agent_selection", None) is not None:
                assert self._skip_agent_selection is not None
                self.agent_selection = self._skip_agent_selection
            self._skip_agent_selection = None
        self._clear_rewards()

    def __str__(self) -> str:
        """Returns a name which looks like: `space_invaders_v1`."""
        if hasattr(self, "metadata"):
            return self.metadata.get("name", self.__class__.__name__)
        else:
            return self.__class__.__name__

    @property
    def unwrapped(self) -> AECEnv[AgentID, ObsType, ActionType]:
        return self


class raw_env(AECEnv, EzPickle):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "moving_company_v0",
        "render_modes": ["human", "rgb_array", "grid"],
        "is_parallelizable": True,
        "render_fps": FPS,
        "has_manual_policy": True,
    }

    def __init__(self, size: int = 6, seed: int = 42, max_cycles: int = 30, render_mode=None):
        """The init method takes in environment arguments.
        The environment is a sizexsize grid representing two towers
        separated by distance equal to their height.
        3 agents are spawned randomly in the towers or in the space
        seperating the two towers.
        A package is located at the top of the first tower.
        Goals: Agents have to bring it to the top of the second tower
        the fastest way as possible.

        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """

        self.possible_agents = [f"agent_{i}" for i in range(3)]
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.size = size
        self._seed = seed
        self.max_cycles = max_cycles
        self._best_reward = 0
        self.init_grid_environment(seed)
        self.render_mode = render_mode
        self.renderer = GridRenderer(self.size)

        EzPickle.__init__(
            self,
            size=size,
            seed=seed,
            max_cycles=max_cycles,
            render_mode=render_mode
        )

        self.observation_spaces = {agent: MultiDiscrete(
            [6] * 3**2, seed=self._seed) for agent in self.possible_agents}

        self.action_spaces = {agent: Discrete(
            7, seed=self._seed) for agent in self.possible_agents}

        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations: {agent: False for agent in self.possible_agents}

    def init_grid_environment(self, seed: int):
        self.grid = np.ones((self.size, self.size), dtype=np.int64)
        for i in range(0, self.size):
            for j in range(0, self.size):
                if i == 0 or j == 0 or i == self.size - 1 or j == self.size - 1 \
                        or (i > 1 and i < self.size - 2 and j > 1 and j < self.size - 2) \
                        or (i == 1 and 1 < j and j < self.size - 2):
                    self.grid[i][j] = 0

        self.grid[1][1] = 5  # Setting the package in initial position
        self.grid[1][self.size - 2] = 4
        self.grid[self.size - 2][1] = 4
        self.grid[self.size - 2][self.size - 2] = 4

        agents_counter = len(self.possible_agents)
        agent_condition_positions = [
            (None, 1), (self.size-2, None), (None, self.size-2)]
        self.agents_position = {agent: (None, None)
                                for agent in self.possible_agents}
        while (agents_counter > 0):
            for i in range(1, self.size-1):
                for j in range(1, self.size-1):
                    if self.grid[i][j] == 1:
                        random.seed(seed)
                        if (random.random() > 0.5):
                            if agents_counter > 0:
                                ic, jc = agent_condition_positions[-1]
                                if (ic is not None and i == ic) or (jc is not None and j == jc):
                                    self.agents_position[f"agent_{agents_counter-1}"] = (
                                        i, j)
                                    self.grid[i][j] = 2
                                    agents_counter -= 1
                                    agent_condition_positions.pop()
                                    continue
        self.best_trajectory = []
        for i in range(1, self.size - 1):
            self.best_trajectory += [(i, 1)]
        for j in range(2, self.size - 1):
            self.best_trajectory += [(self.size - 2, j)]
        for i in range(1, self.size - 2):
            self.best_trajectory += [(self.size - 2 - i, self.size - 2)]

    def apply_action(self, agent_name: str, action: int):
        agent_position = self.agents_position[agent_name]

        if action == 0:
            return

        action -= 1

        if 0 <= action and action <= 3:
            direction = self.directions[action]
            targeted_cell_pos = (
                agent_position[0] + direction[0], agent_position[1] + direction[1])
            # move up, move down, move left, move right
            if self.grid[targeted_cell_pos] == 1:
                agent_cell = self.grid[agent_position]
                self.grid[agent_position] = 1
                self.grid[targeted_cell_pos] = agent_cell
                self.agents_position[agent_name] = targeted_cell_pos

        else:
            cross_surrouding_cells = [self.grid[agent_position[0]+direction[0]]
                                      [agent_position[1]+direction[1]] for direction in self.directions]

            # take package
            if action == 4:
                if 5 in cross_surrouding_cells:
                    dir = self.directions[cross_surrouding_cells.index(5)]
                    package_cell_pos = (
                        agent_position[0] + dir[0], agent_position[1] + dir[1])
                    self.grid[package_cell_pos] = 4
                    self.grid[agent_position] = 3

            # drop package
            if action == 5:
                if 4 in cross_surrouding_cells and self.grid[agent_position] == 3:
                    dir = self.directions[cross_surrouding_cells.index(4)]
                    dropzone_cell_pos = (
                        agent_position[0] + dir[0], agent_position[1] + dir[1])
                    self.grid[agent_position] = 2
                    self.grid[dropzone_cell_pos] = 5

    def generate_action_masks(self, agent_name: str):

        action_mask = np.zeros(self.action_space(agent_name).n, dtype=np.int8)

        for action in range(self.action_space(agent_name).n):

            if action == 0:
                action_mask[0] = 1

            elif action in range(1, 4):
                agent_position = self.agents_position[agent_name]

                direction = self.directions[action - 1]
                targeted_cell_pos = (
                    agent_position[0] + direction[0], agent_position[1] + direction[1])

                # move up, move down, move left, move right
                if self.grid[targeted_cell_pos] == 1:
                    action_mask[action] = 1

            else:
                cross_surrouding_cells = [self.grid[agent_position[0]+direction[0]]
                                          [agent_position[1]+direction[1]] for direction in self.directions]

                # take package
                if action == 5:
                    if 5 in cross_surrouding_cells:
                        action_mask[action] = 1

                # drop package
                if action == 6:
                    if 4 in cross_surrouding_cells and self.grid[agent_position] == 3:
                        action_mask[action] = 1

        return action_mask

    def check_terminated(self) -> bool:
        return self.grid[1][-2] == 5

    def compute_reward(self) -> float:

        package_pos = None
        for i in range(0, self.size):
            for j in range(0, self.size):
                if self.grid[i][j] in [3, 5]:
                    package_pos = (i, j)
                    break
            if package_pos is not None:
                break

        for i, pos in enumerate(self.best_trajectory):
            progress_counter = i
            if package_pos == pos:
                break

        progress_difference = progress_counter - self._best_reward
        if (progress_difference > 0):
            self._best_reward = progress_counter
            return progress_counter ** 2

        return progress_difference ** 2

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # An agent sees the neighboring cells (3X3 grid):
        # [ ][ ][ ]
        # [ ][X][ ]
        # [ ][ ][ ]
        # Each cell has 6 possible states: Wall (0), Empty (1), Agent (2), Agent+Package (3), EmptyPackageZone (4), NonEmptyPackageZone (5)
        return MultiDiscrete([6] * 3**2, seed=self._seed)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # An has 6 actions: nothing (0), move up (1), move down (2), move left (3), move right (4), take package (5), drop package (6)
        return Discrete(7, seed=self._seed)

    def compute_pixel_image(self):
        """
        This method should return a pixel representation of the environment.
        """

        pass

    def render(self, mode="human") -> np.ndarray:
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.  
        """

        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.render_mode == "human":
            # Display pyGame window
            print(self.grid)

        if self.render_mode == "grid":
            return {"grid": self.grid, "agents_position": self.agents_position}

        if self.render_mode == "rgb_array":
            # Generate an image
            return self.renderer.render_grid_frame(self.grid, self.agents_position)

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        agent_pos = self.agents_position[agent]
        observation = [0] * (3**2)
        for i, di in enumerate([-1, 0, 1]):
            for j, dj in enumerate([-1, 0, 1]):
                observation[i * 3 + j] = self.grid[agent_pos[0] +
                                                   di][agent_pos[1]+dj]

        return np.array(observation, dtype=np.int64)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        self.renderer.close()

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """

        self.init_grid_environment(
            seed=seed if seed is not None else self._seed)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"action_masks": self.generate_action_masks(
            agent)} for agent in self.agents}
        self.observations = {agent: self.observe(
            agent) for agent in self.agents}
        self.num_cycle = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """

        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step()
            return

        agent = self.agent_selection

        self.apply_action(agent, action)

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            common_reward = self.compute_reward()
            # rewards for all agents are placed in the .rewards dictionary
            for ag in self.agents:
                self.rewards[ag] = common_reward

            self.num_cycle += 1
            if self.num_cycle >= self.max_cycles:
                for ag in self.agents:
                    self.terminations[ag] = True

            if self.check_terminated():
                for ag in self.agents:
                    self.terminations[ag] = True

        elif self._agent_selector.is_first():
            for ag in self.agents:
                self.rewards[ag] = 0

        # observe the current state and generate action masks
        for ag in self.agents:
            self.observations[ag] = self.observe(ag)
            self.infos[agent] = {
                "action_masks": self.generate_action_masks(agent)}

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
