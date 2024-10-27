import copy
import functools
import json
import gymnasium
import numpy as np
import random
import warnings

from typing import Any, Dict, Union
from gym.spaces import Discrete, MultiDiscrete
from pettingzoo.utils import agent_selector, wrappers
from gymnasium.utils import EzPickle
from PIL import Image
from pprint import pprint
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.env import AgentID, ObsDict, ActionDict


def createParallelEnv(metadata, agents, observation_spaces, action_spaces, terminations, truncations, rewards, infos, transition_function, reward_function, initial_state=None, options=None) -> 'ParallelEnv':

    class CustomParallelEnv(ParallelEnv):
        """Parallel environment class.

        It steps every live agent at once. If you are unsure if you
        have implemented a ParallelEnv correctly, try running the `parallel_api_test` in
        the Developer documentation on the website.
        """

        def __init__(self, **options):
            """
            The init method takes in environment arguments and should define the following attributes:
            - possible_agents
            - render_mode

            These attributes should not be changed after initialization.
            """

            self.metadata = metadata
            self.agents = agents
            self.possible_agents = agents
            self.observation_spaces = observation_spaces
            self.action_spaces = action_spaces
            self.state = initial_state
            self.option = options
            self.render_mode = options.get("render_mode", "rgb_array")
            self.iteration_num = 0

        def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[ObsDict, dict[str, dict]]:
            """
            Reset needs to initialize the `agents` attribute and must set up the
            environment so that render(), and step() can be called without issues.
            Here it initializes the `num_moves` variable which counts the number of
            hands that are played.
            Returns the observations for each agent
            """
            self.iteration_num = 0
            if seed is None:
                self.seed = random.randint(0, 1000)
            self.state, self.observations = transition_function.reset()
            self.agents = self.possible_agents[:]
            self.rewards = {agent: 0 for agent in self.agents}
            self._cumulative_rewards = {agent: 0 for agent in self.agents}

            self.terminations = {agent: False for agent in self.agents}
            self.truncations = {agent: False for agent in self.agents}
            self.infos = {agent: {} for agent in self.agents}

            self.observations = {agent: self.observe(
                agent) for agent in self.agents}

            """
            Our agent_selector utility allows easy cyclic stepping through the agents list.
            """
            self._agent_selector = agent_selector(self.agents)
            self.agent_selection = self._agent_selector.next()

        def step(self, actions: ActionDict) -> tuple[ObsDict, dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
            """Receives a dictionary of actions keyed by the agent name.

            Returns the observation dictionary, reward dictionary, terminated dictionary, truncated dictionary
            and info dictionary, where each dictionary is keyed by the agent.
            """

            self.iteration_num += 1
            truncations = {agent: self.frame >=
                           self.max_cycles for agent in self.agents}

            terminations = {agent: self.frame >=
                           self.max_cycles for agent in self.agents}

            infos = {agent: {} for agent in self.agents}

            self.state, observations = transition_function(
                self.state, actions)

            rewards = reward_function(self.state, actions)

            if self.render_mode == "human":
                self.render()
            return observations, rewards, terminations, truncations, infos

        def render(self) -> None | np.ndarray | str | list:
            """Displays a rendered frame from the environment, if supported.

            Alternate render modes in the default environments are `'rgb_array'`
            which returns a numpy array and is supported by all environments outside
            of classic, and `'ansi'` which returns the strings printed
            (specific to classic environments).
            """
            raise NotImplementedError

        def close(self):
            """Closes the rendering window."""
            pass

        def state(self) -> np.ndarray:
            """Returns the state.

            State returns a global view of the environment appropriate for
            centralized training decentralized execution methods like QMIX
            """
            raise NotImplementedError(
                "state() method has not been implemented in the environment {}.".format(
                    self.metadata.get("name", self.__class__.__name__)
                )
            )

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

        def __str__(self) -> str:
            """Returns the name.

            Which looks like: "space_invaders_v1" by default
            """
            if hasattr(self, "metadata"):
                return self.metadata.get("name", self.__class__.__name__)
            else:
                return self.__class__.__name__

        @property
        def unwrapped(self) -> ParallelEnv:
            return self

    def env(**kwargs):
        env = CustomParallelEnv(**kwargs)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return CustomParallelEnv


if __name__ == '__main__':

    custom_env = createParallelEnv()
