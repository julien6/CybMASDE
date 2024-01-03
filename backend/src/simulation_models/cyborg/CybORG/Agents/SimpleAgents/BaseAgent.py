from simulation_models.cyborg.CybORG.Simulator.Actions.Action import Action, Sleep
from typing import Any, List
from simulation_models.cyborg.CybORG.Shared import Results
import numpy as np
from gym.utils import seeding
from gym import Space
import random


class BaseAgent:
    def __init__(self, name: str, np_random=None):
        self.name = name
        if np_random is None:
            np_random, seed = seeding.np_random()
        self.np_random = np_random

    def train(self, results: Results):
        """allows an agent to learn a policy"""
        raise NotImplementedError

    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        raise NotImplementedError

    def end_episode(self):
        """Allows an agent to update its internal state"""
        raise NotImplementedError

    def set_initial_values(self, action_space, observation):
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return f"{self.__class__.__name__}"


class PzBaseAgent(BaseAgent):

    @staticmethod
    def add_message_to_act(pz_action, pz_message, action_mapping):
        return pz_message * len(action_mapping.keys()) + pz_action

    @staticmethod
    def cyborg_to_pz_action(action: Action, actions_mapping: Any) -> int:
        return list({act: _action for act, _action in actions_mapping.items(
        ) if str(_action) == str(action) and action.get_params() == _action.get_params()})[0]

    @staticmethod
    def pz_to_cyborg_action(action: int, actions_mapping: Any) -> Action:
        return actions_mapping[action % len(actions_mapping.keys())]

    @staticmethod
    def get_message_from_obs(active_agents: List[str], pz_observation=None, cyborg_observation=None):
        """Get received messages from connected agents
        """
        if pz_observation is not None:
            pz_observation = pz_observation.tolist()
            received_messages = pz_observation[len(
                pz_observation) - len(active_agents):]
            return {other_agent: received_messages[index] for index, other_agent in enumerate(active_agents)}
        elif cyborg_observation != None:
            received_messages = cyborg_observation["message"] if "message" in cyborg_observation else [
                0] * len(active_agents)
            return {other_agent: None if received_messages[index] == 0 else received_messages[index] for index, other_agent in enumerate(active_agents)}

    def get_action(self, cyborg_observation, cyborg_action_space, pz_observation=None, pz_action_space=None, actions_mapping=None):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space
        including pettingzoo observations"""
        raise NotImplementedError


class PzRandom(PzBaseAgent):

    def get_action(self, cyborg_observation, cyborg_action_space, pz_observation: np.ndarray = None, pz_action_space: Space = None,
                   actions_mapping=None, active_agents: List[str] = None) -> int:

        received_messages = PzBaseAgent.get_message_from_obs(
            active_agents, pz_observation=pz_observation)
        received_messages = {other_agent: None if message == 0 else message -
                             1 for other_agent, message in received_messages.items()}
        # print(
        #     f"agent {self.name} received following messages: {received_messages}")

        # received_messages = AgentFreeCommsPettingZooParallelWrapper.get_message_from_obs(
        #     active_agents, cyborg_observation=cyborg_observation)
        # print(
        #     f"agent {self.name} received following messages: {received_messages}")

        # PzBaseAgent.cyborg_to_pz_action()
        # PzBaseAgent.pz_to_cyborg_action()

        to_broadcast_message = 49

        action = pz_action_space.sample() % len(actions_mapping.keys())
        # action = 4

        # print(
        #     f"agent {self.name} plays action {action} and broadcast message {to_broadcast_message}")

        return PzBaseAgent.add_message_to_act(action, to_broadcast_message, actions_mapping)
