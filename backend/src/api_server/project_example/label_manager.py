from typing import List
import gym
from backend.mma.mma_wrapper.label_manager import label_manager


class label_manager:
    """Example label manager that extends the base label_manager class.
    This class should implement the methods for one-hot encoding and decoding
    observations and actions.
    """

    def __init__(self):
        self.action_space = gym.Space
        self.observation_space: gym.Space

    def one_hot_encode_observation(self, observation, agent=None):
        # Example implementation of one-hot encoding an observation
        return super().one_hot_encode_observation(observation, agent)

    def one_hot_decode_observation(self, observation, agent=None):
        # Example implementation of one-hot decoding an observation
        return super().one_hot_decode_observation(observation, agent)

    def one_hot_encode_action(self, action, agent=None):
        # Example implementation of one-hot encoding an action
        return super().one_hot_encode_action(action, agent)

    def one_hot_decode_action(self, action, agent=None):
        # Example implementation of one-hot decoding an action
        return super().one_hot_decode_action(action, agent)
