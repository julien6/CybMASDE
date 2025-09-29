from typing import List
import gym
from world_model.joint_policy import JointPolicy


class RandomJointPolicy(JointPolicy):

    def __init__(self, action_space: gym.Space, agents: List):
        self.action_space = action_space
        self.agents = agents

    def next_action(self, joint_observation):
        return [self.action_space.sample() for _ in range(0, len(self.agents) + 1)]
