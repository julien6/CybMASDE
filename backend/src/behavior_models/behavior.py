import math
import random

from collections import OrderedDict
from copy import copy
from typing import Dict, List, Any
from random import shuffle

from backend.src.behavior_models.qlearning_marl_behavior import QLearningBehavior
from backend.src.behavior_models.decision_tree_behavior import DecisionTreeBehavior
from backend.src.behavior_models.lazy_behavior import LazyBehavior
from backend.src.behavior_models.random_behavior import RandomBehavior

from backend.src.utils.utils import Observation


class Behavior:
    """
    A Behavior is just a data structure that is to be used to define an agent behavior.
    """

    data: Any
    """
    A behavior state is simply a data structure of the behavior.
    It really belongs only to an agent.
    i.e: a Q-Table, a matrix of neural networks, a decision tree mapping, etc.
    """

    def __init__(self, data: Any) -> None:
        self.data = data

    def next_action(self, observations: List[Observation], reward: int) -> str:
        """
        Allows determining the next decision to make according to the core behavior principle and its current state

        :param List[Observation] observations: The observations the agent can get (also includes its own properties)
        """
        pass

def instantiateBehavior(next: Any) -> Behavior:
    if next["type"] == "decision_tree":
        return DecisionTreeBehavior(next["data"])

    if next["type"] == "qlearning-marl":
        return QLearningBehavior(next["data"])

    if next["type"] == "random":
        return RandomBehavior(next["data"])

    if next["type"] == "lazy":
        return LazyBehavior(next["data"])