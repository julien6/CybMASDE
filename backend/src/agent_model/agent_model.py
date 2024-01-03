from typing import List, Any
from backend.src.utils.utils import Observation
from backend.src.behavior_models.behavior import Behavior
from copy import deepcopy

class Agent:
    """
    A agent is an individual pro-active entity with a behavior
    """

    id: str
    behavior: Behavior
    properties: Any
    initial_sate: Any

    def __init__(self, id: str, properties: Any, behavior: Behavior) -> None:
        self.initial_sate = deepcopy(self)
        self.behavior = behavior
        self.properties = properties
        self.id = id

    def next_action(self, observations: List[Observation], reward: int) -> int:
        self.behavior.next_action(observations, reward)

    def reset(self):
        self = self.initial_sate