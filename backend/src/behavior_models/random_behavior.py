import math
import random

from collections import OrderedDict
from copy import copy
from typing import Dict, List, Any
from random import shuffle

from backend.src.utils.utils import Observation
from backend.src.behavior_models.behavior import Behavior


class RandomBehavior(Behavior):

    def __init__(self, data: Any) -> None:
        super().__init__(data)

    def next_action(self, observations: List[Observation], reward: int) -> str:
        agentActSpace: List[str] = self.data["agentActSpace"]
        return random.choice(agentActSpace)
