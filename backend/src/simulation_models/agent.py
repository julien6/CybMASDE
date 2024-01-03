from collections import OrderedDict
from copy import copy
from typing import Dict, List, Any
from random import shuffle

import math
import random


def fromObsToState(observations: List[int]) -> str:
    return "".join([str(obs) for obs in observations])


def fromStateToObs(obsState: str) -> List[int]:
    return [int(gymId) for gymId in obsState.split("")]

class Agent:

    agentID: str
    behaviour: Behaviour

    def __init__(self, agentID: str) -> None:
        self.agentID = agentID

    def next_action(self, observationGym: Any, reward: float) -> int:
        pass

    def reset(self):
        pass