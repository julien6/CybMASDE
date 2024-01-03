import math
import random

from collections import OrderedDict
from copy import copy
from typing import Dict, List, Any
from random import shuffle

from backend.src.utils.utils import Observation
from backend.src.behavior_models.behavior import Behavior, instantiateBehavior

class DecisionTreeBehavior(Behavior):
    
    def __init__(self, data: Any) -> None:
        super().__init__(data)

    def next_action(self, observations: List[Observation], reward: int) -> str:

        for condition in list(self.data["branches"].keys()):
            extendedCondition = str(condition)

            for observation in observations:
                extendedCondition = extendedCondition.replace(observation.id, observation.value)
                try:
                    result = eval(extendedCondition)
                except Exception as err:
                    print(err)
                    result = False
                    continue
                if(result):
                    next = self.data["branches"][condition]
                    if (type(next) == str):
                        return next
                    return instantiateBehavior(next).next_action(observations)

        return "do_nothing"