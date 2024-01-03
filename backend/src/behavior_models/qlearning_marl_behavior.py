import math
import random

from collections import OrderedDict
from copy import copy
from typing import Dict, List, Any
from random import shuffle

from backend.src.utils.utils import Observation
from backend.src.behavior_models.behavior import Behavior


def fromObsToState(observations: List[int]) -> str:
    return "".join([str(obs) for obs in observations])


def fromStateToObs(obsState: str) -> List[int]:
    return [int(gymId) for gymId in obsState.split("")]


class QLearningBehavior(Behavior):

    agentObsSpace: List[str]
    agentActSpace: List[str]

    alpha: float
    gamma: float
    epsilon: float
    qTable: Dict

    lastObsState: str
    lastAction: int
    iteration: int

    def __init__(self, data: Any) -> None:
        super().__init__(data)
        self.alpha = data["alpha"]
        self.gamma = data["gamma"]
        self.epsilon = data["epsilon"]
        self.qTable = data["qTable"]
        self.agentActSpace: List[str] = self.data["agentActSpace"]
        self.agentObsSpace: List[str] = self.data["agentObsSpace"]
        self.lastObsState = None
        self.lastAction = None
        self.iteration = 1

    def next_action(self, observations: List[Observation], reward: int) -> str:

        self.lastObsState = observations[0].value
        self.lastAction = observations[1].value
        observation: List[int] = observations[2].value

        # If this is the first time, choose randomly
        if (self.lastObsState == None or self.lastAction == None):
            self.lastObsState = fromObsToState(observation)
            self.lastAction = random.choice(self.agentActSpace)
            return self.lastAction
        else:
            # First, update the QTable with previous explored state
            previous_state = self.lastObsState

            state = fromObsToState(observation)
            last_action = self.lastAction

            old_value = 0 if self.qTable.get(previous_state, {}).get(
                last_action, None) == None else self.qTable[previous_state][last_action]
            next_max = max(list(self.qTable.get(state, {"k": 0}).values()))

            new_value = (1 - self.alpha) * old_value + self.alpha * \
                (reward + self.gamma * next_max)

            self.qTable.setdefault(previous_state, {})
            self.qTable[previous_state][last_action] = new_value

            action = None
            # Second, explore states randomly or using QTable
            if random.uniform(0, 1) < self.epsilon:
                print(
                    "Exploring the space to know which reward is associated with an action in a given state")
                # Explore action space
                # action = self.envMngr.actGymSpace[self.agentID].sample()

                otherActions = [actionGym for actionGym in range(0, len(list(
                    self.agentActSpace))) if
                    actionGym not in list(OrderedDict(self.qTable.get(state, {})).keys())]
                if (len(otherActions) == 0):
                    action = random.choice(self.agentActSpace)
                else:
                    shuffle(otherActions)
                    action = otherActions[0]

            else:
                if (self.qTable.get(state, None) == None):
                    print(
                        "Exploring the space to know which reward is associated with an action in a given state")
                    action = random.choice(self.agentActSpace)
                else:
                    print("Using the QValues")
                    # print("For state ", self.envMngr.fromObsGymToObsProp({self.agentID: observation})[self.agentID],
                    #       " 'action -> reward' QValues are ", {
                    #     self.envMngr.fromActGymToActPropID({self.agentID: actGym})[
                    #         self.agentID]: qValue for actGym, qValue in self.qTable[state].items()})

                    availableQValues = list(OrderedDict(
                        self.qTable.get(state)).values())
                    maxActionQValueForCurrentState = max(availableQValues)

                    if len(availableQValues) < len(list(self.agentActSpace)):
                        if (maxActionQValueForCurrentState <= 0):
                            print(
                                "Exploring the space to know which reward is associated with an action in a given state")
                            otherActions = [actionGym for actionGym in range(0, len(list(self.agentActSpace))) if
                                            actionGym not in list(OrderedDict(self.qTable.get(state)).keys())]
                            shuffle(otherActions)

                            self.lastAction = otherActions[0]
                            self.lastObsState = state

                            return self.lastAction

                    maxActionQValueForCurrentStateGymID = [action for action, qValue in self.qTable.get(
                        state).items() if qValue == maxActionQValueForCurrentState][0]
                    # print("for state ", state, " available QValues are ", self.qTable.get(state))
                    action = maxActionQValueForCurrentStateGymID  # Exploit learned values

        self.lastAction = action
        self.lastObsState = state

        self.iteration += 1

        return action
