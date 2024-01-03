import dataclasses
import re
import json

from collections import OrderedDict
from gymnasium import spaces
from typing import Dict, List, Any, Tuple, Set
from copy import copy, deepcopy


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


class EnvironmentMngr:

    # the current environment model instance (dict of Nodes), such as:
    # {
    #    "node1": {
    #        "properties": {
    #           ...#current_properties...
    #        },
    #        "actions": {
    #           ...
    #           "action1": {
    #               "precondition": #boolean_property_expression,
    #               "effects": [#potential_effect_properties]
    #           }
    #           ...
    #       }
    #       ...
    #    }
    # }
    # it is to be modified only by actions
    environment: Any

    # initial environment used for reseting
    initialEnvironment: Any

    # all gathered possibles properties to be observed (i.e all possible actions effects properties), such as:
    # [("propertyName1", "propertyValue1"), ("propertyName2", "propertyValue2"), ...]
    obsPropSpace: Dict

    # gym space used to describe any observation in the same format:
    # for instance, if obsPropSpace = [property1, property2],
    # then, observation can be:
    #   - [None, None]
    #   - [None, property2]
    #   - [property1, None]
    #   - [property1, property2]
    # Coding with 0 = None and 1 = property, we have either [0,0] or [0,1] or [1,0] or [1,1]
    # so an observation is like [boolValue, boolValue] which is given by MultiBinary(2)
    obsGymSpace: spaces.Dict

    # all gathered possibles actions to be played, such as
    # [
    #   ("doNothing", {"description": "Do nothing", "success_probability": 1, "precondition": "True", "effects": {}, "cost": 100}),
    #   ("helloWorld", { "description": "Hello world", "success_probability": 1, "precondition": "True", "effects":
    #               { "[#AgentID][\"knowledge\"][\"message\"]": "Hello World"},"cost": 100},),
    #   ...
    # ]
    actPropSpace: OrderedDict

    # gym space used to describe any action in the same format:
    # for instance, if actPropSpace = [action1, action2, action3],
    # then, an action can be either action1 or action2 or action3
    # Coding with 0 = action1, 1 = action2 and 2 = action3, we have either 0 or 1 or 2
    # so an action is like x with x in {0,1,2} which is given by Discrete(3)
    actGymSpace: spaces.Dict

    # the list of all possible agents names (with full name path)
    agtPropSpace: List[str]

    seed: int

    def __init__(self, nodeEnvironment: Any, seed: int = 42) -> None:

        self.seed = seed

        self.environment = nodeEnvironment

        self.initialEnvironment = deepcopy(self.environment)

        # getting all observable properties, all possible action, all possible agent
        self.obsPropSpace, self.actPropSpace, self.agtPropSpace = self.getPropSpaces()

        # getting observation and aciton spaces as Gym spaces as a vector and a finite number interval
        self.obsGymSpace = spaces.Dict({agentName: spaces.MultiBinary(len(
            agentObsPropSpace), seed=self.seed) for agentName, agentObsPropSpace in self.obsPropSpace.items()})
        self.actGymSpace = spaces.Dict({agentName: spaces.Discrete(len(
            agentActPropSpace), seed=self.seed) for agentName, agentActPropSpace in self.actPropSpace.items()})

    def reset(self):
        self.environment = deepcopy(self.initialEnvironment)

    def getNodeIDPropFromAgtID(self, agentName: str) -> str:
        for nodeID, node in self.environment["nodes"].items():
            if agentName in list(node["processes"]["agents"].keys()):
                return nodeID

    def getAgentBehaviour(self, agent: str) -> str:
        nodeID = self.getNodeIDPropFromAgtID(agent)
        return self.environment["nodes"][nodeID]["processes"]["agents"][agent]["behaviour"]

    def getAgentObservations(self, agent):
        nodeID = self.getNodeIDPropFromAgtID(agent)
        return [(propID, propVal) for propID, propVal in list(self.environment["nodes"][nodeID]["processes"]["agents"][agent]["observations"].items())]

    def getValueFromJsonPath(self, jsonEntry, jsonPath):
        obj = jsonEntry
        for jsonKey in jsonPath[1:-1].split("]["):
            obj = obj.get(jsonKey[1:-1], None)
            if obj == None:
                return obj
        return obj

    def delValueFromJsonPath(self, jsonEntry, jsonPath):
        def _delValueFromJsonPath(jsonEntry, jsonKeys):
            if len(jsonKeys) == 1:
                del jsonEntry[jsonKeys[0]]
            else:
                key = jsonKeys.pop()
                _delValueFromJsonPath(jsonEntry[key], jsonKeys)
                return
        jsonKeys = [jsonKey[1:-1] for jsonKey in jsonPath[1:-1].split("][")]
        jsonKeys.reverse()
        return _delValueFromJsonPath(jsonEntry, jsonKeys)

    def setValueFromJsonPath(self, jsonEntry, jsonPath, value):
        def _setValueFromJsonPath(jsonEntry, jsonKeys):
            key = jsonKeys.pop()
            if (not key in jsonEntry.keys()):
                jsonEntry[key] = {}
            if len(jsonKeys) == 0:
                jsonEntry[key] = value
                return
            else:
                if type(jsonEntry[key]) == str:
                    jsonEntry[key] = {}
                _setValueFromJsonPath(jsonEntry[key], jsonKeys)

        jsonKeys = [jsonKey[1:-1] for jsonKey in jsonPath[1:-1].split("][")]
        jsonKeys.reverse()
        return _setValueFromJsonPath(jsonEntry, jsonKeys)

    def getPropSpaces(self) -> Tuple:
        agtPropSpace: List[str] = []
        actPropSpace: List[Tuple[str, Dict]] = []
        obsPropSpace: Dict[str, Set] = {}
        agentsOnNodes = {}

        # Retrieving all agents on the node with their initial observations
        for nodeID, node in self.environment["nodes"].items():
            agentsOnNodes[nodeID] = list(
                node["processes"]["agents"].keys())
            agtPropSpace += agentsOnNodes[nodeID]
            for agent in agentsOnNodes[nodeID]:
                obsPropSpace[agent] = {(propID, propValue) for propID, propValue in list(
                    node["processes"]["agents"][agent]["observations"].items())}

        actPropSpace = {agent: [(actionID, action) for actionID,
                        action in list(self.environment["actions"].items())] for agent in agtPropSpace}

        # Once we got all the agents, we can infer all of their potential observations
        for actionID, action in self.environment["actions"].items():

            # retrieve the post condition properties
            actionPostcondition = action["postcondition"]
            for propID, propValue in actionPostcondition.items():

                if ".observations" in propID:

                    simplePropID = propID.split("observations.")[-1]

                    if "{{agent}}" in propID:
                        # as any agent can be on any node, we must infer the potential effects for {{agent}} shortcut
                        for agent in agtPropSpace:
                            obsPropSpace[agent].add((simplePropID, propValue))
                    else:
                        agent = propID.split(
                            "processes.agents.")[-1].split(".observations")[0]
                        obsPropSpace[agent].add((simplePropID, propValue))

        return ({agent: list(obsProp) for agent, obsProp in obsPropSpace.items()}, actPropSpace, agtPropSpace)

    def fromActGymToActProp(self, actionGymDict):
        return {agentName: self.actPropSpace[agentName][actGymID] for agentName, actGymID in actionGymDict.items()}

    def fromActGymToActPropID(self, actionGymDict):
        return {agentName: self.actPropSpace[agentName][actGymID][0] for agentName, actGymID in actionGymDict.items()}

    def fromActPropToActGym(self, actionPropDict):
        return {agentName: int(self.actPropSpace[agentName].index(actProp)) for agentName, actProp in actionPropDict.items()}

    def fromActPropIDToActGym(self, actionPropDict):
        return {agentName: int([actProp[0] for actProp in self.actPropSpace[agentName]].index(actPropID)) for agentName, actPropID in actionPropDict.items()}

    def fromObsGymToObsProp(self, observationGymDict):
        return {agentName: [self.obsPropSpace[agentName][observationGymIndex] for observationGymIndex in range(0, len(observationGymVector)) if observationGymVector[observationGymIndex] == 1] for agentName, observationGymVector in observationGymDict.items()}

    def fromObsPropToObsGym(self, observationPropDict):
        return {agentName: [(1 if (observationProp in observationPropVector) else 0) for observationProp in self.obsPropSpace[agentName]] for agentName, observationPropVector in observationPropDict.items()}

    def getActGymID(self, agent, actionPropID):
        return list(self.fromActPropIDToActGym({agent: actionPropID}).values())[0]

    def getActPropID(self, agent, actionGymID):
        return list(self.fromActGymToActPropID({agent: actionGymID}).values())[0]

    def applyAction(self, actionGymID: int, agentPropID: str):
        """
        Apply action postcondition changes for the current agent
        """

        agentNodeID = self.getNodeIDPropFromAgtID(agentPropID)

        actionName, action = list(self.fromActGymToActProp(
            {agentPropID: actionGymID}).values())[0]
        precondition: str = action["precondition"]

        # We first infer the shortcut values for {{agent}} and {{node}}
        precondition = precondition.replace(
            "{{agent}}", "{}.processes.agents.{}".format(agentNodeID, agentPropID))
        precondition = precondition.replace("{{node}}", agentNodeID)

        # Then, we want to replace the json path with the json environment value
        areAllPropertiesPresent = True
        for match in re.compile(r"([[a-zA-Z0-9_.]*)").finditer(" " + precondition + " "):
            if "." in match.groups(0)[0]:
                conditionComponent = match.groups(0)[0]
                valuedConditionComponent = self.getValueFromJsonPath(
                    self.environment, '["nodes"]' + "".join(['["{}"]'.format(pathID) for pathID in conditionComponent.split(".")]))
                if (valuedConditionComponent == None):
                    print("{} not present".format(conditionComponent))
                    areAllPropertiesPresent = False
                    break
                if type(valuedConditionComponent) == str:
                    valuedConditionComponent = '"{}"'.format(
                        valuedConditionComponent)
                else:
                    valuedConditionComponent = str(valuedConditionComponent)
                precondition = precondition.replace(
                    conditionComponent, valuedConditionComponent)

        if (areAllPropertiesPresent):
            # print("EXPRESSION: ", precondition)
            precondition = precondition.replace('"True"', "True")
            precondition = precondition.replace('"False"', "False")
            preconditionSatisfied = eval(precondition)
            if (not preconditionSatisfied):
                print("error: precondition not meet : {}".format(precondition))
            else:

                # apply the post condition properties
                actionPostcondition = action["postcondition"]
                for propID, propValue in actionPostcondition.items():

                    inferedPropID = propID

                    # infering shortcut values for ID part of the property
                    inferedPropID = inferedPropID.replace(
                        "{{agent}}", "{}.processes.agents.{}".format(agentNodeID, agentPropID))
                    inferedPropID = inferedPropID.replace(
                        "{{node}}", agentNodeID)

                    # updating the environment
                    self.setValueFromJsonPath(
                        self.environment, '["nodes"]' + "".join(['["{}"]'.format(pathID)
                                                                 for pathID in inferedPropID.split(".")]), propValue)

        # retrieving the observations and taking into account the fact the agent might have moved to another node
        agentObservation = {agent: self.getAgentObservations(
            agent) for agent in self.agtPropSpace}

        return self.fromObsPropToObsGym(agentObservation), areAllPropertiesPresent and preconditionSatisfied

    def getReward(self) -> Dict[str, Tuple[float, bool]]:

        propertiesIDs = list(OrderedDict(
            self.environment["nodes"]["PC1"]).keys())

        if "getting_flag" in propertiesIDs:
            return {
                "attackers": (20000, True),
                "defenders": (-100, False)
            }

        if "install_a_custom_spyware" in propertiesIDs and "exfiltrate_data_over_C2_channel" in propertiesIDs:
            return {
                "attackers": (10000, False),
                "defenders": (-10, False)
            }

        # if "get_control_on_DB_server" in propertiesIDs \
        #         and "get_control_on_PS_server":
        #     return {
        #         "attackers": (500, False),
        #         "defenders": (-100, False)
        #     }

        if "pass_the_hash_to_lateral_move_to_db_server" in propertiesIDs \
                and "use_an_ingress_tool_transfer_to_control_server" in propertiesIDs:
            return {
                "attackers": (2500, False),
                "defenders": (-100, False)
            }

        if "use_lateral_tool_transfer_to_lateral_move_on_PS_server" in propertiesIDs \
                and "use_an_ingress_tool_transfer_to_control_server" in propertiesIDs:
            return {
                "attackers": (2500, False),
                "defenders": (-100, False)
            }

        # if "discover_PS_server" in propertiesIDs or "discover_DB_server" in propertiesIDs:
        #     return {
        #         "attackers": (100, False),
        #         "defenders": (-100, False)
        #     }

        if "access_on_ws" in propertiesIDs and "use_remote_system_discovery_on_ws" in propertiesIDs \
                and "use_dll_side_loading_to_execute_script" in propertiesIDs:
            return {
                "attackers": (500, False),
                "defenders": (-100, False)
            }

        if "access_on_ws" in propertiesIDs and "exploit_system_network_configuration_discovery_on_ws" in propertiesIDs:
            return {
                "attackers": (500, False),
                "defenders": (-100, False)
            }

        if "use_valid_accounts" in propertiesIDs or "exploit_public_facing_application" in propertiesIDs \
                or "use_powershell" in propertiesIDs or "use_command_powershell" in propertiesIDs:
            return {
                "attackers": (1, False),
                "defenders": (-100, False)
            }

        return {
            "attackers": (-1, False),
            "defenders": (-1, False)
        }


if __name__ == '__main__':
    # new envMng from a predefined two nodes state
    # envMngr = EnvironmentMngr(nodeEnvironment=environmentTest)
    envMngr = EnvironmentMngr(nodeEnvironment=json.load(
        open("worldStates/t1.json", "r")))

    # gymSpace = envMngr.obsGymSpace  # geting the gym observation space
    # obsGymSample = gymSpace.sample()  # to a sample (as an agent would)
    # print({agent: list(obs)
    #       for agent, obs in dict(OrderedDict(obsGymSample)).items()})
    # obsProp = envMngr.fromObsGymToObsProp(obsGymSample)
    # obsGym = envMngr.fromObsPropToObsGym(obsProp)
    # print(obsGym)

    # actGymSpace = envMngr.actGymSpace
    # actGymSample = actGymSpace.sample()
    # print({agent: act for agent, act in dict(OrderedDict(actGymSample)).items()})
    # actProp = envMngr.fromActGymToActPropID(actGymSample)
    # actGym = envMngr.fromActPropIDToActGym(actProp)
    # print(actGym)

    # actGym = envMngr.fromActPropIDToActGym({"attacker1": "action1"})

    # print({agent: envMngr.getAgentObservations(agent) for agent in envMngr.agtPropSpace})

    # obsProp = envMngr.applyAction(actGym["attacker1"], "attacker1")

    # print(envMngr.fromObsGymToObsProp(obsProp))
