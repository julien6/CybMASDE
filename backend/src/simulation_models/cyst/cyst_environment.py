import functools
import json
import numpy as np

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from typing import Dict, List, Any, Tuple, Union
from agents.agents import Agent, DecisionTreeAgent, LazyAgent, MARLAgent, RandomAgent
from environment import EnvironmentMngr
import matplotlib.pyplot as plt


def env(environment: Any, render_mode=None):

    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = CystEnvironment(node_environment=environment,
                          render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    # env = wrappers.OrderEnforcingWrapper(env)
    return env


class CystEnvironment(AECEnv):

    metadata = {"render_modes": ["human"], "name": "Network nodes"}

    envMngr: EnvironmentMngr

    def __init__(self, node_environment: Any, render_mode=None):

        self.envMngr = EnvironmentMngr(nodeEnvironment=node_environment)

        self.possible_agents = self.envMngr.agtPropSpace

        # ====================
        # TODO: voir ce qu'un agent peut observer réellement sur un SI
        # TODO: ajouter et integrer des precondition_script et postcondition_script
        # TODO : l'espace d'observation est seulement l'union de toutes les propriétés des "effect actions"
        # TODO: avoir les actions implémentées dans le code mais les afficher dans le json lors de la serialization sous forme de {...precondition... effects...}
        # TODO: avoir des classes de comportement dans le code bien organisées avec les actions dans plusieurs fichiers
        # TODO: avoir une fonction envAfterEffect qui agit après chaque execution d'une action par un agent pour faire appliquer les éventuels after-effects
        # TODO : décoreller les identifiants des jsons avec les identifiants attachés aux agents (peu importe sur quel noeud ils se trouvent)
        # ====================
        self.observation_spaces = self.envMngr.obsGymSpace

        self.action_spaces = self.envMngr.actGymSpace

        self.render_mode = render_mode

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        renderedState = json.dumps(self.envMngr.environment, indent=4)
        # print(renderedState)
        return renderedState

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        return self.observations[agent]

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """

        self.envMngr.reset()

        self.agents = self.possible_agents
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = self.envMngr.initialEnvironment

        self.observations = self.envMngr.fromObsPropToObsGym(
            {agent: self.envMngr.getAgentObservations(agent) for agent in self.agents})

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            return

        currentAgent = self.agent_selection

        obs, actionApplied = self.envMngr.applyAction(action, currentAgent)

        if (currentAgent == self.agents[-1]):
            reward, end = self.envMngr.getReward()["attackers"]

            for attacker in self.agents:

                print("reward: ", reward)

                self.rewards[attacker] = reward

                self._cumulative_rewards[attacker] += reward

                if (end):
                    print("Attacker {} reached the ultimate state".format(attacker))

        self.observations[currentAgent] = obs[currentAgent]

        self.state = self.envMngr.environment

        if self.render_mode == "human":
            self.render()

        self.terminations[currentAgent] = False  # end

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()


class EnvironmentPlayer:
    env: CybMASDEEnvironment
    iteration: int
    iterationMax: int
    iterator: Any
    agents: Dict[str, Agent]

    def __init__(self, env: CybMASDEEnvironment, iterationMax=2 ** 63) -> None:
        self.env = env
        self.iteration = 0
        self.iterationMax = iterationMax
        self.env.reset()
        self.iterator = self.env.agent_iter().__iter__()
        self.agents = {}
        self.init_agent_behaviours()

    def init_agent_behaviours(self):

        singleAttackerDT: List[int] = ["use_powershell",
                                       "use_dll_side_loading_to_execute_script",
                                       "use_remote_system_discovery_on_ws",
                                       "exploit_public_facing_application",
                                       "access_on_ws",
                                       "discover_DB_server",
                                       "pass_the_hash_to_lateral_move_to_db_server",
                                       "use_an_ingress_tool_transfer_to_control_server",
                                       "get_control_on_DB_server",
                                       "exfiltrate_data_over_C2_channel",
                                       "exploit_system_network_configuration_discovery_on_ws",
                                       "discover_PS_server",
                                       "use_lateral_tool_transfer_to_lateral_move_on_PS_server",
                                       "get_control_on_PS_server",
                                       "install_a_custom_spyware",
                                       "getting_flag"]

        singleDefenderDT: List[int] = [
            "use_code_signing",
            "doNothing",
            "doNothing",
            "detect_in_application_log_content",
            "doNothing",
            "doNothing",
            "use_privilege_account_management",
            "doNothing",
            "doNothing",
            "monitor_executed_commands_and_arguments",
            "doNothing",
            "doNothing",
            "doNothing",
            "doNothing",
            "doNothing",
            "doNothing"]

        attacker1DT: List[int] = [
            # "use_valid_accounts",
            #                       "access_on_ws",
            #                       "exploit_system_network_configuration_discovery_on_ws",
            #                       "discover_PS_server",
            #                       "use_lateral_tool_transfer_to_lateral_move_on_PS_server",
            #                       "use_an_ingress_tool_transfer_to_control_server",
                                  "get_control_on_PS_server",
                                  "install_a_custom_spyware"]

        attacker2DT: List[int] = [
            # "exploit_public_facing_application",
            # "access_on_ws",
            # "use_powershell",
            # "use_dll_side_loading_to_execute_script",
            # "use_remote_system_discovery_on_ws",
            # "discover_DB_server",
            # "pass_the_hash_to_lateral_move_to_db_server",
            # "use_an_ingress_tool_transfer_to_control_server",
            "get_control_on_DB_server",
            "exfiltrate_data_over_C2_channel",
            "getting_flag"]

        defender1DT: List[int] = ["detect_in_application_log_content"]
        defender2DT: List[int] = []

        for agent in self.env.envMngr.agtPropSpace:
            behaviour = self.env.envMngr.getAgentBehaviour(agent)
            if behaviour == "lazy":
                self.agents[agent] = LazyAgent(self.env.envMngr, agent)
            if behaviour == "random":
                self.agents[agent] = RandomAgent(self.env.envMngr, agent)
            if behaviour == "singleAttackerDT":
                self.agents[agent] = DecisionTreeAgent(
                    self.env.envMngr, agent, singleAttackerDT)
            if behaviour == "singleDefenderDT":
                self.agents[agent] = DecisionTreeAgent(
                    self.env.envMngr, agent, singleDefenderDT)
            if behaviour == "attacker1DT":
                self.agents[agent] = DecisionTreeAgent(
                    self.env.envMngr, agent, attacker1DT)
            if behaviour == "attacker2DT":
                self.agents[agent] = DecisionTreeAgent(
                    self.env.envMngr, agent, attacker2DT)
            if behaviour == "defender1DT":
                self.agents[agent] = DecisionTreeAgent(
                    self.env.envMngr, agent, defender1DT)
            if behaviour == "defender2DT":
                self.agents[agent] = DecisionTreeAgent(
                    self.env.envMngr, agent, defender2DT)
            if behaviour == "marl":
                self.agents[agent] = MARLAgent(self.env.envMngr, agent)

    def next(self) -> Tuple[str, str]:

        if self.iteration >= self.iterationMax:
            print("Reached maxIteration !")
            return

        # return the next agent to play
        agent = self.iterator.__next__()

        observation, reward, termination, truncation, info = self.env.last()

        observationProp = self.env.envMngr.fromObsGymToObsProp(
            {agent: observation})

        action = None
        if not termination:
            action = self.agents[agent].next_action(observation, reward)
            actionPropName = self.env.envMngr.getActPropID(agent, action)
            print("{} has chosen action: {} ({})".format(
                agent, actionPropName, action))

        self.env.step(action)

        lastValues = {"agent": agent, "observation": observationProp, "reward": reward,
                      "termination": termination, "truncation": truncation, "info": info, "next_action": action}

        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        self.iteration += 1
        logs = "Iteration {}: {}".format(
            self.iteration, json.dumps(lastValues, cls=NpEncoder))
        return (agent, logs)

    def dumpEnvironment(self) -> str:
        return json.dumps(self.env.envMngr.environment)

    def dictEnvironment(self) -> Any:
        return self.env.envMngr.environment


def loadFile(filePath: str):
    newDictEnv = json.load(open(filePath, "r"))
    return EnvironmentPlayer(CybMASDEEnvironment(
        node_environment=newDictEnv, render_mode="human"))


if __name__ == '__main__':

    envPlayer: EnvironmentPlayer = loadFile("worldStates/t1.json")

    executionNumber = 1
    episodeNumber = 1000
    iterationNumberPerEpisode = 50

    averageCumulativeRewardsList: Dict[str, List[float]] = {
        agent: [0] * episodeNumber for agent in envPlayer.env.envMngr.agtPropSpace}

    for executionIndex in range(0, executionNumber):

        cumulativeRewardsList: Dict[str, List[float]] = {
            agent: [] for agent in envPlayer.env.envMngr.agtPropSpace}

        for k in range(0, episodeNumber):
            print("============= Episode {} ===================".format(str(k)))
            for i in range(0, iterationNumberPerEpisode):
                print("---- Iteration {} ----".format(str(i)))
                envPlayer.next()
                print("")
            print(" ===> Cumulative reward at the end of the espisode: ",
                  envPlayer.env._cumulative_rewards)

            cumulativeRewardsList = {agent: cumulativeRewardsList[agent] + [
                envPlayer.env._cumulative_rewards[agent]] for agent in envPlayer.env.envMngr.agtPropSpace}

            # reset the environement and agent behaviours
            envPlayer.env.reset()
            for agent in envPlayer.agents:
                envPlayer.agents[agent].reset()

            print("\n\n")

        for agent, cumulativeRewards in cumulativeRewardsList.items():
            averageCumulativeRewardsList[agent] = [sum(i) for i in zip(
                averageCumulativeRewardsList[agent], cumulativeRewards)]

    for agent, averageCumulativeRewards in averageCumulativeRewardsList.items():
        averageCumulativeRewardsList[agent] = [
            float(i/executionNumber) for i in averageCumulativeRewards]

    x = [int(episodeIndex) for episodeIndex in range(0, episodeNumber)]

    markers = ["v", "o", "x", "s", "*", ".", "+", "s", "d"]

    json.dump(averageCumulativeRewardsList, open("dt.json", "w+"))

    for agent, averageCumulativeRewards in averageCumulativeRewardsList.items():
        plt.scatter(x, averageCumulativeRewards,
                    marker=markers.pop(), s=70, alpha=0.25, label=agent)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()

    plt.show()
