from ipaddress import IPv4Address, IPv4Network
from typing import Any, Dict, List
from simulation_models.cyborg.CybORG import CybORG
from simulation_models.cyborg.CybORG.Agents.SimpleAgents.BaseAgent import PzBaseAgent, PzRandom
from simulation_models.cyborg.CybORG.Agents.Wrappers.CommsPettingZooParallelWrapper import AgentCommsPettingZooParallelWrapper, \
    AgentFreeCommsPettingZooParallelWrapper, ActionsCommsPettingZooParallelWrapper
from simulation_models.cyborg.CybORG.Simulator.Actions.Action import Action, Sleep
from simulation_models.cyborg.CybORG.Simulator.Scenarios.FileReaderScenarioGenerator import FileReaderScenarioGenerator
from simulation_models.cyborg.CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import DroneSwarmScenarioGenerator
from simulation_models.cyborg.CybORG.Agents.Wrappers.PettingZooParallelWrapper import PettingZooParallelWrapper
from simulation_models.cyborg.CybORG.Agents import B_lineAgent, BlueReactRestoreAgent, BlueReactRemoveAgent, SleepAgent, DroneRedAgent, \
    RandomAgent, BaseAgent, RedMeanderAgent
from os.path import dirname
from pprint import pprint

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import inspect
from simulation_models.cyborg.CybORG.Agents.Wrappers.TrueTableWrapper import true_obs_to_table
from simulation_models.cyborg.CybORG.Simulator.SimulationController import SimulationController
import json
import enum

from simulation_models.SimulationModel import SimulationModel
from networkx.readwrite.json_graph.node_link import node_link_data


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, enum.Enum):
            return str(o)
        if isinstance(o, IPv4Address):
            return str(o)
        if isinstance(o, IPv4Network):
            return str(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class CyborgEnvironment(SimulationModel):

    def __init__(self, scenario_plan: Any, max_episode: int = 1, max_iteration: int = 1) -> None:
        super().__init__(max_episode, max_iteration)

        # path = inspect.getfile(CybORG)
        # path = dirname(path) + f'/Simulator/Scenarios/scenario_files/Scenario1b.yaml'
        # sg = FileReaderScenarioGenerator(path)

        sg = DroneSwarmScenarioGenerator(num_drones=8)

        env = CybORG(sg, 'sim')

        # self.wrapped_cyborg = PettingZooParallelWrapper(env)
        self.wrapped_cyborg = AgentFreeCommsPettingZooParallelWrapper(env)
        self.wrapped_cyborg.reset(42)

        self.max_eps = max_episode
        self.max_its = max_iteration

        # agents = {agent: RandomAgent() for agent in ["Blue", "Red", "Green"]}

        self.agents = {f"blue_agent_{agent}": PzRandom(
            name=f"blue_agent_{agent}") for agent in range(8)}

        self.iteration_data = {}

        self.pz_observations = None
        self.pz_observation_spaces = None
        self.pz_action_spaces = None
        self.cyborg_observations = None
        self.cyborg_actions = None
        self.cyborg_action_spaces = None

        self.agents_actions = None
        self.agents_observations = None
        self.agents_rewards = None
        self.true_states = None
        self.action_spaces = None
        self.observation_spaces = None

    def gym_to_cyborg_actions(self, gym_actions) -> Dict:
        gym_cyborg_mapping = self.wrapped_cyborg.agent_actions
        return {
            agent: gym_cyborg_mapping[agent][gym_action] for agent, gym_action in gym_actions.items()
        }

    def get_agent_state(self, agent_name: str):
        """Returns the full knowledge of an agent
        """
        true_state = self.wrapped_cyborg.env.get_agent_state(agent_name)
        state = json.loads(json.dumps(
            true_state, cls=EnhancedJSONEncoder, indent=4))
        return state

    def get_team_agent_mapping(self):
        """Return the host agent mapping
        """
        return self.wrapped_cyborg.env.environment_controller.team_assignments

    def get_true_state(self):
        """Returns the real full host nodes data
        """
        true_state = self.wrapped_cyborg.env.get_agent_state("True")
        state = json.loads(json.dumps(
            true_state, cls=EnhancedJSONEncoder))
        if "success" in list(state.keys()):
            del state["success"]
        return state

    def get_pz_cyborg_actions(self) -> Any:
        """Returns the mapping from int PettingZoo to Cyborg actions 
        """
        agent_actions = {agent: {int_pz: {k: v for k, v in action.__dict__.items() if k not in [
            "priority", "agent"]} for int_pz, action in mapping.items()} for agent, mapping in self.wrapped_cyborg.agent_actions.items()}
        pz_cyborg_actions = json.loads(json.dumps(
            agent_actions, cls=EnhancedJSONEncoder))
        return pz_cyborg_actions

    def get_cyborg_actions(self) -> Any:
        """Returns the cyborg actions
        """
        actions = json.loads(json.dumps({agent: {k: v for k, v in ({} if action == None else action).__dict__.items() if k not
                                                 in ["priority", "agent"]} for agent, action in self.cyborg_actions.items()
                                         if action != None}, cls=EnhancedJSONEncoder))
        return actions

    def get_cyborg_observations(self) -> Any:
        """Returns the cyborg observations
        """
        observations = json.loads(json.dumps(
            self.cyborg_observations, cls=EnhancedJSONEncoder))
        return observations

    def get_agents_cyborg_comms(self) -> Any:
        """Returns the cyborg comms from cyborg observations
        """
        return {agent:
                PzBaseAgent.get_message_from_obs(
                    self.wrapped_cyborg.agents, cyborg_observation=self.cyborg_observations[agent])
                for agent in self.wrapped_cyborg.agents}

    def get_agents_pz_comms(self) -> Any:
        """Returns the pz comms from pz observations
        """
        return {agent: {other_agent: None if message == 0 else (message - 1) for other_agent, message in PzBaseAgent.get_message_from_obs(self.wrapped_cyborg.agents, pz_observation=self.pz_observations[agent]).items()}
                for agent in self.wrapped_cyborg.agents}

    def next(self, requested_info: List = ["episode_number", "iteration_number", "observation_spaces", "agents_actions",
                                           "agents_observations", "agents_rewards", "true_states"]) -> Dict:

        if self.current_it == 0:

            print("Episode {}".format(str(self.current_ep)))

            # PettingZoo observations and action spaces
            self.pz_observations = self.wrapped_cyborg.reset()
            self.pz_observation_spaces = self.wrapped_cyborg._observation_spaces
            self.pz_action_spaces = self.wrapped_cyborg.action_spaces

            # Initializing actions
            self.cyborg_actions = {
                agent: None for agent in self.wrapped_cyborg.active_agents}

            #  CybORG observations and action spaces
            self.cyborg_observations = {agent: self.wrapped_cyborg.env.get_observation(agent)
                                        for agent in self.wrapped_cyborg.agents}
            # cyborg_observation_space = {agent: self.wrapped_cyborg.env.get_observation_space(
            #     agent) for agent in self.wrapped_cyborg.agents}
            self.cyborg_action_spaces = {agent: self.wrapped_cyborg.env.get_action_space(
                agent) for agent in self.wrapped_cyborg.agents}

            # =========== Data to save =============

            # Initialization for current episode
            self.agents_actions = {
                agent: None for agent in self.wrapped_cyborg.agents}
            self.agents_observations = {agent: observations.tolist()
                                        for agent, observations in self.pz_observations.items()}
            self.agents_rewards = {
                agent: None for agent in self.wrapped_cyborg.agents}
            self.true_states = self.get_true_state()

            self.action_spaces = {agent: sp.n for agent,
                                  sp in self.pz_action_spaces.items()}
            self.observation_spaces = {agent: sp.nvec.tolist()
                                       for agent, sp in self.pz_observation_spaces.items()}

        else:
            actions = {}
            for agent_name, agent in self.agents.items():
                if agent_name in self.wrapped_cyborg.agents:
                    # action can be in PettingZoo or Cyborg
                    action = None
                    # if the agent needs to include the pettingzoo observations and action spaces to choose the next action (AI)
                    if isinstance(agent, PzBaseAgent):
                        action = agent.get_action(self.cyborg_observations[agent_name],
                                                  self.cyborg_action_spaces[agent_name],
                                                  self.pz_observations[agent_name],
                                                  self.pz_action_spaces[agent_name],
                                                  self.wrapped_cyborg.agent_actions[agent_name],
                                                  self.wrapped_cyborg.agents)
                    # use regular Cyborg observation and action spaces
                    else:
                        action = agent.get_action(
                            self.cyborg_observations[agent_name], self.cyborg_action_spaces[agent_name])

                    # if the chosen action is not Pz encoded, encode it to
                    if isinstance(action, Action):
                        action = list({act: _action for act, _action in self.wrapped_cyborg.agent_actions[agent_name].
                                       items() if str(_action) == str(action) and
                                       action.get_params() == _action.get_params()})[0]

                    actions[agent_name] = action

            # Applying actions (actions must be in PettingZoo encoding)
            self.pz_observations, rew, done, info = self.wrapped_cyborg.step(
                actions)

            # ===================
            # getting iteration data
            self.agents_actions = actions
            self.agents_observations = [{agent: observations.tolist()
                                         for agent, observations in self.pz_observations.items()}]
            self.agents_rewards = rew
            self.true_states = self.get_true_state()
            # ===================

            # getting the cyborg observation
            self.cyborg_observations = {agent: self.wrapped_cyborg.env.get_observation(agent)
                                        for agent in self.wrapped_cyborg.agents}
            # to get the cyborg actions
            self.cyborg_actions = {agent: self.wrapped_cyborg.env.get_last_action(agent)
                                   for agent in self.wrapped_cyborg.agents}

        print("Iteration {}".format(str(self.current_it)))

        if "episode_number" in requested_info:
            self.iteration_data["episode_number"] = self.current_ep
        if "iteration_number" in requested_info:
            self.iteration_data["iteration_number"] = self.current_it
        if "action_spaces" in requested_info:
            self.iteration_data["action_spaces"] = self.action_spaces
        if "observation_spaces" in requested_info:
            self.iteration_data["observation_spaces"] = self.observation_spaces
        if "agents_actions" in requested_info:
            self.iteration_data["agents_actions"] = {agt: int(0 if act == None else act) %
                                                     len(self.wrapped_cyborg.agent_actions[agt]) for agt, act in self.agents_actions.items()}
        if "agents_observations" in requested_info:
            self.iteration_data["agents_observations"] = self.agents_observations
        if "agents_rewards" in requested_info:
            self.iteration_data["agents_rewards"] = self.agents_rewards
        if "true_states" in requested_info:
            self.iteration_data["true_states"] = self.true_states
        if "network_graph" in requested_info:
            self.iteration_data["network_graph"] = node_link_data(
                self.wrapped_cyborg.env.environment_controller.state.link_diagram)
        if "team_agent_mapping" in requested_info:
            self.iteration_data["team_agent_mapping"] = self.get_team_agent_mapping(
            )
        if "pz_cyborg_actions" in requested_info:
            self.iteration_data["pz_cyborg_actions"] = self.get_pz_cyborg_actions(
            )
        if "cyborg_actions" in requested_info:
            self.iteration_data["cyborg_actions"] = self.get_cyborg_actions()
        if "cyborg_observations" in requested_info:
            self.iteration_data["cyborg_observations"] = self.get_cyborg_observations(
            )
        if "agents_cyborg_comms" in requested_info:
            self.iteration_data["agents_cyborg_comms"] = self.get_agents_cyborg_comms(
            )
        if "agents_pz_comms" in requested_info:
            self.iteration_data["agents_pz_comms"] = self.get_agents_pz_comms()

        self.current_it += 1
        if self.current_it == self.max_its:
            self.current_ep += 1
            self.current_it = 0

        if self.current_ep >= self.max_eps:
            return None

        if self.current_ep >= self.max_eps and self.current_it > 0:
            return None

        return self.iteration_data


if __name__ == '__main__':

    ce = CyborgEnvironment({}, max_episode=1, max_iteration=5)

    res = ce.next(["true_states"])

    json.dump(res["true_states"], open("./st.json", "w+"))

    # res = None
    # while True:
    #     res = ce.next(["true_states"])
    #     if res != None:
    #         print("\t", end="")
    #         pprint(res["true_states"])
    #         pass
    #     else:
    #         break
