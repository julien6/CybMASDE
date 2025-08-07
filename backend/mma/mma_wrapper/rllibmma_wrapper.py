import json
import numpy as np
import os
import time

from copy import copy, deepcopy
from typing import Dict, List, TypedDict
from mma_wrapper.organizational_model import organizational_model
from mma_wrapper.organizational_specification_logic import goal_logic, role_logic
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class RLlibMMA_wrapper(MultiAgentEnv):

    temm_enabled = False

    def __init__(self, env: MultiAgentEnv, organizational_model: organizational_model = None, render_mode=None, analysis_folder=None):
        self.env = env
        self.metadata = env.metadata
        self.metadata["render.modes"] = ['human', 'rgb_array']

        self.organizational_model = organizational_model
        self.last_observations = {}
        self.histories = {agent: [] for agent in self.agents}

        self.render_mode = render_mode
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.agents = self.env.agents
        self.num_agents = self.env.num_agents
        self.env_config = self.env.env_config
        self.analysis_folder = analysis_folder
        self.episode_number = 0

        if self.temm_enabled and not os.path.exists(os.path.join(self.analysis_folder, "trajectories")):
            os.mkdir(os.path.join(self.analysis_folder))
            os.mkdir(os.path.join(self.analysis_folder, "trajectories"))

        class os_logics(TypedDict):
            role_logic: 'role_logic'
            goal_logics: List['goal_logic']

        self.agent_to_os_logics: Dict[str, os_logics] = {}
        if organizational_model is not None:
            for obligation in self.organizational_model.deontic_specifications.obligations:
                agents = obligation.agents
                goal_logics = []
                for sch_tag, sch in self.organizational_model.functional_specifications.social_scheme.items():
                    for mission in obligation.missions:
                        goal_logics.extend([self.organizational_model.functional_specifications.goals[goal]
                                            for goal in sch.mission_to_goals[mission]])
                role_logic = self.organizational_model.structural_specifications.roles[
                    obligation.role]
                time_constraint = obligation.time_constraint

                for agent in agents:
                    if not agent in self.agent_to_os_logics:
                        self.agent_to_os_logics[agent] = {
                            "role_logic": None, "goals_logics": []}
                    self.agent_to_os_logics[agent]["role_logic"] = role_logic
                    self.agent_to_os_logics[agent]["goals_logics"].extend(
                        goal_logics)

            for permission in self.organizational_model.deontic_specifications.permissions:
                agents = permission.agents
                goal_logics = []
                for sch_tag, sch in self.organizational_model.functional_specifications.social_scheme.items():
                    for mission in permission.missions:
                        goal_logics.extend([self.organizational_model.functional_specifications.goals[goal]
                                            for goal in sch.mission_to_goals[mission]])
                role_logic = self.organizational_model.structural_specifications.roles[
                    permission.role]
                time_constraint = permission.time_constraint

                for agent in agents:
                    if not agent in self.agent_to_os_logics:
                        self.agent_to_os_logics[agent] = {
                            "role_logic": None, "goal_logics": []}
                    self.agent_to_os_logics[agent]["role_logic"] = role_logic
                    self.agent_to_os_logics[agent]["goal_logics"].extend(
                        goal_logics)

    def step(self, action_dict):
        # 1 - update the action space according to hard constraint role rules
        corrected_actions = action_dict
        if self.organizational_model:
            for agent, action in action_dict.items():
                if agent in self.agent_to_os_logics:
                    corrected_action = self.agent_to_os_logics[agent]['role_logic'].next_action(
                        trajectory=self.histories[agent], observation=self.last_observations[agent], agent_name=agent)
                    if corrected_action is not None:
                        corrected_actions[agent] = corrected_action

        # 2 - run step with regular step function
        obs, rewards, dones, info = self.env.step(corrected_actions)

        # 3 - update the rewards to soft constraint role rules and goal rules
        corrected_rewards = rewards
        if self.organizational_model:
            for agent, reward in rewards.items():
                if agent in self.agent_to_os_logics:

                    # handling soft constraint role rules
                    role_reward_correction = self.agent_to_os_logics[agent].get('role_logic').next_reward(
                        trajectory=self.histories[agent], observation=self.last_observations[agent], action=corrected_actions[agent], agent_name=agent)

                    # handling goal rules
                    goal_reward_correction = 0
                    for goal_logic in self.agent_to_os_logics[agent].get('goal_logics', []):
                        goal_reward_correction += goal_logic.next_reward(
                            self.histories[agent], agent_name=agent)

                    corrected_rewards[agent] += goal_reward_correction + \
                        role_reward_correction

        for agent, o in obs.items():
            # Save last observation and chosen/corrected action as a new transition in history
            self.histories.setdefault(agent, [])
            self.histories[agent] += [(self.last_observations[agent],
                                       corrected_actions[agent])]
            if "obs" in o:
                o = o.get('obs')
                if isinstance(o, np.ndarray):
                    o = o.tolist()
            self.last_observations[agent] = o

        return obs, corrected_rewards, dones, info

    def reset(self):

        _last_observations = self.env.reset()
        self.last_observations = {}
        for agent, obs in _last_observations.items():
            if "obs" in obs:
                obs = obs.get('obs')
                if isinstance(obs, np.ndarray):
                    obs = obs.tolist()
            self.last_observations[agent] = obs

        if len(self.histories[list(_last_observations.keys())[0]]) > 0:
            if self.temm_enabled:
                json.dump(self.histories, open(os.path.join(
                    self.analysis_folder, "trajectories", f"trajectories_{self.episode_number}.json"), "w+"))
                self.episode_number += 1

            for agent, obs in _last_observations.items():
                self.histories[agent] = []

        return _last_observations

    def close(self):
        self.env.close()

    def render(self, mode=None):
        m = self.render_mode
        if mode is not None:
            m = mode
        if m is None:
            rendered = self.env.render()
        else:
            rendered = self.env.render(mode=m)
        return rendered

    def get_env_info(self):
        return self.env.env_info

    def __getattr__(self, name):
        return getattr(self.env, name)
