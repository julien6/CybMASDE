# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Discrete, Box
import supersuit as ss
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from pettingzoo.butterfly.pistonball import pistonball
from gym import spaces

import time

REGISTRY = {}
REGISTRY["pistonball"] = pistonball.parallel_env

policy_mapping_dict = {
    "pistonball": {
        "description": "Pistonball",
        "team_prefix": ("piston_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    }
}


class RLlibBTF(MultiAgentEnv):

    def __init__(self, env_config):
        map = env_config["map_name"]
        env_config.pop("map_name", None)
        env = REGISTRY[map](**env_config)

        # keep obs and action dim same across agents
        # pad_action_space_v0 will auto mask the padding actions
        # env = ss.pad_observations_v0(env)
        # env = ss.pad_action_space_v0(env)

        self.env = ParallelPettingZooEnv(env)

        self.action_space = spaces.Box(
            self.env.action_spaces[self.env.agents[0]].low[0],
            self.env.action_spaces[self.env.agents[0]].high[0],
            shape=(self.env.action_spaces[self.env.agents[0]].shape[0],),
            dtype=self.env.action_spaces[self.env.agents[0]].dtype)

        self.observation_space = GymDict({"obs": Box(
            low=self.env.observation_spaces[self.env.agents[0]].low[0][0][0],
            high=self.env.observation_spaces[self.env.agents[0]].high[0][0][0],
            shape=(self.env.observation_spaces[self.env.agents[0]].shape),
            dtype=self.env.observation_spaces[self.env.agents[0]].dtype)
        })

        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i in self.agents:
            obs[i] = {"obs": original_obs[i]}
        return obs

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        rewards = {}
        obs = {}
        for key in action_dict.keys():
            rewards[key] = r[key]
            obs[key] = {
                "obs": o[key]
            }
        dones = {"__all__": d["__all__"]}
        return obs, rewards, dones, info

    def close(self):
        self.env.close()

    def render(self, mode=None):
        self.env.render()
        time.sleep(0.05)
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 30,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info


legal_scenarios = ["pistonball"]


class RLlibBTF_FCOOP(RLlibBTF):

    def __init__(self, env_config):
        if env_config["map_name"] not in legal_scenarios:
            raise ValueError("must in: 1.pistonball")
        super().__init__(env_config)

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        reward = 0
        for key in r.keys():
            reward += r[key]
        rewards = {}
        obs = {}
        for key in action_dict.keys():
            rewards[key] = reward/self.num_agents
            obs[key] = {
                "obs": o[key]
            }
        dones = {"__all__": d["__all__"]}
        return obs, rewards, dones, info


if __name__ == "__main__":
    env = RLlibBTF({"map_name": "pistonball"})
    env.reset()    
    for i in range(100):
        action = {i: env.action_space.sample() for i in env.agents}
        obs, reward, done, info = env.step(action)
        env.render()
        if done["__all__"]:
            env.reset()
    env.close()
    print("done")
