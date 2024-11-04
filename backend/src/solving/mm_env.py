import itertools
import random
import yaml
import subprocess
import sys
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from typing import Dict
from marllib.envs.base_env.mpe import RLlibMPE
from marllib.marl.common import dict_update
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl.models import BaseRNN, BaseMLP, CentralizedCriticRNN, CentralizedCriticMLP, ValueDecompRNN, \
    ValueDecompMLP, JointQMLP, JointQRNN, DDPGSeriesRNN, DDPGSeriesMLP
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune import register_env
from copy import deepcopy
from tabulate import tabulate
from typing import Dict, Tuple
from mm_wrapper.observable_reward_function import observable_reward_function
from mm_wrapper.observable_policy_constraint import observable_policy_constraint
from mm_wrapper.osr_model import obligation, organizational_model, permission
from mm_wrapper.deontic_specifications_ttl import deontic_specifications_ttl
from mm_wrapper.utils import OBLIGATION_REWARD_FACTOR


class RLlibMPE_action_wrapper(RLlibMPE):

    def __init__(self, env: RLlibMPE, organizational_model: organizational_model = None, bonus: int = 10, malus: int = -10):
        self.env = env
        self.organizational_model = organizational_model
        if organizational_model is not None:
            self.deontic_specifications_ttl = deontic_specifications_ttl(organizational_model)
        self.bonus = bonus
        self.malus = malus
        self.last_observation_dict = {}
        self.histories = {agent: [] for agent in self.agents}

    def step(self, action_dict):

        corrected_actions = deepcopy(action_dict)
        reward_corrections = {
            agent_name: 0 for agent_name in list(action_dict.keys())}
        for agent, action in action_dict.items():

            if self.organizational_model is not None:

                obs = self.last_observation_dict[agent]

                if "obs" in list(obs.keys()):
                    obs = obs["obs"]

                obligations = []
                permissions = []
                if self.organizational_model.deontic_specifications.obligations is not None:
                    obligations = [obligation for obligation, agent_names in self.organizational_model.deontic_specifications.obligations.items(
                    ) if agent in agent_names]
                if self.organizational_model.deontic_specifications.permissions is not None:
                    permissions = [permission for permission, agent_names in self.organizational_model.deontic_specifications.permissions.items(
                    ) if agent in agent_names]

                if len(set([d.role for d in obligations+permissions])) != 1:
                    raise Exception(
                        "An agent should be mapped to one role")

                # Dealing with the associated role
                role = obligations[0].role
                opc = self.organizational_model.structural_specifications.roles[role]
                mapped_actions = opc.get_actions(
                    self.histories[agent], obs, agent)
                if mapped_actions is not None and len(mapped_actions) > 0:
                    mapped_action = random.choice(mapped_actions)
                    corrected_actions[agent] = mapped_action

                self.histories.setdefault(agent, [])
                # TODO: remove the str after label <-> obs established
                self.histories[agent] += [str(obs),
                                          str(corrected_actions[agent])]

                # Dealing with the associated missions
                for ds in obligations+permissions:
                    mission = ds.mission
                    if mission is None:
                        continue
                    orfs = list(
                        set(list(itertools.chain.from_iterable([[sch.goals[goal] for goal in sch.mission_to_goals[mission]] for sch_tag,
                            sch in self.organizational_model.functional_specifications.social_scheme.items() if mission in sch.missions]))))
                    reward_corrections[agent] = (OBLIGATION_REWARD_FACTOR if type(
                        ds) == obligation else 1) * sum([x if x is not None else 0 for x in [o.reward(self.histories[agent]) for o in orfs]])

        if self.organizational_model is not None:
            self.deontic_specifications_ttl.decrease()
            self.organizational_model = self.deontic_specifications_ttl.osr_model

        # Take a step in the environment with modified actions
        obs, rewards, dones, info = self.env.step(corrected_actions)

        rewards = {agent_name: reward + reward_corrections.get(agent_name, 0)
                   for agent_name, reward in rewards.items()}

        self.last_observation_dict = obs

        return obs, rewards, dones, info

    def reset(self):
        self.last_observation_dict = self.env.reset()
        return self.last_observation_dict

    def __getattr__(self, name):
        return getattr(self.env, name)


SYSPARAMs = deepcopy(sys.argv)


def set_ray(config: Dict):
    """
    function of combining ray config with other configs
    :param config: dictionary of config to be combined with
    """
    # default config
    with open(os.path.join(os.path.dirname(local_file), "ray/ray.yaml"), "r") as f:
        ray_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    # user config
    user_ray_args = {}
    for param in SYSPARAMs:
        if param.startswith("--ray_args"):
            if "=" in param:
                key, value = param.split(".")[1].split("=")
                user_ray_args[key] = value
            else:  # local_mode
                user_ray_args[param.split(".")[1]] = True

    # update config
    ray_config_dict = dict_update(ray_config_dict, user_ray_args, True)

    for key, value in ray_config_dict.items():
        config[key] = value

    return config


def get_conda_env_path(env_name):
    # Run the 'conda info --envs' command
    result = subprocess.run(['conda', 'info', '--envs'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check if the command was successful
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to get conda environments: {result.stderr}")

    # Parse the output to find the path of the specified environment
    for line in result.stdout.splitlines():
        if env_name in line:
            # The environment name might appear more than once, ensure to pick the correct line
            parts = line.split()
            if len(parts) > 1 and parts[0] == env_name:
                return parts[-1]

    raise ValueError(f"Environment '{env_name}' not found")


# Example usage
env_name = 'marllib'
env_path = ""
try:
    env_path = get_conda_env_path(env_name)
    print(f"The path of the '{env_name}' environment is: {env_path}")
except Exception as e:
    print(e)

local_file = f"{env_path}/lib/python3.8/site-packages/marllib/marl/__init__.py"


def make_env(
        environment_name: str,
        map_name: str,
        force_coop: bool = False,
        organizational_model: organizational_model = None,
        **env_params
) -> Tuple[MultiAgentEnv, Dict]:
    """
    construct the environment and register.
    Args:
        :param environment_name: name of the environment
        :param map_name: name of the scenario
        :param force_coop: enforce the reward return of the environment to be global
        :param env_params: parameters that can be pass to the environment for customizing the environment

    Returns:
        Tuple[MultiAgentEnv, Dict]: env instance & env configuration dict
    """

    # default config
    env_config_file_path = os.path.join(os.path.dirname(local_file),
                                        "../envs/base_env/config/{}.yaml".format(environment_name))
    if not os.path.exists(env_config_file_path):
        env_config_file_path = os.path.join(os.path.dirname(local_file),
                                            "../../examples/config/env_config/{}.yaml".format(environment_name))

    with open(env_config_file_path, "r") as f:
        env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    print("TODO: ", env_config_dict)

    # update function-fixed config
    env_config_dict["env_args"] = dict_update(
        env_config_dict["env_args"], env_params, False)

    # user commandline config
    user_env_args = {}
    for param in SYSPARAMs:
        if param.startswith("--env_args"):
            key, value = param.split(".")[1].split("=")
            user_env_args[key] = value

    # update commandline config
    env_config_dict["env_args"] = dict_update(
        env_config_dict["env_args"], user_env_args, True)
    env_config_dict["env_args"]["map_name"] = map_name
    env_config_dict["force_coop"] = force_coop

    # combine with exp running config
    env_config = set_ray(env_config_dict)

    # initialize env
    env_reg_ls = []
    check_current_used_env_flag = False
    for env_n in ENV_REGISTRY.keys():
        if isinstance(ENV_REGISTRY[env_n], str):  # error
            info = [env_n, "Error", ENV_REGISTRY[env_n], "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/base_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
        else:
            info = [env_n, "Ready", "Null", "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/base_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
            if env_n == env_config["env"]:
                check_current_used_env_flag = True

    print(tabulate(env_reg_ls,
                   headers=['Env_Name', 'Check_Status', "Error_Log",
                            "Config_File_Location", "Env_File_Location"],
                   tablefmt='grid'))

    if not check_current_used_env_flag:
        raise ValueError(
            "environment \"{}\" not installed properly or not registered yet, please see the Error_Log below".format(
                env_config["env"]))

    env_reg_name = env_config["env"] + "_" + env_config["env_args"]["map_name"]

    if env_config["force_coop"]:
        register_env(env_reg_name, lambda _: RLlibMPE_action_wrapper(
            COOP_ENV_REGISTRY[env_config["env"]](env_config["env_args"]), organizational_model))
        env = COOP_ENV_REGISTRY[env_config["env"]](env_config["env_args"])
    else:
        register_env(env_reg_name, lambda _: RLlibMPE_action_wrapper(
            ENV_REGISTRY[env_config["env"]](env_config["env_args"]), organizational_model))
        env = ENV_REGISTRY[env_config["env"]](env_config["env_args"])

    return env, env_config
