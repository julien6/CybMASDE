import json
import os
import gym
import importlib.util
from marllib import marl


from typing import List

from mma_wrapper.organizational_model import organizational_model
from world_model.project_configuration import Configuration
from marllib.patch.rllib.policy.torch_policy import TorchPolicy


class joint_policy:

    def __init__(self):
        pass

    def from_dict(self, data):
        pass

    def to_dict(self):
        pass

    def next_action(self, joint_observation):
        """Compute the next joint action based on the joint observation."""
        pass


class random_joint_policy(joint_policy):

    def __init__(self, action_space: gym.Space, agents: List):
        self.action_space = action_space
        self.agents = agents

    def next_action(self, joint_observation):
        return [self.action_space.sample() for _ in range(0, len(self.agents))]


class marllib_joint_policy(joint_policy):
    def __init__(self, path: str, configuration: Configuration, agents: List):

        self.agents = agents

        params_path = os.path.join(path, "model/params.json")
        checkpoint_name = [f for f in os.listdir(os.path.join(
            path, "model")) if f.startswith("checkpoint_")][0]
        model_path = os.path.join(
            path, "model", checkpoint_name, f"checkpoint-{int(checkpoint_name.split('_')[1])}")
        self.params = json.load(open(params_path, 'r'))

        algorithm = self.params["model"]["custom_model_config"]['algorithm']

        organizational_specifications = organizational_model.from_dict(json.load(open(os.path.join(
            configuration.training.organizational_specifications, "organizational_specifications.json"), "r")))

        env = marl.make_env(
            environment_name="cybmasde",
            map_name="default",
            force_coop=False,
            organizational_model=organizational_specifications,
            jopm_path=os.path.join(configuration.common.project_path, os.path.dirname(
                configuration.modelling.generated_environment.world_model.jopm.initial_joint_observations)),
            component_functions_path=os.path.join(
                configuration.common.project_path, configuration.modelling.generated_environment.component_functions_path),
            label_manager_path=os.path.join(
                configuration.common.project_path, configuration.common.label_manager),
            no_print=True
        )

        best_hp = json.load(open(os.path.join(path, "model_config.json"), "r"))
        algo = marl.algos.__getattribute__(algorithm)(
            hyperparam_source="common", **best_hp["algorithms"][algorithm]["algorithm"])

        model = marl.build_model(
            env, algo, best_hp["algorithms"][algorithm]["model"])

        params_path = os.path.join(configuration.common.project_path,
                                   configuration.training.joint_policy, "model/params.json")

        checkpoint_name = [f for f in os.listdir(os.path.join(
            configuration.common.project_path, configuration.training.joint_policy, "model")) if f.startswith("checkpoint_")][0]

        model_path = os.path.join(configuration.common.project_path, configuration.training.joint_policy,
                                  "model", checkpoint_name, f"checkpoint-{int(checkpoint_name.split('_')[1])}")

        self.policy: TorchPolicy = algo.get_policy(env, model,
                                                   restore_path={
                                                       'params_path': params_path,
                                                       'model_path': model_path,
                                                   },
                                                   local_mode=True,
                                                   share_policy="group")

    def next_action(self, joint_observation):
        obs = {}
        for i in self.agents:
            obs[i] = {"obs": joint_observation[int(i)]}

        actions = self.policy.compute_actions(
            obs, policy_id="shared_policy")

        return [int(act) for agt, act in actions.items()]
