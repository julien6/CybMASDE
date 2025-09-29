import importlib.util
import random
import torch

from typing import Callable
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from jopm import JOPM, JointObservationPredictionModel

# TODO : transformer les éléments (jopm, component_functions...) en un seul package installable


class WorldModelEnv(MultiAgentEnv):

    def __init__(self,
                 jopm_path=None,
                 component_functions_path=None,
                 initial_joint_observations_path=None):
        """
        Args:
            jopm_path (str): Path to the saved JOPM model.
            component_functions_path (str): Path to the Python file containing the ComponentFunctions class.
            initial_joint_observations_path (str): Path to initial joint observations, shape (N, obs_dim).
        """
        super().__init__()

        self.jopm = self._load_jopm(jopm_path)
        self.component_functions = self._load_component_functions(
            component_functions_path)

        self.reset()

    def _load_jopm(self, jopm_path) -> JOPM:
        return JOPM.load(jopm_path)

    def _load_component_functions(self, file_path):
        # Dynamically load the ComponentFunctions class from a Python file
        spec = importlib.util.spec_from_file_location(
            "component_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Find the first class that inherits from ComponentFunctions
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and hasattr(obj, "reward_fn") and hasattr(obj, "done_fn") and hasattr(obj, "render_fn"):
                return obj()
        raise ImportError(
            "No valid ComponentFunctions class found in the provided file.")

    def reset(self):
        return self.jopm.reset_internal_state(
            batch_size=1, device=torch.device('cpu'))

    def step(self, action_dict):
        agent_ids = sorted(action_dict.keys())

        act_t = torch.cat([torch.tensor(action_dict[aid], dtype=torch.float32).unsqueeze(
            0) for aid in agent_ids], dim=1)

        observation_dict = self.current_joint_obs
        obs_t = torch.cat([torch.tensor(observation_dict[aid], dtype=torch.float32).unsqueeze(
            0) for aid in agent_ids], dim=1)

        next_joint_obs = self.jopm.predict_next_joint_observation(obs_t, act_t)

        next_observation_dict = {}
        obs_dim_per_agent = next_joint_obs.shape[-1] // len(agent_ids)
        for i, aid in enumerate(agent_ids):
            next_observation_dict[aid] = next_joint_obs[0, i *
                                                        obs_dim_per_agent:(i+1)*obs_dim_per_agent].cpu().numpy()

        # Use component_functions for reward, done, and render
        reward = self.component_functions.reward_fn(
            observation_dict, action_dict, next_observation_dict)
        rewards = {aid: reward for aid in agent_ids}
        infos = {aid: {} for aid in agent_ids}
        self.current_joint_obs = next_observation_dict

        if hasattr(self.component_functions, "done_fn"):
            done = self.component_functions.done_fn(
                observation_dict, action_dict, next_observation_dict)
        else:
            done = False

        dones = {aid: done for aid in agent_ids}
        dones["__all__"] = done

        return next_observation_dict, rewards, dones, infos

    def render(self, mode=None):
        if hasattr(self.component_functions, "render_fn"):
            return self.component_functions.render_fn(self.current_joint_obs, None, None)
