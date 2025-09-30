import importlib.util
import random
from mma_wrapper.label_manager import label_manager
import torch

from typing import Callable
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from world_model.component_functions import ComponentFunctions
from world_model.jopm import JOPM


class WorldModelEnv(MultiAgentEnv):

    def __init__(self,
                 jopm_path=None,
                 component_functions_path=None,
                 label_manager_path=None):
        """
        Args:
            jopm_path (str): Path to the saved JOPM model.
            component_functions_path (str): Path to the Python file containing the ComponentFunctions class.
            label_manager_path (str): Path to the Python file containing the label_manager class.
        """
        super().__init__()

        self.jopm = self._load_jopm(jopm_path)
        self.component_functions = self._load_component_functions(
            component_functions_path, label_manager_path)

        initial_observation = self.reset()

        self.agents = [f"agent_{i}" for i in range(
            0, initial_observation.shape[0])]

        self.current_joint_obs = None

    def _load_jopm(self, jopm_path) -> JOPM:
        return JOPM.load(jopm_path)

    def _load_component_functions(self, component_functions_path: str, label_manager_path: str) -> ComponentFunctions:

        def load_label_manager(file_path: str) -> label_manager:
            spec = importlib.util.spec_from_file_location(
                "label_manager", file_path)

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Recherche la classe label_manager dans le module
            lbl_manager: label_manager = module
            if hasattr(module, "label_manager"):
                lbl_manager = getattr(
                    module, "label_manager")()
            else:
                raise ImportError(
                    "No label_manager class found in ", file_path)
            return lbl_manager

        lbl_manager = load_label_manager(label_manager_path)

        spec = importlib.util.spec_from_file_location(
            "ComponentFunctions", component_functions_path)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Recherche la classe ComponentFunctions dans le module
        component_functions: ComponentFunctions = module
        if hasattr(module, "ComponentFunctions"):
            component_functions = getattr(
                module, "ComponentFunctions")(label_manager=lbl_manager)
        else:
            raise ImportError(
                "No ComponentFunctions class found in ", component_functions_path)
        return component_functions

    def reset(self):
        self.current_joint_obs = self.jopm.reset_internal_state(
            batch_size=1, device=torch.device('cpu'))
        return self.current_joint_obs

    def step(self, action_dict):
        agent_ids = sorted(action_dict.keys())

        act_t = torch.cat([torch.tensor([action_dict[aid]], dtype=torch.float32)
                          for aid in agent_ids], dim=0).unsqueeze(0)

        observation_dict = self.current_joint_obs
        print("observation_dict: ", observation_dict)
        if isinstance(observation_dict, dict):
            obs_t = torch.cat([torch.tensor(observation_dict[aid],
                                            dtype=torch.float32) for aid in agent_ids], dim=0).unsqueeze(0)
        else:
            obs_t = torch.cat([torch.tensor(observation_dict[int(aid.split("_")[1])],
                                            dtype=torch.float32) for aid in agent_ids], dim=0).unsqueeze(0)
        print("obs_t: ", obs_t)

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
