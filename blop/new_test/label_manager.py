import importlib
import inspect
import os
import gym
import numpy as np

from typing import Dict, List, Tuple, Union, Any
import re

observation = Union[int, np.ndarray]
action = Union[int, np.ndarray]
label = str
pattern_trajectory = str
trajectory = List[Tuple[observation, action]]
labeled_trajectory = Union[List[Tuple[label, label]], List[label], str]
trajectory_pattern_str = str
trajectory_str = str


class label_manager:
    """Example label manager that extends the base label_manager class.
    This class should implement the methods for one-hot encoding and decoding
    observations and actions.
    """

    def __init__(self):
        self.action_space = gym.spaces.Discrete(6)

        high = np.ones((96,)) * 10
        low = np.ones((96,)) * -10
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def one_hot_encode_observation(self, observation, agent=None):
        # Example implementation of one-hot encoding an observation
        return super().one_hot_encode_observation(observation, agent)

    def one_hot_decode_observation(self, observation, agent=None):
        # Example implementation of one-hot decoding an observation
        return super().one_hot_decode_observation(observation, agent)

    def one_hot_encode_action(self, action, agent=None):
        # Example implementation of one-hot encoding an action
        return super().one_hot_encode_action(action, agent)

    def one_hot_decode_action(self, action, agent=None):
        # Example implementation of one-hot decoding an action
        return super().one_hot_decode_action(action, agent)

    def one_hot_decode_trajectory(self, trajectory: 'trajectory', agent: str = None) -> List[Tuple[Any, Any]]:
        """One-hot decode a one-hot encoded trajectory into a trajectory of readable (observation, action) couples
        """
        return [(self.one_hot_decode_observation(_observation), self.one_hot_decode_action(_action)) for _observation, _action in trajectory]

    def one_hot_encode_trajectory(self, trajectory: List[Tuple[Any, Any]], agent: str = None) -> trajectory:
        """One-hot encode a readable trajectory into a one-hot encoded trajectory
        Args:
            trajectory: a readable trajectory
        Returns
            trajectory: a one-hot encoded trajectory
        """
        return [(self.one_hot_encode_observation(_observation), self.one_hot_encode_action(_action)) for _observation, _action in trajectory]

    def label_observation(self, observation: observation, agent: str = None) -> label:
        """Label a one-hot encoded observation into label
        Args:
            observation: a one-hot encoded observation
        Returns:
            label: the labelized observation
        """
        raise NotImplementedError

    def unlabel_observation(self, observation: label, agent: str = None) -> List[observation]:
        """Unlabel a one-hot encoded observation into a list of one-hot encoded observation
        Args:
            observation: the label to be mapped to the matching observations
        Returns:
            List[observation]: a list of one-hot encoded observations
        """
        raise NotImplementedError

    def label_action(self, action: action, agent: str = None) -> label:
        """Label a one-hot encoded action into label
        Args:
            action: a one-hot encoded action
        Returns:
            label: the labelized action
        """
        raise NotImplementedError

    def unlabel_action(self, action: label, agent: str = None) -> List[action]:
        """Unlabel a one-hot encoded action into a list of one-hot encoded action
        Args:
            action: the label to be mapped to the matching action
        Returns:
            List[action]: a list of one-hot encoded action
        """
        raise NotImplementedError

    def label_trajectory(self, trajectory: trajectory, agent: str = None) -> labeled_trajectory:
        """Label a one-hot encoded trajectory into a labeled trajectory
        Args:
            trajectory: a one-hot encoded trajectory
        Returns
            trajectory: a labeled trajectory
        """
        return [(self.label_observation(observation), self.label_action(action)) for observation, action in trajectory]

    def unlabel_trajectory(self, labeled_trajectory: labeled_trajectory, agent: str = None) -> trajectory:
        """Unlabel a labeled trajectory into a one-hot encoded trajectory
        Args:
            trajectory: a labeled trajectory
        Returns
            trajectory: a one-hot encoded trajectory
        """
        return [(self.unlabel_observation(observation)[0], self.unlabel_action(action)[0]) for observation_label, action_label in labeled_trajectory.items()]

    def to_dict(self, save_source=False) -> Dict:
        module_name = self.__class__.__module__
        class_name = self.__class__.__name__
        source_file = os.path.abspath(__file__)

        if save_source:
            return {
                "source_file": source_file,
                "module_name": module_name,
                "class_name": class_name,
                "source": inspect.getsource(self.__class__)
            }
        return {
            "source_file": source_file,
            "module_name": module_name,
            "class_name": class_name
        }

    @staticmethod
    def from_dict(d: Dict) -> 'label_manager':

        source_file_path = d.get('source_file', None)

        if not 'module_name' in d:
            raise Exception("Module should be given")

        module_name = d['module_name']

        # Load the source file before importing the module
        if source_file_path:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                module_name, source_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)

        if 'source' in d:
            match = re.search(
                r"^\s*class\s+([a-zA-Z_]\w*)\s*[\(:]", d['source'], re.MULTILINE)
            if match:
                function_name = match.group(1)
            lcs = {}
            exec(d["source"], module.__dict__, lcs)
            _lbl_mngr_class = lcs.get(d)
        elif 'class_name' in d:
            function_name = d['class_name']
            _lbl_mngr_class = getattr(module, function_name)
        return _lbl_mngr_class()
