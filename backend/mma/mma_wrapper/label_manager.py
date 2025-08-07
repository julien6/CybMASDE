import importlib
import inspect
import re
import gym

from typing import Any, Dict, List, Tuple
from mma_wrapper.utils import observation, action, label, trajectory, labeled_trajectory


class label_manager:
    """ Manage observation/action one-hot decode/encode as well as labeling observation/action and one-hot encoded observations or actions.
    """

    def __init__(self, action_space: gym.Space = None, observation_space: gym.Space = None):
        self.action_space = action_space
        self.observation_space = observation_space

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

    def one_hot_encode_observation(self, observation: Any, agent: str = None) -> observation:
        """One-hot encode an observation
        Args:
            observation: a readable observation
        Returns:
            observation: a one-hot encoded observation
        """
        raise NotImplementedError

    def one_hot_decode_observation(self, observation: observation, agent: str = None) -> Any:
        """One-hot decode an observation
        Args:
            observation: a one-hot encoded observation
        Returns:
            observation: a readable observation
        """
        raise NotImplementedError

    def one_hot_encode_action(self, action: Any, agent: str = None) -> action:
        """One-hot encode an action
        Args:
            action: a readable action
        Returns:
            action: a one-hot encoded action
        """
        raise NotImplementedError

    def one_hot_decode_action(self, action: action, agent: str = None) -> Any:
        """One-hot decode an action
        Args:
            action: a one-hot encoded action
        Returns:
            action: a readable action
        """
        return NotImplementedError

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
        if save_source:
            return {
                "module_name": module_name,
                "class_name": class_name,
                "source": inspect.getsource(self)
            }
        return {
            "module_name": module_name,
            "class_name": class_name
        }

    @staticmethod
    def from_dict(d: Dict) -> 'label_manager':

        if not 'module_name' in d:
            raise Exception("Module should be given")

        module_name = d['module_name']
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

