import importlib
import inspect

import itertools
from typing import Callable, Dict, List, Tuple, Union

from mm_wrapper.utils import label, history, history_str
from mm_wrapper.history_pattern import history_pattern

import inspect
from typing import Callable, List, Tuple, get_type_hints


class history_functions:

    def __init__(self):
        self.custom_functions = {
            "policy_functions": [], "reward_functions": []}

    def add_function(self, function: Callable[[Tuple[history, label, str]], Union[List[label], float]]) -> None:

        # if function.__name__ == '<lambda>':
        #     pass
        # else:
        #     pass

        module_name = function.__module__
        custom_function = {
            'function_name': function.__name__,
            'module_name': module_name  # ,
            # 'source_code': source_code
        }
        return_type = get_type_hints(function).get('return')

        if return_type == float:
            self.custom_functions["reward_functions"] += [custom_function]
        elif return_type.__origin__ == list:
            self.custom_functions["policy_functions"] += [custom_function]
        else:
            raise Exception("The return type is nor List[label] nor float")

    def load_function(self, data: Dict) -> Callable[[Tuple[history, label, str]], Union[List[label], float]]:
        function_name = data['function_name']
        module_name = data['module_name']

        module = importlib.import_module(module_name)

        return getattr(module, function_name)

    def get_actions(self, history: history_str, observation_label: label, agent_name: str = None) -> List[label]:
        actions = []
        for custom_function in self.custom_functions["policy_functions"]:
            act = self.load_function(custom_function)(history.split(
                ",") if history is not None else None, observation_label, agent_name)
            if act is None:
                return None
            else:
                actions += [act]

        return list(set(list(itertools.chain.from_iterable(actions))))

    def get_reward(self, history: history_str, agent_name: str = None) -> float:

        if type(history) == list:
            if len(history) > 0 and type(history[0]) == tuple:
                history = list(
                    set(list(itertools.chain.from_iterable([[l1, l2] for l1, l2 in history]))))
            history = ",".join(history)
        if len(history.split(",")) % 2 == 1:
            raise Exception(
                "History should have full (observation, action) couples")

        reward = 0

        for custom_reward_func in self.custom_functions["reward_functions"]:
            rew = self.load_function(custom_reward_func)(history.split(
                ",") if history is not None else None, agent_name)
            if rew is None:
                raise Exception(
                    "Custom reward function should not return None")
            else:
                reward += rew

        # TODO: handle policy
        # for custom_policy_func in self.custom_functions["policy_functions"]:
        #     pass

        return reward

    def to_dict(self) -> Dict:
        return self.custom_functions

    def from_dict(self, data: Dict) -> None:
        self.custom_functions = data


if __name__ == '__main__':

    def manual_custom1(history: history, observation_label: label, agent_name: str) -> List[label]:
        if "o2" in history and observation_label == "o13":
            return ["a1", "a2"]
        else:
            return ["a0"]

    def manual_custom2(history: history, observation_label: label, agent_name: str) -> List[label]:
        if "o3" in history:
            return ["a1", "a2"]
        return ["a12", "a14"]

    def manual_rew_func(history: history, agent_name: str) -> float:
        if "o4" in history:
            return 10
        return -10

    cf = history_functions()
    cf.add_function(manual_custom1)
    cf.add_function(manual_custom2)
    print(cf.get_actions("o2,a3,o3,a3,o4,a4", "o13"))

    cf.add_function(manual_rew_func)
    print(cf.get_reward("o2,a3,o3,a3,o4,a4"))
