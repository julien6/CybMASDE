import itertools
from typing import Callable, Dict, List, Tuple, Union

import numpy

from mm_wrapper.history_function import history_functions
from mm_wrapper.history_rule import history_rules
from mm_wrapper.utils import label, history, history_pattern_str
from mm_wrapper.history_pattern import history_pattern, history_patterns


class observable_reward_function:

    def __init__(self) -> None:
        self.history_functions = history_functions()
        self.history_patterns = history_patterns()
        self.history_rules = history_rules()

    def add_custom_function(self, function: Callable[[history], float]) -> 'observable_reward_function':
        self.history_functions.add_function(function)
        return self

    def add_pattern(self, history_pattern_string: history_pattern_str) -> 'observable_reward_function':
        self.history_patterns.add_pattern(history_pattern_string)
        return self

    def add_rule(self, history_pattern_string: str, observation: label, actions: List[label]) -> 'observable_reward_function':
        self.history_rules.add_rule(
            history_pattern_string, observation, actions)
        return self

    def add_rules(self, rules: List[Tuple[str, label, List[label]]]) -> 'observable_reward_function':
        for history_pattern_string, observation_label, actions in rules:
            self.history_rules.add_rule(
                history_pattern_string, observation_label, actions)
        return self

    def reward(self, history: history) -> float:
        if history is not None:
            pattern_rew = self.history_patterns.get_reward(history)
            if pattern_rew is None:
                pattern_rew = 0
            func_rew = self.history_functions.get_reward(history)
            if func_rew is None:
                func_rew = 0
            rule_rew = self.history_rules.get_reward(history)
            if rule_rew is None:
                rule_rew = 0
            return pattern_rew + func_rew + rule_rew

    def to_dict(self) -> Dict:
        return {
            "history_functions": self.history_functions.to_dict(),
            "history_patterns": self.history_patterns.to_dict(),
            "history_rules": self.history_rules.to_dict()
        }

    @staticmethod
    def from_dict(data: Dict) -> 'observable_reward_function':
        opc = observable_reward_function()
        opc.history_functions.from_dict(data["history_functions"])
        opc.history_patterns.from_dict(data["history_patterns"])
        opc.history_rules.from_dict(data["history_rules"])
        return opc

        # history = history.split(",")
        # on_build_history = []
        # still_match = True
        # i = 0
        # while i < len(history):
        #     observation = history[i]
        #     expected_actions = self.get_actions(on_build_history, observation)

        #     on_build_history += [observation]

        #     if len(expected_actions) == 0:
        #         break

        #     i += 1

        #     action = history[i]
        #     on_build_history += [action]
        #     if action not in expected_actions:
        #         still_match = False
        #         break

        #     i += 1

        # if not still_match:
        #     reward = -reward

        return reward


if __name__ == '__main__':

    # hps = history_patterns()
    # hps.add_pattern("[0,1,2,3,4,5,6,7,8,9](1,1)")
    # print(hps.get_actions("0,1,2", "3"))

    def sub_reward_fun(hist: history) -> float:
        if "14" in hist:
            return 15
        return -10

    orf = observable_reward_function()
    orf.add_pattern(
        "[0,[#any](0,*),[9,#any](1,1)](1,1)").add_custom_function(sub_reward_fun).add_rule("[0,1,5,6](1,1)", "9", ["14"])

    print(orf.reward("0,1,5,6,9,14"))

    orf_dict = orf.to_dict()
    print(orf_dict)

    orf1 = observable_reward_function.from_dict(orf_dict)

    orf_dict1 = orf1.to_dict()
    print(orf_dict1)

    print(orf1.reward("0,1,5,6,9,14"))
