import itertools
from typing import Callable, Dict, List, Tuple

from mm_wrapper.history_function import history_functions
from mm_wrapper.history_rule import history_rules
from mm_wrapper.utils import label, history, history_pattern_str, history_str
from mm_wrapper.history_pattern import history_pattern, history_patterns


class observable_policy_constraint:

    def __init__(self) -> None:
        self.history_functions = history_functions()
        self.history_patterns = history_patterns()
        self.history_rules = history_rules()

    def add_custom_function(self, function: Callable[[Tuple[history, label]], Tuple[List[label], float]]) -> 'observable_policy_constraint':
        self.history_functions.add_function(function)
        return self

    def add_pattern(self, history_pattern_string: history_pattern_str) -> 'observable_policy_constraint':
        self.history_patterns.add_pattern(history_pattern_string)
        return self

    def add_rule(self, history_pattern_string: str, observation: label, actions: List[label]) -> 'observable_policy_constraint':
        self.history_rules.add_rule(
            history_pattern_string, observation, actions)
        return self

    def add_rules(self, rules: List[Tuple[str, label, List[label]]]) -> 'observable_policy_constraint':
        for history_pattern_string, observation_label, actions in rules:
            self.history_rules.add_rule(
                history_pattern_string, observation_label, actions)
        return self

    def get_actions(self, history: history, observation_label: label, agent_name: str = None) -> List[label]:
        if history is not None:
            if len(history) > 0:
                history = ""
            else:
                history = list(itertools.chain.from_iterable(
                    [[l1, l2] for l1, l2 in history]))
                history = ",".join(history)
        function_actions = self.history_functions.get_actions(
            history, observation_label, agent_name)
        pattern_actions = self.history_patterns.get_actions(
            history, observation_label, agent_name)
        rule_actions = self.history_rules.get_actions(
            history, observation_label, agent_name)

        return list(set(([] if function_actions is None else function_actions)
                        + ([] if pattern_actions is None else pattern_actions)
                        + ([] if rule_actions is None else rule_actions)))

    def to_dict(self) -> Dict:
        return {
            "history_functions": self.history_functions.to_dict(),
            "history_patterns": self.history_patterns.to_dict(),
            "history_rules": self.history_rules.to_dict()
        }

    @staticmethod
    def from_dict(data: Dict) -> 'observable_policy_constraint':
        opc = observable_policy_constraint()
        opc.history_functions.from_dict(data["history_functions"])
        opc.history_patterns.from_dict(data["history_patterns"])
        opc.history_rules.from_dict(data["history_rules"])
        return opc


if __name__ == '__main__':

    def manual_custom1(history: history, observation_label: label) -> List[label]:
        if history is not None and len(history) > 0:
            if "o2" in history and observation_label == "o13":
                return ["a1", "a2"]
        else:
            return ["a0"]

    opc = observable_policy_constraint()

    opc.add_rule("[o1,a1,[#any](0,*),o3](1,1)", "o4", ["a1", "a2", "a3"])\
        .add_custom_function(manual_custom1)\
        .add_pattern("[0,1,[[2,3](1,1),4,5](1,1),6](1,1)")

    # print(opc.get_actions(["o1", "a1", "o2", "o3"], "o13"))
    print(opc.get_actions(None, "1"))

    dict_opc = opc.to_dict()
    print(dict_opc)

    opc1 = observable_policy_constraint.from_dict(dict_opc)

    dict_opc1 = opc1.to_dict()

    print(dict_opc1)
