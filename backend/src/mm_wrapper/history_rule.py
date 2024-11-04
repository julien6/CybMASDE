from typing import Dict, List, Tuple

from mm_wrapper.utils import MATCH_REWARD
from mm_wrapper.utils import label, history, history_pattern_str, history_str
from mm_wrapper.history_pattern import history_pattern


class history_rules:

    def __init__(self) -> None:
        self.rules: Dict[Tuple[str, label], List[label]] = {}

    def add_rule(self, history_pattern_string: str, observation: label, actions: List[label]) -> None:
        self.rules[(history_pattern_string, observation)] = actions

    def get_actions(self, history: history_str, observation_label: label, agent_name: str = None) -> List[label]:
        actions = [actions for hist_obs, actions in self.rules.items() if history_pattern(
            hist_obs[0]).match(history, None)[0] and hist_obs[1] == observation_label]
        return actions[0] if len(actions) > 0 else None

    def get_reward(self, history: history_str) -> float:

        if type(history) == list:
            if len(history) > 0 and type(history[0]) == tuple:
                history = list(
                    set(list(itertools.chain.from_iterable([[l1, l2] for l1, l2 in history]))))
            history = ",".join(history)
        if len(history.split(",")) % 2 == 1:
            raise Exception(
                "History should have full (observation, action) couples")

        history = history.split(",")
        observation_label = history[-2]
        act_label = history[-1]

        history = ",".join(history[:-2])

        reward = 0

        for hist_pattern_obs_label, act_labels in self.rules.items():
            if history_pattern(hist_pattern_obs_label[0]).match(history, observation_label) and act_label in act_labels:
                reward += MATCH_REWARD

        if len(self.rules) > 0 and reward == 0:
            reward -= MATCH_REWARD

        return reward

    def to_dict(self) -> Dict:
        return self.rules

    def from_dict(self, data: Dict) -> None:
        self.rules = data


if __name__ == '__main__':

    hr = history_rules()
    hr.add_rule("[o1,[#any](0,*),a1](1,1)", "o4", ["a1", "a2", "a3"])
    print(hr.get_actions("o1,k,k,a1", "o4"))
