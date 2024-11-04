import importlib

from typing import Callable, Dict, List, Tuple, Union

from mm_wrapper.observations_labels_manager import observations_labels_manager
from mm_wrapper.history_graph import history_graph, observation, action, label, history, history_pattern


class history_subset:

    def __init__(self, label_to_obs_act: Dict[label, Union[observation, action]] = {},
                 custom_functions: List[Callable[[
                     history, observation], List[action]]] = [],
                 history_graphs: List[history_graph] = [],
                 obs_hist_act_rules: Dict[Tuple[history, observation], List[action]] = {}) -> None:
        self.label_to_obs_act: Dict[label,
                                    Union[observation, action]] = label_to_obs_act
        self.custom_functions: List[Callable[[
            history, observation], List[action]]] = custom_functions
        self.history_graphs: List[history_graph] = history_graphs
        self.obs_hist_act_rules: Dict[Tuple[history,
                                            observation], List[action]] = obs_hist_act_rules

    def to_dict(self) -> Dict:

        def serialize_functions(functions):
            serialized = []
            for func in functions:
                serialized.append({
                    'module': func.__module__,
                    'name': func.__name__
                })
            return serialized

        return {"label_to_obs_act": self.label_to_obs_act, "custom_functions": serialize_functions(self.custom_functions), "history_graphs": [hg.to_dict() for hg in self.history_graphs], "obs_hist_act_rules": self.obs_hist_act_rules}

    @staticmethod
    def from_dict(obj: Dict) -> 'history_subset':

        def deserialize_functions(json_str):
            serialized = json_str
            functions = []
            for item in serialized:
                module = importlib.import_module(item['module'])
                func = getattr(module, item['name'])
                functions.append(func)
            return functions

        return history_subset(obj["label_to_obs_act"], deserialize_functions(obj["custom_functions"]), obj["history_graphs"], obj["obs_hist_act_rules"])

    def add_pattern(self, pattern: history_pattern, label_to_obs_act: Dict[label, Union[observation, action]] = {}):
        self.label_to_obs_act.update(label_to_obs_act)
        self.history_graphs += [
            self._convert_pattern_to_history_graph(pattern)]

    def add_rules(self, rules: Dict[Tuple[history, observation], List[action]], label_to_obs_act: Dict[label, Union[observation, action]] = {}):
        self.label_to_obs_act.update(label_to_obs_act)
        self.obs_hist_act_rules.update(rules)

    def add_history(self, history: history, label_to_obs_act: Dict[label, Union[observation, action]] = {}):
        self.label_to_obs_act.update(label_to_obs_act)
        self.history_graphs += [
            self._convert_history_to_history_graph(history)]

    def add_custom_function(self, custom_function: Callable[[history, observation], List[action]], label_to_obs_act: Dict[label, Union[observation, action]] = {}):
        self.label_to_obs_act.update(label_to_obs_act)
        self.custom_functions += [custom_function]

    def _convert_pattern_to_history_graph(self, pattern: history_pattern) -> history_graph:
        hs = history_graph()
        hs.add_pattern(pattern)
        return hs

    # def _convert_rules_to_history_graph(self, rules: Dict[Tuple[history, observation], List[action]]) -> history_graph:
    #     hs = history_graph()
    #     hs.add
    #     return

    def next_actions(self, history: history, observation: observation) -> List[action]:
        possible_actions = self.obs_hist_act_rules.get(
            (history, observation), None)
        for history_graph in self.history_graphs:
            if history_graph.next_actions(history, observation) is not None:
                if possible_actions == None:
                    possible_actions = history_graph.next_actions(
                        history, observation)
                else:
                    possible_actions = [
                        value for value in possible_actions if value in history_graph.next_actions(history, observation)]
        for custom_function in self.custom_functions:
            if custom_function(history, observation) is not None:
                if possible_actions == None:
                    possible_actions = custom_function(history, observation)
                else:
                    possible_actions = [
                        value for value in possible_actions if value in custom_function(history, observation)]
        return possible_actions


class history_subset_factory:

    def __init__(self) -> None:
        pass

    def new(self) -> 'history_subset_factory':
        self.hs = history_subset()
        return self

    def add_pattern(self, pattern: history_pattern, observation_manager: observations_labels_manager = None) -> 'history_subset_factory':
        self.hs.add_pattern(pattern)
        return self

    def add_rules(self, rules: Dict[Tuple[history, observation], List[action]], label_to_obs_act: Dict[label, Union[observation, action]] = {}) -> 'history_subset_factory':
        self.hs.add_rules(rules)
        return self

    def add_history(self, history: history, label_to_obs_act: Dict[label, Union[observation, action]] = {}) -> 'history_subset_factory':
        self.hs.add_history(history)
        return self

    def add_custom_function(self, custom_function: Callable[[history, observation], List[action]], label_to_obs_act: Dict[label, Union[observation, action]] = {}) -> 'history_subset_factory':
        self.hs.add_custom_function(custom_function)
        return self

    def set_observations_labels_manager(self, observations_labels_manager: observations_labels_manager) -> 'history_subset_factory':
        self.observations_labels_manager = observations_labels_manager
        return self

    def create(self) -> history_subset:
        return self.hs


hs_factory = history_subset_factory()

if __name__ == '__main__':

    def dummy_function(h, obs):
        return "act1" if obs == "obs1" else "act0"

    hs = hs_factory.new().add_custom_function(dummy_function, {
        "act1": 0, "obs1": [0, 0, 1], "act0": 0}).create()
    # print(hs.next_actions(None, "obs1"))

    hs = hs_factory.new().add_rules(
        {(None, "obs1"): ["act1", "act2"]}, {}).create()
    # print(hs.next_actions(None, "obs1"))

    print(hs.from_dict(hs.to_dict()).custom_functions[0]({}, "oszbs1"))
