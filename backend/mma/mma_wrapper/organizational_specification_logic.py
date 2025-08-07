import copy
import inspect
import random
import mma_wrapper

from typing import Any, Callable, Dict, List, Tuple
import mma_wrapper.label_manager
from mma_wrapper.utils import observation, action, label, pattern_trajectory, trajectory, trajectory_pattern_str
from mma_wrapper.trajectory_pattern import trajectory_pattern
from mma_wrapper.label_manager import label_manager
from mma_wrapper.utils import load_function, dump_function


class os_logic:

    def to_dict(self):
        raise NotImplementedError

    @staticmethod
    def from_dict(d: Any) -> 'os_logic':
        raise NotImplementedError

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __hash__(self):
        return hash(self.__dict__())

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class role_logic(os_logic):
    """ Implements both Role Action Guide (RAG) and Role Reward Guide (RRG) as a common way
    to restrict/guide agents to expected behavior as a role.

        **rag : trajectory x observation -> powerset(action x float)**

            A RAG outputs a list of authorized (action, weight) couples from a trajectory and an
            observation to restrict next possible agent's actions.

        **rrg = trajectory x observation x action -> float**

            A RRG outputs a reward indicating how far an agent is from expected behavior
            according to previous trajectory and the received observation and the made action.

        *Note: When RRG not provided, RRG is derived from RAG by default considering a default
        bonus/malus reward when the chosen actions in trajectory do not comply with authorized ones*
    """

    def __init__(self,
                 pattern_rules: Dict[Tuple[trajectory_pattern_str,
                                           label], List[Tuple[label, float]]] = None,
                 function_rules: List[Callable[[
                     trajectory, observation, str, 'label_manager'], List[Tuple[action, float]]]] = None,
                 label_manager: label_manager = None,
                 reward_reshape_standard: float = 100):
        manager = label_manager
        self.pattern_rules: Dict[Tuple[trajectory_pattern_str,
                                       label], List[Tuple[label, float]]] = {} if pattern_rules is None else pattern_rules
        self.script_rules: List[Callable[[
            trajectory, observation, str, 'label_manager'], List[Tuple[action, float]]]] = [] if function_rules is None else function_rules
        self.label_manager: 'label_manager' = manager if manager is not None else mma_wrapper.label_manager.label_manager()
        if inspect.isclass(self.label_manager):
            self.label_manager = self.label_manager()
        self.reward_reshape_standard = reward_reshape_standard

    def register_pattern_rule(self, trajectory_pattern_str: trajectory_pattern_str, observation_label: label, weighted_actions: List[Tuple[label, float]]):
        """
        Register a pattern-based trajectory (TP) rule to the current TP rule register.
        A TP rule is triggered when an agent's trajectory and received observation do match its 'trajectory_pattern_str' and 'observation' resulting in a list of weighted actions.
        Args:
            trajectory_pattern_str: the TP to register into forming the first condition of the TP rule
            observation_label: the observation label to register into forming the second condition of the TP rule
            weighted_actions: the weighted actions to be outputed when the two conditions are satisfied
        Returns:
            None
        """
        if trajectory_pattern_str is None:
            trajectory_pattern_str = "any"
        if observation_label is None:
            observation_label = "any"
        if (trajectory_pattern_str, observation_label) not in self.pattern_rules:
            self.pattern_rules[(trajectory_pattern_str,
                                observation_label)] = []
        self.pattern_rules[(trajectory_pattern_str, observation_label)
                           ] += weighted_actions
        return self

    def register_pattern_rules(self, rules: List[Tuple[Tuple[trajectory_pattern_str, label], List[Tuple[label, float]]]]):
        """
        Register a list of TP rules to the current TP rule register
        Args:
            rules: The TP rules to be registered as a list of ((trajectory pattern, observation label), weighted actions) couples
        Return
            None
        """
        for (trajectory_pattern_str, observation), weighted_actions in rules:
            self.register_pattern_rule(
                trajectory_pattern_str, observation, weighted_actions)
        return self

    def registrer_script_rule(self, function: Callable[[trajectory, observation, str, label_manager], List[Tuple[action, float]]], save_source: bool = False):
        """
        Register a script-based rule.
        A script-based rule is a function that outputs weighted actions from a raw trajectory and a raw observation.

        Args:
            function: the callable function to be added in the register

        Returns:
            None
        """
        self.script_rules.append((function, save_source, None, None))
        return self

    def registrer_script_rules(self, functions: List[Callable[[trajectory, observation, str, label_manager], List[Tuple[action, float]]]]):
        """
        Register a list of script-based rules

        Args:
            functions: the list of script-based rules to be added in the register

        Returns:
            None
        """
        for function in functions:
            self.register_pattern_rule(function)
        return self

    def next_weighted_actions(self, trajectory: trajectory, observation: observation, agent_name: str, merge_mode='additive') -> List[Tuple[label, float]]:

        weighted_actions = []

        if len(self.pattern_rules) > 0:
            # dealing with pattern-based trajectory rules
            patternified_trajectory_str = ",".join([f'{_obs},{_act}' for _obs, _act in self.label_manager.label_trajectory(
                trajectory=trajectory)])
            observation_label = self.label_manager.label_observation(observation)
            for _pattern_trajectory_str, _observation_label in self.pattern_rules.keys():
                if trajectory_pattern(_pattern_trajectory_str).match(patternified_trajectory_str) and trajectory_pattern(_observation_label).match(observation_label):
                    if merge_mode == 'substractive':
                        weighted_actions = list(set(weighted_actions) & set([(self.label_manager.unlabel_action(act_label)[0], weight) for act_label, weight in self.pattern_rules[(_pattern_trajectory_str, _observation_label)]]))
                    else:
                        weighted_actions.extend([(self.label_manager.unlabel_action(act_label)[0], weight) for act_label, weight in
                            self.pattern_rules[(_pattern_trajectory_str, _observation_label)]])

        # dealing with script-based trajectory rules
        for _script_rule in self.script_rules:
            if merge_mode == 'substractive':
                weighted_actions = list(set(weighted_actions) & set(
                    _script_rule[0](trajectory, observation, agent_name, self.label_manager)))
            else:
                next_weighted_actions = _script_rule[0](trajectory, observation, agent_name, self.label_manager)
                if isinstance(next_weighted_actions, int):
                    next_weighted_actions = [(next_weighted_actions, 1)]
                weighted_actions.extend(next_weighted_actions)

        weighted_actions = [(item, 1.) if len(
            item) == 1 else item for item in weighted_actions]

        return weighted_actions

    def next_reward(self, trajectory: trajectory, observation: observation, action: action, agent_name: str) -> float:
        """
        Compute the next reward bonus/malus as the product of the 'relative_reward_reshape' (or 'rrr' strictly comprised between -1 and 1) and the 'reward_reshape_standard' (or 'rss' as a positive real number).

        This bonus/malus is added to the "vanilla" reward once an agent has applied an action (after 'env.step' but before 'return obs, rew, done, info') to guide it into adopting the role expected behavior.

        This reward bonus/malus is computed from soft constraint solely (i.e constraint hardness/weight is strictly between -1 et 1):
            - If in [0;1[  -> the associated action should be selected with a weighted priority
            - If in ]-1;0] -> the associated action should not be selected with a weighted priority

        The 'rrr' mechanism can be overviewed as such:
            - If there are no weighted actions that would have been expected to be made / not be made for the current observation and trajectory, then the agent is free to choose any action since the 'rrr' is 0.
            - Else if weighted actions would have been expected to be made / not be made for the current observation and trajectory, then
                - If the agent has chosen an action outside of any of these unwanted/wanted actions
                    - If some of actions are wanted to be chosen (at least a weight is strictly positive), then the 'rrr' is the least bad penalty (strictly negative weights) if there are else -1: we want to penalize the agent for not having chosen a wanted action but not as much as unwanted actions
                - If the agent has chosen action within these unwanted/wanted actions, the the 'rrr' is directly equals to the weight: we want to penalize/reward the agent from a spectrum spanning from -1 to 1

        Args:
            trajectory: an agent's trajectory (containing raw observation and action objects)
            observation: an agent's received raw observation
            action: an agent's chosen action
            agent_name: an agent's name (to be used conveniently in script)

        Returns:
            float the bonus/malus to add in reward to entice the agent's into adopting the role's expected behavior
        """
        expected_weighted_actions = self.next_weighted_actions(
            trajectory, observation, agent_name)
        softly_expected_weighted_actions = [
            (action, weight) for action, weight in expected_weighted_actions if -1 < weight and weight < 1]
        if len(softly_expected_weighted_actions) > 0:
            softly_expected_actions, soft_weights = zip(
                *softly_expected_weighted_actions)
            if any([soft_weight > 0 for soft_weight in soft_weights]) and action not in softly_expected_actions:
                least_bad_penalty = [soft_weight for soft_weight in list(
                    soft_weights) if soft_weight <= 0]
                if len(least_bad_penalty) == 0:
                    least_bad_penalty = -1.
                else:
                    least_bad_penalty = max(least_bad_penalty)
                return least_bad_penalty * self.reward_reshape_standard
            return soft_weights[softly_expected_actions.index(action)] * self.reward_reshape_standard
        return 0

    def next_action(self, trajectory: trajectory, observation: observation, agent_name: str) -> action:
        """
        Randomly selects the next action complying with hard constraint in the TP and script rules (i.e. the associated actions' weights are greater or equal to 0).

        This function is solely used to remplace the initial chosen action by another expected one (according to the hard constraints) during the training and evaluation steps.
        If there are no associated hard constraints, the next action to be chosen is 'None', meaning the initial chosen action is kept for the rest of the step

        Args:
            trajectory: the agent's raw trajectory
            observation: the agent's raw observation
            agent_name: the agent's name

        Returns:
            action: the chosen action
        """
        expected_weighted_actions = self.next_weighted_actions(
            trajectory, observation, agent_name)
        strongly_expected_weighted_actions = [
            (action, weight) for action, weight in expected_weighted_actions if weight >= 1]
        if len(strongly_expected_weighted_actions) > 0:
            strongly_expected_actions, strong_weights = zip(
                *strongly_expected_weighted_actions)
            next_action = random.choices(
                strongly_expected_actions, weights=strong_weights, k=1)[0]
            return next_action
        return None

    def to_dict(self) -> Dict:
        return {
            "label_manager": self.label_manager.to_dict(),
            "pattern_rules": {f'({tp};{obs})': _weighted_actions for (tp, obs), _weighted_actions in self.pattern_rules.items()},
            "reward_reshape_standard": self.reward_reshape_standard,
            "script_rules": [dump_function(_function, _save_source, _function_name, _function_source) for _function, _save_source, _function_name, _function_source in self.script_rules]
        }

    @staticmethod
    def from_dict(d) -> 'role_logic':
        return role_logic(pattern_rules={tuple(tp_obs[1:-1].split(";")): weighted_actions for tp_obs, weighted_actions in d["pattern_rules"].items()},
                          function_rules=[load_function(_function_data)
                                          for _function_data in d["script_rules"]],
                          label_manager=label_manager.from_dict(
                              d["label_manager"]),
                          reward_reshape_standard=d.get("reward_reshape_standard", 100))


class goal_logic(os_logic):
    """ Implements Goal Reward Guide (GRG) a way to guide agents to find a way to reach a goal.

        **grg: trajectory -> float**

            A GRG outputs a reward indicating how far an agent is from expected behavior
            according to previous trajectory characterizing a goal.
    """

    def __init__(self,
                 pattern_rules: Dict[trajectory_pattern_str, float] = None,
                 function_rules: List[Callable[[trajectory, str, label_manager], float]] = None,
                 label_manager: label_manager = None,
                 reward_reshape_standard: float = 100):
        manager = label_manager
        self.pattern_rules: Dict[trajectory_pattern_str, float] = {
        } if pattern_rules is None else pattern_rules
        self.script_rules: List[Callable[[trajectory, str, 'label_manager'], float]] = [
        ] if function_rules is None else function_rules
        self.label_manager: 'label_manager' = manager if manager is not None else mma_wrapper.label_manager.label_manager()
        if inspect.isclass(self.label_manager):
            self.label_manager = self.label_manager()
        self.reward_reshape_standard = reward_reshape_standard

    def register_pattern_rule(self, trajectory_pattern_str: trajectory_pattern_str, relative_reward_reshape: float):
        """
        Register a pattern-based trajectory (TP) rule to the current TP rule register.
        A TP rule is triggered when an agent's trajectory do match its 'trajectory_pattern_str' resulting in a float reward.
        Args:
            trajectory_pattern_str: the TP to register into forming the first condition of the TP rule
            relative_reward_reshape: the relative reward reshape value (supposedly between -1 and 1)
        Returns:
            None
        """
        if trajectory_pattern_str is None:
            trajectory_pattern_str = "any"
        self.pattern_rules[trajectory_pattern_str] = relative_reward_reshape
        return self

    def register_pattern_rules(self, rules: List[Tuple[trajectory_pattern_str, float]]):
        """
        Register a list of TP rules to the current TP rule register
        Args:
            rules: The TP rules to be registered as a list of (trajectory pattern, float) couples
        Return
            None
        """
        for trajectory_pattern_str, rrr in rules:
            self.register_pattern_rule(trajectory_pattern_str, rrr)
        return self

    def registrer_script_rule(self, function: Callable[[trajectory, str, label_manager], float], save_source: bool = False):
        """
        Register a script-based rule.
        A script-based rule is a function that outputs a relative reward reshape value (supposedely between -1 and 1) from a raw trajectory.

        Args:
            function: the callable function to be added in the register

        Returns:
            None
        """
        self.script_rules.append((function, save_source, None, None))
        return self

    def registrer_script_rules(self, functions: List[Callable[[trajectory, str, label_manager], float]]):
        """
        Register a list of script-based rules

        Args:
            functions: the list of script-based rules to be added in the register

        Returns:
            None
        """
        self.script_rules.extend(functions)
        return self

    def next_reward(self, trajectory: trajectory, agent_name: str) -> float:
        """
        Compute the next reward bonus/malus as the product of the 'relative_reward_reshape' (or 'rrr' strictly comprised between -1 and 1) and the 'reward_reshape_standard' (or 'rss' as a positive real number).

        This bonus/malus is added to the "vanilla" reward once an agent has applied an action (after 'env.step' but before 'return obs, rew, done, info') to guide it into finding its own way to reach a specific subsequence characterizing a goal.

        This reward bonus/malus is computed from the 'rrr' obtained from goal rules (TP and script rules)

        The 'rrr' mechanism can be overviewed as such:
            - If there are no goal rules that is triggered, then the agent receives a null 'rrr', hence a 0 bonus/malus.
            - Else if some goal rules are triggered, the final 'rrr' is the sum of all those applied ones' results

        Args:
            trajectory: an agent's trajectory (containing raw observation and action objects)

        Returns:
            float the bonus/malus to add in reward to entice the agent's into adopting the role's expected behavior
        """
        rrr = 0

        # dealing with pattern rules
        if len(self.pattern_rules) > 0:
            patternified_trajectory_str = ",".join(
                [f'{_obs},{_act}' for _obs, _act in self.label_manager.label_trajectory(trajectory)])
            for _pattern, _rrr in self.pattern_rules.items():
                if trajectory_pattern(_pattern).match(patternified_trajectory_str):
                    rrr += _rrr

        # dealing with script rules
        for _script_rule in self.script_rules:
            rrr += _script_rule[0](trajectory, agent_name, self.label_manager)

        return rrr

    def to_dict(self) -> Dict:
        return {
            "label_manager": self.label_manager.to_dict(),
            "pattern_rules": self.pattern_rules,
            "reward_reshape_standard": self.reward_reshape_standard,
            "script_rules": [dump_function(_function, _save_source, _function_name, _function_source) for _function, _save_source, _function_name, _function_source in self.script_rules]
        }

    @staticmethod
    def from_dict(d) -> 'role_logic':
        return goal_logic(pattern_rules=d["pattern_rules"],
                          function_rules=[load_function(_function_data)
                                          for _function_data in d["script_rules"]],
                          label_manager=label_manager.from_dict(
                              d["label_manager"]),
                          reward_reshape_standard=d.get("reward_reshape_standard", 100))


class role_factory:

    def __init__(self, pattern_rules: Dict[Tuple[trajectory_pattern_str,
                                                 label], List[Tuple[label, float]]] = None,
                 function_rules: List[Callable[[
                     trajectory, observation, str, 'label_manager'], List[Tuple[action, float]]]] = None,
                 label_manager: label_manager = None,
                 reward_reshape_standard: float = 100):
        manager = label_manager
        self.pattern_rules: Dict[Tuple[trajectory_pattern_str,
                                       label], List[Tuple[label, float]]] = {} if pattern_rules is None else pattern_rules
        self.script_rules: List[Callable[[
            trajectory, observation, str, "label_manager"], List[Tuple[action, float]]]] = [] if function_rules is None else function_rules
        self.label_manager: 'label_manager' = manager if manager is not None else mma_wrapper.label_manager.label_manager()
        self.reward_reshape_standard = reward_reshape_standard

    def create(self) -> role_logic:
        return copy.deepcopy(role_logic(self.pattern_rules, self.script_rules, self.label_manager, self.reward_reshape_standard))


class goal_factory:

    def __init__(self,
                 pattern_rules: Dict[trajectory_pattern_str, float] = None,
                 function_rules: List[Callable[[trajectory, str, label_manager], float]] = None,
                 label_manager: label_manager = None,
                 reward_reshape_standard: float = 100):
        manager = label_manager
        self.pattern_rules: Dict[trajectory_pattern_str, float] = {
        } if pattern_rules is None else pattern_rules
        self.script_rules: List[Callable[[trajectory, str, 'label_manager'], float]] = [
        ] if function_rules is None else function_rules
        self.label_manager: 'label_manager' = manager if manager is not None else mma_wrapper.label_manager.label_manager()
        self.reward_reshape_standard = reward_reshape_standard

    def create(self) -> goal_logic:
        return copy.deepcopy(goal_logic(self.pattern_rules, self.script_rules, self.label_manager, self.reward_reshape_standard))
