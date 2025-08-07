import json
import gym

from typing import Any, List
from mma_wrapper.utils import observation, action, label, trajectory
from mma_wrapper.label_manager import label_manager
from mma_wrapper.organizational_specification_logic import role_logic, goal_logic, goal_factory, role_factory
from mma_wrapper.utils import cardinality
from mma_wrapper.organizational_model import organizational_model, link, link_type, group_specifications, structural_specifications, plan, plan_operator, sub_plan, mission_preference, social_scheme, functional_specifications, deontic_specification, deontic_specifications, time_constraint_type
from mma_wrapper.organizational_model import organizational_model


class label_manager_ex(label_manager):

    def __init__(self, action_space: gym.Space = None, observation_space: gym.Space = None):
        super().__init__(action_space, observation_space)
        self.decode_observation_mapping = {
            0: "o0",
            1: "o1",
            2: "o2",
            3: "o3",
            4: "o4",
            5: "o5",
            6: "o6",
            7: "o7",
        }
        self.encode_observation_mapping = {
            v: k for k, v in self.decode_observation_mapping.items()}
        self.decode_action_mapping = {
            0: "a0",
            1: "a1",
            2: "a2",
            3: "a3",
            4: "a4",
            5: "a5",
            6: "a6",
            7: "a7",
        }
        self.encode_action_mapping = {
            v: k for k, v in self.decode_action_mapping.items()}

    def one_hot_encode_observation(self, observation: Any) -> 'observation':
        return self.encode_observation_mapping.get(observation, -1)

    def one_hot_decode_observation(self, observation: observation) -> Any:
        return self.decode_observation_mapping.get(observation, "o-1")

    def one_hot_encode_action(self, action: Any) -> action:
        return self.encode_action_mapping.get(action, -1)

    def one_hot_decode_action(self, action: action) -> Any:
        return self.decode_action_mapping.get(action, "a-1")

    def label_observation(self, observation: observation) -> label:
        return self.decode_observation_mapping.get(observation, "o-1")

    def unlabel_observation(self, observation: label) -> List[observation]:
        return [self.encode_observation_mapping.get(observation, -1)]

    def label_action(self, action: action) -> label:
        return self.decode_action_mapping.get(action, "o-1")

    def unlabel_action(self, action: label) -> List[action]:
        return [self.encode_action_mapping.get(action, -1)]


label_mngr_ex = label_manager_ex()
role_fac = role_factory(label_manager=label_mngr_ex)
goal_fac = goal_factory(label_manager=label_mngr_ex)

print("0 - Creating an organizational model\n\n")

##############################################
# Instantiate the structural specifications
##############################################

# --------------------------------------------
# Define all the roles
roles = ["role_1", "role_2", "role_3"]
# --------------------------------------------

# --------------------------------------------
# Define the roles inheritance relations
role_inheritance_relations = {
    "role_2": ["role_1"], "role_3": ["role_1"]}
# --------------------------------------------

# --------------------------------------------
# Define the groups

#  - Group 1
intra_links = [link("role_1", "role_2", link_type.ACQ),
               link("role_2", "role_3", link_type.ACQ)]
inter_links = [link("role_1", "role_3", link_type.ACQ)]
intra_compatibilities = []
inter_compatibilities = []
role_cardinalities = {
    'role_0': cardinality(1, 1),
    'role_1': cardinality(1, 1),
    'role_2': cardinality(1, 1),
}
sub_group_cardinalities = {}
group2 = group_specifications(["role_1", "role_2", "role_3"], {}, intra_links, inter_links,
                              intra_compatibilities, inter_compatibilities, role_cardinalities, sub_group_cardinalities)

#  - Group 2
intra_links = [link("role_1", "role_2", link_type.ACQ),
               link("role_2", "role_3", link_type.ACQ)]
inter_links = []
intra_compatibilities = []
inter_compatibilities = []
role_cardinalities = {
    'role_0': cardinality(1, 1),
    'role_1': cardinality(1, 1),
    'role_2': cardinality(1, 1)
}
sub_group_cardinalities = {
}
group1 = group_specifications(roles, {}, intra_links, inter_links,
                              intra_compatibilities, inter_compatibilities, role_cardinalities, sub_group_cardinalities)
# --------------------------------------------


def role_3_func(tp, ob, agt, lbl_mngr):
    return [(3, 1.1)]


def role_4_func(tp, ob, agt, lbl_mngr): return [(4., 1.7)]


def role_5_func(tp, obs, agt, lbl_mngr): return [(5, 1.2)]


structural_specs = structural_specifications(
    roles={
        'role_1': role_logic(label_manager=label_mngr_ex)
        .register_pattern_rule("o0,a0", "o1", [("a1", 1.)]),
        'role_2': role_logic(label_manager=label_mngr_ex)
        .registrer_script_rule(lambda tp, obs, agt, lbl_mngr: [(2., 1)], save_source=True),
        'role_3': role_logic(label_manager=label_mngr_ex)
        .registrer_script_rule(role_3_func, save_source=True),
        'role_4': role_logic(label_manager=label_mngr_ex)
        .registrer_script_rule(role_4_func, save_source=True),
        'role_5': role_fac.create().registrer_script_rule(role_5_func, save_source=True),
        'role_6': role_fac.create().register_pattern_rule("o5,a5", "o6", [("a6", 1)])},
    role_inheritance_relations=role_inheritance_relations,
    root_groups={"group1": group1}
)

##############################################
# Instantiate the functional specifications
##############################################

# --------------------------------------------
# Instantiate the social schemes


def func1(x: trajectory, agt: str, lbl_mngr: label_manager) -> float:
    return 1.


def func3(x, agt: str, lbl_mngr: label_manager): return (3.)


def func4(x, agt: str, lbl_mngr: label_manager): return (4.)


def func5(x, agt: str, lbl_mngr: label_manager): return (5.)


class goal6(goal_logic):
    def __init__(self, pattern_rules=None, function_rules=None, label_manager=None, reward_reshape_standard=100):
        super().__init__(pattern_rules, function_rules,
                         label_manager, reward_reshape_standard)

        def func6(tp: trajectory, agent_name: str, lbl_mngr: 'label_manager') -> float:
            # I can use the label manager and the agent name here...
            labeled_trajectory = lbl_mngr.label_trajectory(
                tp)
            return 6.

        self.registrer_script_rule(func6, save_source=True)


goals = {
    'goal_1': goal_logic(label_manager=label_mngr_ex)
    .registrer_script_rule(func1),
    'goal_2': goal_logic(label_manager=label_mngr_ex)
    .registrer_script_rule(lambda x, agt, lbl_mngr: 2, save_source=True),
    'goal_3': goal_logic(label_manager=label_mngr_ex)
    .registrer_script_rule(func3, save_source=True),
    'goal_4': goal_fac.create()
    .registrer_script_rule(func4, save_source=True),
    'goal_5': goal_fac.create()
    .registrer_script_rule(func5, save_source=True),
    'goal_6': goal6(label_manager=label_mngr_ex)}

goals_structure = plan(
    goal='goal_1',
    sub_plan=sub_plan(
        operator=plan_operator.SEQUENCE,
        sub_goals=[
            plan('goal_2'),
            plan('goal_3'),
            plan('goal_4',
                 sub_plan=sub_plan(
                     operator=plan_operator.PARALLEL,
                     sub_goals=[plan("goal5"), plan("goal6")]),
                 probability=0.8),
        ]
    )
)

social_schemes = {
    'scheme1': social_scheme(
        goals_structure=goals_structure,
        mission_to_goals={
            'mission11': ['goal_1', 'goal_2'],
            'mission12': ['goal_1', 'goal_3'],
        },
        mission_to_agent_cardinality={
            'mission11': cardinality(1, 1),
            'mission12': cardinality(1, 1),
        }
    ),
    'scheme2': social_scheme(
        goals_structure=goals_structure,
        mission_to_goals={
            'mission21': ['goal_1', 'goal_2'],
            'mission22': ['goal_1', 'goal_3'],
        },
        mission_to_agent_cardinality={
            'mission21': cardinality(1, 9),
            'mission22': cardinality(1, 1),
        }
    )
}
# --------------------------------------------

# --------------------------------------------
# Instantiate the mission preferences
mission_preferences = [
    mission_preference(
        prefer='mission1',
        over='mission2'
    )
]
# --------------------------------------------

functional_specs = functional_specifications(
    goals=goals,
    social_scheme=social_schemes,
    mission_preferences=mission_preferences
)

##############################################
# Instantiate the deontic specifications
##############################################

# --------------------------------------------
# Define the permissions
permissions = [
    deontic_specification(
        role='role_1',
        agents=["agent_1"],
        missions='mission1',
        time_constraint=time_constraint_type.ANY
    ),
    deontic_specification(
        role='role_3',
        agents=["agent_2"],
        missions='mission1',
        time_constraint=time_constraint_type.ANY
    )
]
# --------------------------------------------
# --------------------------------------------
# Define the obligations
obligations = [
    deontic_specification(
        role='role_2',
        agents=["agent_1"],
        missions='mission2',
        time_constraint=time_constraint_type.ANY
    )
]
# --------------------------------------------

deontic_specs = deontic_specifications(
    permissions=permissions,
    obligations=obligations
)

############################################
# Instantiate the organizational model
org_model = organizational_model(
    structural_specifications=structural_specs,
    functional_specifications=functional_specs,
    deontic_specifications=deontic_specs
)
############################################


def test_global():

    print("1 - Printing the organizational model")
    print(org_model)
    print("\n\n\n")

    print("2 - Saving the model as a JSON file")
    file = open("organizational_model1.json", "w+")
    json.dump(org_model.to_dict(), file, indent=4)
    file.close()
    print("\n\n\n")

    print("3 - Loading the saved JSON file")
    file1 = open("organizational_model1.json", "r")
    os1_json = json.load(file1)
    file1.close()

    os1 = organizational_model.from_dict(os1_json)
    print(os1)
    print("\n\n\n")

    print("4 - Checking the 'next_action()' function does work as expected with the loaded model")

    print("role_1: ", os1.structural_specifications.roles["role_1"].next_action([(0, 0)], 1, "ag"))
    print("role_2: ", os1.structural_specifications.roles["role_2"].next_action([(1, 1)], 2, "ag"))
    print("role_3: ", os1.structural_specifications.roles["role_3"].next_action([(2, 2)], 3, "ag"))
    print("role_4: ", os1.structural_specifications.roles["role_4"].next_action([(3, 3)], 4, "ag"))
    print("role_5: ", os1.structural_specifications.roles["role_5"].next_action([(4, 4)], 5, "ag"))
    print("role_6: ", os1.structural_specifications.roles["role_6"].next_action([(5, 5)], 6, "ag"))

    check = 1 == os1.structural_specifications.roles["role_1"].next_action([(0, 0)], 1, "ag")
    check &= 2 == os1.structural_specifications.roles["role_2"].next_action([(1, 1)], 2, "ag")
    check &= 3 == os1.structural_specifications.roles["role_3"].next_action([(2, 2)], 3, "ag")
    check &= 4 == os1.structural_specifications.roles["role_4"].next_action([(3, 3)], 4, "ag")
    check &= 5 == os1.structural_specifications.roles["role_5"].next_action([(4, 4)], 5, "ag")
    check &= 6 == os1.structural_specifications.roles["role_6"].next_action([(5, 5)], 6, "ag")
    print("OK" if check else "KO")
    print("\n\n\n")

    print("5 - Checking the 'next_reward()' function does work as expected with the loaded model")

    print("goal_1: ", os1.functional_specifications.goals["goal_1"].next_reward([(0, 0)], "ag"))
    print("goal_2: ", os1.functional_specifications.goals["goal_2"].next_reward([(1, 1)], "ag"))
    print("goal_3: ", os1.functional_specifications.goals["goal_3"].next_reward([(2, 2)], "ag"))
    print("goal_4: ", os1.functional_specifications.goals["goal_4"].next_reward([(3, 3)], "ag"))
    print("goal_5: ", os1.functional_specifications.goals["goal_5"].next_reward([(4, 4)], "ag"))
    print("goal_6: ", os1.functional_specifications.goals["goal_6"].next_reward([(5, 5)], "ag"))

    check = 1. == os1.functional_specifications.goals["goal_1"].next_reward([(0, 0)], "ag")
    check &= 2. == os1.functional_specifications.goals["goal_2"].next_reward([(1, 1)], "ag")
    check &= 3. == os1.functional_specifications.goals["goal_3"].next_reward([(2, 2)], "ag")
    check &= 4. == os1.functional_specifications.goals["goal_4"].next_reward([(3, 3)], "ag")
    check &= 5. == os1.functional_specifications.goals["goal_5"].next_reward([(4, 4)], "ag")
    check &= 6. == os1.functional_specifications.goals["goal_6"].next_reward([(5, 5)], "ag")
    print("OK" if check else "KO")


if __name__ == '__main__':

    test_global()
