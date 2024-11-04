import math
from typing import List

from mm_wrapper.observable_policy_constraint import observable_policy_constraint
from mm_wrapper.observable_reward_function import observable_reward_function
from mm_wrapper.utils import label, history


normal_leader_adversary_sizes = {
    'self_vel': 2,
    'self_pos': 2,
    'landmark_rel_positions': 10,
    'other_agent_rel_positions': 10,
    'other_agent_velocities': 4,
    'self_in_forest': 2,
    'leader_comm': 4
}

good_sizes = {
    'self_vel': 2,
    'self_pos': 2,
    'landmark_rel_positions': 10,
    'other_agent_rel_positions': 10,
    'other_agent_velocities': 2,
    'self_in_forest': 2
}


def extract_values(concat_array, sizes):
    extracted_values = {}
    index = 0
    for key, size in sizes.items():
        extracted_values[key] = concat_array[index:index+size]
        index += size
    return extracted_values


def dist(pos1, pos2) -> float:
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)


def leader_adversary_opc_fun(history: history, observation: label, agent_name=None) -> List[label]:
    # print("Leader adversary")
    data = extract_values(observation, normal_leader_adversary_sizes)
    other_agent_rel_positions = data["other_agent_rel_positions"]
    # self_pos = (data["self_pos"][0], data["self_pos"][1])
    # other_agent_rel_positions = {agent: (other_agent_rel_positions[i*2] + self_pos[0], other_agent_rel_positions[i*2+1] + self_pos[1]) for i, agent in enumerate(
    #     ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0', 'agent_1'])}
    other_agent_rel_positions = {agent: (other_agent_rel_positions[i*2], other_agent_rel_positions[i*2+1]) for i, agent in enumerate(
        ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0', 'agent_1'])}
    # print(other_agent_rel_positions)

    for adversary in ["adversary_0", "adversary_1", "adversary_2"]:
        min_dist = 10000
        min_agent = None
        for good_agent in ["agent_0", "agent_1"]:
            # d = dist(
            #     other_agent_rel_positions[adversary], other_agent_rel_positions[good_agent])
            d = math.sqrt(
                (other_agent_rel_positions[good_agent][0])**2 + (other_agent_rel_positions[good_agent][1])**2)
            if d < min_dist:
                min_dist = d
                min_agent = good_agent
        # target_agent_vec = np.array(other_agent_rel_positions[min_agent]) - np.array(self_pos)
        target_agent_vec = other_agent_rel_positions[min_agent]
        if abs(target_agent_vec[0]) / abs(target_agent_vec[1]) > 1:
            if target_agent_vec[0] > 0:
                return [2]
            return [1]
        else:
            if target_agent_vec[1] > 0:
                return [4]
            return [3]
    return [0]


leader_opc = observable_policy_constraint()
leader_opc.add_custom_function(leader_adversary_opc_fun)


def normal_adversary_opc_fun(history: history, observation: label, agent_name=None) -> List[label]:
    # print("Normal adversary")
    agent_names = ['leadadversary_0', 'adversary_0',
                   'adversary_1', 'adversary_2', 'agent_0', 'agent_1']
    agent_names.remove(agent_name)
    data = extract_values(observation, normal_leader_adversary_sizes)
    other_agent_rel_positions = data["other_agent_rel_positions"]
    other_agent_rel_positions = {agent: (other_agent_rel_positions[i*2], other_agent_rel_positions[i*2+1]) for i, agent in enumerate(
        agent_names)}

    min_dist = 10000
    min_agent = None
    for good_agent in ["agent_0", "agent_1"]:
        d = math.sqrt(
            (other_agent_rel_positions[good_agent][0])**2 + (other_agent_rel_positions[good_agent][1])**2)
        if d < min_dist:
            min_dist = d
            min_agent = good_agent
    target_agent_vec = other_agent_rel_positions[min_agent]
    if abs(target_agent_vec[1]) == 0:
        return [0]
    if float(abs(target_agent_vec[0]) / abs(target_agent_vec[1])) >= 1.:
        if target_agent_vec[0] > 0:
            return [2]
        return [1]
    else:
        if target_agent_vec[1] > 0:
            return [4]
        return [3]
    return [0]


normal_opc = observable_policy_constraint()
normal_opc.add_custom_function(normal_adversary_opc_fun)


def good_opc_fun(history: history, observation: label, agent_name=None) -> List[label]:
    # print("Good agents")
    # pprint(extract_values(observation, good_sizes))
    # print()
    return None


good_opc = observable_policy_constraint()
good_opc.add_custom_function(good_opc_fun)
