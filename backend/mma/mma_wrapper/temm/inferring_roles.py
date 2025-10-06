# mma_wrapper/temm/inference_roles.py

import numpy as np
import hashlib
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from pprint import pprint


def extract_roles_from_trajectories(
    selected_trajectories: Dict[int, List[List[Tuple[np.ndarray, int]]]], min_rule_weight: float = 0.5
) -> Dict[int, Dict[str, Any]]:
    """
    Extract abstract roles for each cluster by generating observation → action rules.

    Args:
        selected_trajectories: Mapping from cluster id to list of (obs, act) trajectories.
        min_rule_weight: Minimum threshold for a rule to be kept.

    Returns:
        Dict[int, Dict]: Inferred roles with their rules, centroid, and support count.
    """
    inferred_roles = {}

    for cluster_id, trajectories in selected_trajectories.items():
        transition_counter = {}
        agents = set()
        rules = {}
        agents_owner = {}
        total = 0
        total_valid = 0

        for traj, agent_name in trajectories:
            agents.add(agent_name)
            for obs, act in traj:
                transition_counter.setdefault(
                    str(obs.tolist()), {}).setdefault(str(act), {}).setdefault(agent_name, 0)
                transition_counter[str(obs.tolist())
                                   ][str(act)][agent_name] += 1
                total += 1

        for obs, data in transition_counter.items():

            for act, _ in data.items():

                weight = sum(transition_counter[obs][act].values()) / total
                if weight > min_rule_weight:

                    rules.setdefault(obs, {}).setdefault(act, {}).setdefault(
                        "agents_owner", [])
                    rules[obs][act]["agents_owner"] = sorted(
                        [(agent, transition_counter[obs][act][agent]) for agent in agents], key=lambda x: x[1], reverse=True)

                    rules[obs][act]["weight"] = weight

                    agents_owner = {agent_name: agents_owner.get(
                        agent_name, 0) + count for agent_name, count in rules[obs][act]["agents_owner"]}

                    total_valid += 1

        inferred_roles[cluster_id] = {
            "rules": rules,
            "agents_owner": {agent: weight / sum(agents_owner.values()) for agent, weight in agents_owner.items()},
            "support": total_valid
        }

    return inferred_roles
