# mma_wrapper/temm/inference_roles.py

import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict


def extract_roles_from_trajectories(
    selected_trajectories: Dict[int, List[List[Tuple[np.ndarray, int]]]]
) -> Dict[int, Dict[str, Any]]:
    """
    Extract abstract roles for each cluster by generating observation → action rules.

    Args:
        selected_trajectories: Mapping from cluster id to list of (obs, act) trajectories.

    Returns:
        Dict[int, Dict]: Inferred roles with their rules, centroid, and support count.
    """
    inferred_roles = {}

    for cluster_id, trajectories in selected_trajectories.items():
        rule_counter = {}
        agents = set()
        rules = []

        for traj, agent_name in trajectories:
            agents.add(agent_name)
            for obs, act in traj:
                rule_counter.setdefault(str(obs), {}).setdefault(
                    str(act), {}).setdefault(agent_name, 0)
                rule_counter[str(obs)][str(act)][agent_name] += 1

        for traj, agent_name in trajectories:
            for obs, act in traj:
                total = sum([rule_counter[str(obs)][str(act)][agent]
                            for agent in agents])
                total = 1.0 if total == 0 else total
                rules.append({"observation": obs.tolist(),
                             "action": act, "weight": rule_counter[str(obs)][str(act)][agent_name] / total})

        inferred_roles[cluster_id] = {
            "rules": rules,
            "support": len(trajectories)
        }

    return inferred_roles


def summarize_roles(
    roles: Dict[int, Dict[str, Any]],
    min_rule_weight: float = 0.5
) -> Dict[int, Dict[str, Any]]:
    """
    Keep only the most important rules per role, based on their weight.

    Args:
        roles: Dictionary of inferred roles.
        min_rule_weight: Minimum threshold for a rule to be kept.

    Returns:
        Filtered and summarized roles dictionary.
    """
    summarized_roles = {}

    for cluster_id, role_data in roles.items():
        rules = role_data["rules"]
        filtered_rules = [r for r in rules if r["weight"] >= min_rule_weight]
        summarized_roles[cluster_id] = {
            "rules": filtered_rules,
            "support": role_data["support"]
        }

    return summarized_roles
