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
        rule_counter = defaultdict(lambda: {"count": 0, "obs_sum": None})

        for traj in trajectories:
            for obs, act in traj:
                key = int(act)
                if rule_counter[key]["obs_sum"] is None:
                    rule_counter[key]["obs_sum"] = np.array(obs)
                else:
                    rule_counter[key]["obs_sum"] += obs
                rule_counter[key]["count"] += 1

        total = sum([v["count"] for v in rule_counter.values()])
        rules = []

        for act, data in rule_counter.items():
            avg_obs = data["obs_sum"] / data["count"]
            weight = data["count"] / total if total > 0 else 0.0
            rules.append({"observation": avg_obs.tolist(),
                         "action": act, "weight": weight})

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
