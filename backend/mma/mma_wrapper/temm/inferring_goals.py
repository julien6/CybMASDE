# mma_wrapper/temm/inference_goals.py

import numpy as np
from typing import List, Dict, Any


def extract_goals_from_trajectories(
    selected_trajectories: Dict[int, List[List[np.ndarray]]]
) -> Dict[int, Dict[str, Any]]:
    """
    Extract abstract goals from a set of selected observation trajectories near the centroid.

    Each goal is characterized by low-variance observations across trajectories.

    Args:
        selected_trajectories: Mapping from cluster_id to list of observation trajectories.

    Returns:
        Dictionary mapping each cluster_id to a list of goal observations with their weight.
    """
    goals = {}

    for cluster_id, trajectories in selected_trajectories.items():
        all_obs = []

        # Flatten all observations from all trajectories
        for traj in trajectories:
            all_obs.extend(traj)

        if not all_obs:
            continue

        all_obs_array = np.array(all_obs)  # shape: (num_samples, obs_dim)
        variances = np.var(all_obs_array, axis=0)
        avg_variance = np.mean(variances)

        # Compute weight for each observation = inverse of distance to mean / variance
        centroid = np.mean(all_obs_array, axis=0)
        distances = np.linalg.norm(all_obs_array - centroid, axis=1)
        weights = 1 / (1 + distances)  # Avoid division by zero

        weighted_obs = []
        for obs, w in zip(all_obs_array, weights):
            weighted_obs.append(
                {"observation": obs.tolist(), "weight": float(w)})

        goals[cluster_id] = {
            "observations": weighted_obs,
            "avg_variance": float(avg_variance),
            "support": len(all_obs_array)
        }

    return goals


def summarize_goals(
    goals: Dict[int, Dict[str, Any]],
    min_goal_weight: float = 0.5
) -> Dict[int, Dict[str, Any]]:
    """
    Summarize inferred goals by filtering observations with weights above a threshold.

    Args:
        goals: Dictionary of goals extracted from observation trajectories.
        min_goal_weight: Threshold above which goal observations are kept.

    Returns:
        Filtered dictionary of goals with representative observations.
    """
    summarized = {}

    for cluster_id, goal_data in goals.items():
        filtered = [g for g in goal_data["observations"]
                    if g["weight"] >= min_goal_weight]
        summarized[cluster_id] = {
            "observations": filtered,
            "avg_variance": goal_data["avg_variance"],
            "support": goal_data["support"]
        }

    return summarized
