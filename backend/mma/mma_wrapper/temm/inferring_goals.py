# mma_wrapper/temm/inference_goals.py

import numpy as np
from typing import List, Dict, Any


def extract_goals_from_trajectories(
    selected_trajectories: Dict[int, List[List[np.ndarray]]], min_obs_weight: float = 0.
) -> Dict[int, Dict[str, Any]]:
    """
    Extract abstract goals from a set of selected observation trajectories near the centroid.

    Each goal is characterized by low-variance observations across trajectories.

    Args:
        selected_trajectories: Mapping from cluster_id to list of observation trajectories.
        min_obs_weight: Minimum threshold for an observation to be considered a goal.

    Returns:
        Dictionary mapping each cluster_id to a list of goal observations with their weight.
    """
    goals = {}

    for cluster_id, trajectories in selected_trajectories.items():
        observations_counter = {}
        agents_owner = {}
        observations = {}
        total = 0
        total_valid = 0

        # Flatten all observations from all trajectories
        for traj, agent_name in trajectories:
            for obs in traj:
                observations_counter.setdefault(
                    str(obs.tolist()), {}).setdefault(agent_name, 0)
                observations_counter[str(obs.tolist())][agent_name] += 1
                total += 1

        for obs, data in observations_counter.items():
            total_obs = sum(data.values())
            weight = float(total_obs / (total if total != 0 else 1))
            if weight > min_obs_weight:
                observations.setdefault(obs, {}).setdefault("agents_owner", sorted([(agent_name, count) for agent_name, count in observations_counter[obs]
                                                                                    .items()], key=lambda x: x[1], reverse=True))
                observations[obs]["weight"] = weight

                agents_owner = {agent_name: agents_owner.get(
                    agent_name, 0) + count for agent_name, count in observations[obs]["agents_owner"]}
                total_valid += 1

        goals[cluster_id] = {
            "observations": observations,
            "agents_owner": {agent_name: count / sum(agents_owner.values()) for agent_name, count in agents_owner.items()},
            "support": total_valid
        }

    return goals
