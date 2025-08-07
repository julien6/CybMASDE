# io_utils.py

import os
import json
from typing import List, Dict, Tuple, Any
import numpy as np


def load_trajectories(analysis_results_path: str) -> List[Dict[str, List[Tuple[np.ndarray, int]]]]:
    """
    Load all trajectory files from the trajectory folder located inside the analysis results path.
    """
    trajectories = []
    traj_dir = os.path.join(analysis_results_path, "trajectories")
    if not os.path.exists(traj_dir):
        raise FileNotFoundError(
            f"Trajectory directory not found at {traj_dir}")

    for file in sorted(os.listdir(traj_dir)):
        if file.startswith("trajectories_") and file.endswith(".json"):
            with open(os.path.join(traj_dir, file), 'r') as f:
                data = json.load(f)
                # Convert observations to numpy arrays for consistency
                for agent in data:
                    data[agent] = [(np.array(obs), act)
                                   for obs, act in data[agent]]
                trajectories.append(data)
    return trajectories


def export_to_json(data: Any, file_path: str):
    """
    Save a dictionary or serializable object to a JSON file.
    Automatically creates parent directories if they don't exist.
    Also converts numpy types (including keys) to standard JSON types.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def sanitize(obj):
        if isinstance(obj, dict):
            return {k.item() if isinstance(k, (np.integer, np.floating)) else k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(sanitize(i) for i in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    sanitized_data = sanitize(data)

    with open(file_path, 'w') as f:
        json.dump(sanitized_data, f, indent=4)
