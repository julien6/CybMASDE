# preprocessing.py

import numpy as np
from typing import List, Dict, Tuple, Any


def get_agents(raw_episodes: List[Dict[str, List[List[Any]]]]) -> List[str]:
    """
    Extract the list of unique agent names from raw episodes.
    Args:
        raw_episodes: List of dictionaries mapping agent names to their sequences of (obs, act) pairs.
    Returns:
        List of unique agent names.
    """
    agents = set()
    for episode in raw_episodes:
        agents.update(episode.keys())
    return list(agents)


def extract_full_trajectories(raw_episodes: List[Dict[str, List[List[Any]]]]) -> List[List[Tuple[np.ndarray, int]]]:
    """
    Transform raw episodes into full trajectories (observation, action) per agent.

    Args:
        raw_episodes: List of dictionaries mapping agent names to their sequences of (obs, act) pairs.

    Returns:
        A list of full trajectories, where each trajectory is a list of (observation, action) tuples.
    """
    full_trajectories = []
    for episode in raw_episodes:
        for agent, transitions in episode.items():
            full_trajectories.append([(np.array(obs), act)
                                     for obs, act in transitions])
    return full_trajectories


def extract_action_trajectories(raw_episodes: List[Dict[str, List[List[Any]]]]) -> Dict[str, List[int]]:
    """
    Extract sequences of actions per agent from raw episodes.

    Args:
        raw_episodes: List of dictionaries mapping agent names to their sequences of (obs, act) pairs.

    Returns:
        Dictionary mapping each agent name to a list of actions.
    """
    action_trajectories = []
    for episode in raw_episodes:
        for agent, transitions in episode.items():
            action_trajectories.append(
                [act for obs, act in transitions])
    return action_trajectories


def extract_observation_trajectories(raw_episodes: List[Dict[str, List[List[Any]]]]) -> Dict[str, List[np.ndarray]]:
    """
    Extract sequences of observations per agent from raw episodes.

    Args:
        raw_episodes: List of dictionaries mapping agent names to their sequences of (obs, act) pairs.

    Returns:
        Dictionary mapping each agent name to a list of observations (as numpy arrays).
    """
    observation_trajectories = []
    for episode in raw_episodes:
        for agent, transitions in episode.items():
            observation_trajectories.append(
                [np.array(obs) for obs, act in transitions])
    return observation_trajectories
