# mma_wrapper/temm/visualization.py

import os
from mma_wrapper.temm.clustering import cluster_full_trajectories, cluster_trajectories_from_action, cluster_trajectories_from_observation
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform


def plot_dendrogram(data: List[np.ndarray], title: str, save_path: str):
    """Generic function to plot and save a dendrogram from a list of vectors."""
    plt.figure(figsize=(10, 5))
    dendrogram(data)
    plt.title(title)
    plt.xlabel("Trajectory")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_pca(data: List[np.ndarray], title: str, save_path: str):
    """Generic function to compute and plot PCA projection from a list of vectors."""
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(np.array(data))
    plt.figure(figsize=(8, 6))
    for i, point in enumerate(reduced):
        plt.scatter(point[0], point[1], label=str(i))
        plt.text(point[0], point[1], str(i), fontsize=8)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


# Wrappers for specific use cases

def generate_actions_dendrogram(action_trajectories: List[np.ndarray], save_path="actions_dendrogram.png"):
    clusters, linkage_matrix = cluster_trajectories_from_action(
        action_trajectories, "cosine")
    return plot_dendrogram(linkage_matrix, "Dendrogram - Action Trajectories", save_path)


def generate_observations_dendrogram(observation_trajectories: List[np.ndarray], save_path="observations_dendrogram.png"):
    clusters, linkage_matrix = cluster_trajectories_from_observation(
        observation_trajectories, "cosine")
    return plot_dendrogram(linkage_matrix, "Dendrogram - Observation Trajectories", save_path)


def generate_dendrogram(full_trajectories: List[np.ndarray], save_path="full_dendrogram.png"):
    clusters, linkage_matrix = cluster_full_trajectories(
        full_trajectories, "cosine")
    return plot_dendrogram(linkage_matrix, "Dendrogram - Full Trajectories", save_path)


def visualize_action_pca(data: List[np.ndarray], save_path="action_pca.png"):
    data = np.array(data)
    if data[0][0].shape == ():
        nb_actions = max([a for seq in data for a in seq]) + 1
        data = np.array([np.eye(nb_actions)[a] for seq in data for a in seq])
    else:
        data = np.array([[a] for seq in data for a in seq])
    return plot_pca(data, "PCA - Actions (1 point = 1 action)", save_path)


def visualize_observation_pca(data: List[np.ndarray], save_path="observation_pca.png"):
    data = [o for seq in data for o in seq]
    return plot_pca(data, "PCA - Observations (1 point = 1 observation)", save_path)


def visualize_transition_pca(data: List[np.ndarray], save_path="transition_pca.png"):
    nb_actions = max([a for seq in data for _, a in seq]) + 1
    data = [np.concatenate((o, np.eye(nb_actions)[a])) for seq in data for o, a in seq]
    return plot_pca(data, "PCA - Transitions (1 point = 1 transition)", save_path)


def visualize_action_trajectory(data: List[np.ndarray], save_path="action_trajectory_pca.png"):
    nb_actions = max([a for seq in data for a in seq]) + 1
    data = [np.concatenate([np.eye(nb_actions)[a] for a in seq]) for seq in data]
    return plot_pca(data, "PCA - Action Trajectories (1 point = 1 trajectory)", save_path)


def visualize_observation_trajectory(data: List[np.ndarray], save_path="observation_trajectory_pca.png"):
    data = [np.concatenate(seq) for seq in data]
    return plot_pca(data, "PCA - Observation Trajectories (1 point = 1 trajectory)", save_path)


def visualize_trajectory(data: List[np.ndarray], save_path="full_trajectory_pca.png"):
    nb_actions = max([a for seq in data for o, a in seq]) + 1
    data = [np.concatenate([np.concatenate((o, np.eye(nb_actions)[a])) for o, a in seq]) for seq in data]
    return plot_pca(data, "PCA - Full Trajectories (1 point = 1 trajectory)", save_path)
