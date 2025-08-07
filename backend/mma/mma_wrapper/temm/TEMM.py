# TEMM.py

import os
from typing import List, Dict

from mma_wrapper.temm.io_utils import load_trajectories, export_to_json
from mma_wrapper.temm.preprocessing import extract_full_trajectories, extract_action_trajectories, extract_observation_trajectories
from mma_wrapper.temm.clustering import cluster_trajectories_from_action, cluster_trajectories_from_observation, cluster_full_trajectories
from mma_wrapper.temm.visualization import (
    generate_actions_dendrogram,
    generate_observations_dendrogram,
    generate_dendrogram,
    visualize_action_pca,
    visualize_observation_pca,
    visualize_transition_pca,
    visualize_action_trajectory,
    visualize_observation_trajectory,
    visualize_trajectory
)
from mma_wrapper.temm.centroid import compute_action_centroids_per_cluster, compute_observation_centroids_per_cluster, compute_full_centroids_per_cluster
from mma_wrapper.temm.selection import select_near_centroid_observation_trajectories, select_near_centroid_full_trajectories
from mma_wrapper.temm.inferring_roles import extract_roles_from_trajectories, summarize_roles
from mma_wrapper.temm.inferring_goals import extract_goals_from_trajectories, summarize_goals
from mma_wrapper.temm.organizational_fit import compute_sof, compute_fof, compute_overall_fit


class TEMM:
    def __init__(self, analysis_results_path: str = "./"):
        self.analysis_results_path = analysis_results_path
        os.makedirs(os.path.join(self.analysis_results_path,
                                 "trajectories"), exist_ok=True)
        os.makedirs(os.path.join(self.analysis_results_path,
                                 "figures"), exist_ok=True)
        os.makedirs(os.path.join(self.analysis_results_path,
                                 "inferred_organizational_specifications"), exist_ok=True)

    def run_global(self,
                   distance_method_action="euclidean",
                   distance_method_obs="euclidean",
                   distance_method_full="euclidean",
                   centroid_mode="mean",
                   inclusion_radius=0.2,
                   min_rule_weight=0.5,
                   min_goal_weight=0.5):
        """
        Run the full TEMM pipeline: from data loading to fit evaluation.
        """
        print("1. Loading trajectories...")
        raw_episodes = load_trajectories(self.analysis_results_path)
        full_trajectories = extract_full_trajectories(raw_episodes)
        action_trajectories = list(
            extract_action_trajectories(raw_episodes).values())
        observation_trajectories = list(
            extract_observation_trajectories(raw_episodes).values())

        print("2. Clustering trajectories...")
        action_clusters, linkage_matrix = cluster_trajectories_from_action(
            action_trajectories, distance_method_action)
        observation_clusters, linkage_matrix = cluster_trajectories_from_observation(
            observation_trajectories, distance_method_obs)
        full_clusters, linkage_matrix = cluster_full_trajectories(
            full_trajectories, distance_method_full)

        print("3. Generating visualizations...")
        generate_actions_dendrogram(action_trajectories, save_path=os.path.join(
            self.analysis_results_path, "figures", "actions_dendrogram.png"))
        generate_observations_dendrogram(observation_trajectories, save_path=os.path.join(
            self.analysis_results_path, "figures", "observations_dendrogram.png"))
        generate_dendrogram(full_trajectories, save_path=os.path.join(
            self.analysis_results_path, "figures", "full_dendrogram.png"))
        visualize_action_pca(action_trajectories, save_path=os.path.join(
            self.analysis_results_path, "figures", "action_pca.png"))
        visualize_observation_pca(observation_trajectories, save_path=os.path.join(
            self.analysis_results_path, "figures", "observation_pca.png"))
        visualize_transition_pca(full_trajectories, save_path=os.path.join(
            self.analysis_results_path, "figures", "transition_pca.png"))
        visualize_action_trajectory(action_trajectories, save_path=os.path.join(
            self.analysis_results_path, "figures", "action_trajectory_pca.png"))
        visualize_observation_trajectory(observation_trajectories, save_path=os.path.join(
            self.analysis_results_path, "figures", "observation_trajectory_pca.png"))
        visualize_trajectory(full_trajectories, save_path=os.path.join(
            self.analysis_results_path, "figures", "full_trajectory_pca.png"))

        print("4. Computing centroids...")
        action_centroids = compute_action_centroids_per_cluster(
            action_clusters, mode=centroid_mode)
        observation_centroids = compute_observation_centroids_per_cluster(
            observation_clusters, mode=centroid_mode)
        full_centroids = compute_full_centroids_per_cluster(
            full_clusters, mode=centroid_mode)

        print("5. Selecting near-centroid trajectories...")
        selected_for_roles = select_near_centroid_full_trajectories(
            full_clusters, full_centroids, inclusion_radius)
        selected_for_goals = select_near_centroid_observation_trajectories(
            observation_clusters, observation_centroids, inclusion_radius)

        print("6. Extracting roles and goals...")
        inferred_roles = extract_roles_from_trajectories(selected_for_roles)
        inferred_goals = extract_goals_from_trajectories(selected_for_goals)

        print("7. Summarizing roles and goals...")
        summarized_roles = summarize_roles(inferred_roles, min_rule_weight)
        summarized_goals = summarize_goals(inferred_goals, min_goal_weight)

        export_to_json(summarized_roles, os.path.join(
            self.analysis_results_path, "inferred_organizational_specifications", "inferred_roles_summary.json"))
        export_to_json(summarized_goals, os.path.join(
            self.analysis_results_path, "inferred_organizational_specifications", "inferred_goals_summary.json"))

        print("8. Computing organizational fit scores...")
        sof = compute_sof(selected_for_roles, full_centroids)
        fof = compute_fof(selected_for_goals, observation_centroids)
        of = compute_overall_fit(sof, fof)

        print(f"Structural Fit (SOF): {sof:.3f}")
        print(f"Functional Fit (FOF): {fof:.3f}")
        print(f"Overall Organizational Fit (OF): {of:.3f}")
