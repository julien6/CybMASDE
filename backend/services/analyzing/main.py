import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from pettingzoo.mpe.simple_world_comm_v2 import parallel_env
import random

# Création d'une politique "mock" qui retourne des actions aléatoires


class MockPolicy:
    def __init__(self, env):
        self.env = env

    def compute_action(self, obs, policy_id=None):
        return self.env.action_space(policy_id).sample()

# Fonction pour collecter des trajectoires avec la politique "mock" dans un environnement parallèle


def collect_trajectories(env, policy, num_episodes=10):
    trajectories = {}
    for episode_index in range(num_episodes):
        observations = env.reset()
        episode_key = f"episode_{episode_index}"
        trajectories[episode_key] = {}

        step_index = 0
        done = {agent: False for agent in env.agents}

        while not all(done.values()):
            step_key = f"step_{step_index}"
            trajectories[episode_key][step_key] = {
                "observations": {}, "actions": {}}

            actions = {agent: policy.compute_action(obs, policy_id=agent)
                       for agent, obs in observations.items() if not done[agent]}

            # Enregistrer les observations et les actions
            for agent, action in actions.items():
                trajectories[episode_key][step_key]["observations"][agent] = observations[agent].tolist(
                )
                trajectories[episode_key][step_key]["actions"][agent] = [
                    action]

            # Effectuer un pas dans l'environnement
            observations, rewards, dones, infos = env.step(actions)
            done.update(dones)
            step_index += 1

    return trajectories

# Fonction pour transformer les trajectoires en séquences d'actions pour chaque agent


def extract_action_sequences(trajectories):
    agent_sequences = {}
    for episode_key, steps in trajectories.items():
        for step_key, data in steps.items():
            for agent, action in data["actions"].items():
                if agent not in agent_sequences:
                    agent_sequences[agent] = []
                # Ajouter l'action à la séquence de l'agent
                agent_sequences[agent].append(action[0])
    return agent_sequences

# Clustering hiérarchique et visualisation du dendrogramme


def hierarchical_clustering(sequences):
    # Conversion des séquences en matrices de distance
    sequences_list = list(sequences.values())
    dist_matrix = np.array([
        [np.sum(np.array(seq1) != np.array(seq2)) / len(seq1)
         for seq2 in sequences_list]
        for seq1 in sequences_list
    ])
    linkage_matrix = linkage(dist_matrix, method='ward')

    # Affichage du dendrogramme
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, labels=list(sequences.keys()))
    plt.title("Hierarchical Clustering of Action Sequences")
    plt.xlabel("Agents")
    plt.ylabel("Distance")
    plt.show()


# Créer l'environnement PettingZoo en mode parallèle
env = parallel_env()

# Instancier la politique "mock"
mock_policy = MockPolicy(env)

# Collecter des trajectoires avec la politique "mock"
trajectories = collect_trajectories(env, mock_policy, num_episodes=5)

# Extraire les séquences d'actions par agent
action_sequences = extract_action_sequences(trajectories)

# Appliquer le clustering hiérarchique et afficher le dendrogramme
hierarchical_clustering(action_sequences)
