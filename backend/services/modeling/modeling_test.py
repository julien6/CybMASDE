import os
import random
import tensorflow as tf
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple
from pettingzoo.butterfly import pistonball_v4
from PIL import Image
from typing import Dict, List
from typing import Dict
from tqdm import tqdm  # Pour afficher la progression
from torch.nn.functional import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from math import log10
from torch.utils.data import DataLoader, TensorDataset

# Une trajectoire ("Trajectory") est une liste de couples (observations, actions) où chaque élément est un dictionnaire {agent: List}
Trajectory = Dict[str, Dict[str, Dict[str, List]]]

Trajectories = Dict[str, Trajectory]

# Exemple de trajectoires collectées dans le fichier JSON
#
# {
# "episode_0":{
# "step_0": {"observations": {"piston_0": [58,...], "piston_1": [58,...]}, "actions": {"piston_0": [0,...], "piston_1": [0,...]}},
# "step_0": {"observations": {"piston_0": [58,...], "piston_1": [58,...]}, "actions": {"piston_0": [0,...], "piston_1": [0,...]}},
# "step_0": {"observations": {"piston_0": [58,...], "piston_1": [58,...]}, "actions": {"piston_0": [0,...], "piston_1": [0,...]}},
# ...
# },
# "episode_0":{
# "step_0": {"observations": {"piston_0": [58,...], "piston_1": [58,...]}, "actions": {"piston_0": [0,...], "piston_1": [0,...]}},
# "step_0": {"observations": {"piston_0": [58,...], "piston_1": [58,...]}, "actions": {"piston_0": [0,...], "piston_1": [0,...]}},
# "step_0": {"observations": {"piston_0": [58,...], "piston_1": [58,...]}, "actions": {"piston_0": [0,...], "piston_1": [0,...]}},
# ...
# },
# ...
# }


def collect_pistonball_trajectories(num_episodes=100, max_steps=100, file_path="trajectories.json"):
    """
    Collecte et enregistre les trajectoires de l'environnement Pistonball dans un fichier JSON incrémental.
    Une trajectoire est une liste de couples (observations, actions) où chaque élément
    est un dictionnaire {agent: List} qui pour chaque agent associe une observation/action
    applatie sous forme de liste.

    Parameters:
        num_episodes (int): Nombre d'épisodes à simuler.
        max_steps (int): Nombre maximal de pas par épisode.
        file_path (str): Chemin du fichier JSON où stocker les trajectoires.

    Returns:
        str: Le chemin vers le fichier JSON contenant les trajectoires collectées respctant le format "Trajectories".
    """
    env = pistonball_v6.parallel_env(
        n_pistons=3, random_rotate=False, random_drop=False, render_mode='rgb_array')
    env.reset()

    # Adapt to the directory name of the script
    file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), file_path)
    # Supprime le fichier existant si nécessaire
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, 'a') as f:
        f.write('{\n')  # Début du fichier JSON (liste de trajectoires)
        for episode in range(num_episodes):
            observations: Dict[str, List] = env.reset()

            f.write('"episode_' + str(episode) + '":{\n')

            for step in range(max_steps):
                actions = {agent: env.action_space(
                    agent).sample() for agent in env.agents}
                new_observations = env.step(actions)

                # Créez le vecteur d'observations pour tous les agents
                observation_vector = {
                    agent: observations[0][agent].flatten().tolist() for agent in observations[0].keys()}
                action_vector = {
                    agent: actions[agent].tolist() for agent in actions.keys()}
                # episode_trajectory.append((observation_vector, action_vector))
                trajectory_step = {
                    "observations": observation_vector, "actions": action_vector}

                observations = new_observations

                # Écrire la trajectoire de l'épisode dans le fichier
                f.write(f'"step_{str(step)}": {json.dumps(trajectory_step)}')
                if step < max_steps - 1:
                    f.write(',\n')  # Séparer chaque step par une virgule

            if episode < num_episodes - 1:
                f.write('},\n')  # Séparer chaque épisode par une virgule
            print(f"Episode {episode} completed.")

        f.write('}\n}')  # Fin du fichier JSON

    return file_path


def load_episode_step_data(file_path: str, episode_idx: int, step_idx: int):
    """
    Charge la trajectoire d'un épisode donné, extrait les actions et observations d'un pas spécifique.

    Parameters:
        file_path (str): Chemin du fichier JSON contenant les trajectoires.
        episode_idx (int): L'indice de l'épisode à charger.
        step_idx (int): Le numéro du pas de l'épisode pour lequel les observations seront concaténées.
    """
    # Variables pour suivre l'épisode et le pas actuels
    current_episode = -1
    found_episode = False
    step_data = None

    with open(file_path, 'r') as f:
        # Aller à l'ouverture de la liste des épisodes dans le JSON
        f.seek(1)  # Skip the first "[" character

        # Lecture ligne par ligne de chaque épisode
        for line in f:
            if not found_episode and f"episode_{episode_idx}" not in line:
                continue
            found_episode = True

            if found_episode:
                if f"step_{step_idx}" not in line:
                    continue
                step_data = line
                break

    if step_data is None:
        print(
            f"La trajectoire de l'épisode {episode_idx} pour l'étape {step_idx} n'a pas été trouvé dans le fichier.")
        return

    step_data = step_data.replace(f'"step_{step_idx}": ', "")
    if step_data[-5:] == "}}},\n":
        step_data = step_data[:-3]
    if step_data[-5:] == "]}}}\n":
        step_data = step_data[:-2]
    if step_data[-5:] == "]}},\n":
        step_data = step_data[:-2]
    step_data = json.loads(step_data)
    return step_data


def save_joint_observations_as_a_concatenated_image(joint_transition, output_path):
    # Concatène les observations en une seule image et l'enregistre en PNG.

    # Concaténer les observations de chaque agent en une seule liste et leur donner une forme (457, 120, 3)
    concatenated_obs = np.concatenate([np.array(obs).reshape(
        (457, 120, 3)) for obs in joint_transition["observations"].values()], axis=1)

    # Sauvegarder l'image en PNG
    img = Image.fromarray(concatenated_obs.astype('uint8'), 'RGB')
    img.save(output_path)
    print(f"Image enregistrée sous {output_path}")
    return output_path


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(457 * 120 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 457 * 120 * 3),
            nn.Sigmoid()  # pour que les valeurs soient entre 0 et 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_autoencoder(model_path: str):
    # Initialiser l'autoencodeur avec les mêmes dimensions
    autoencoder = Autoencoder()

    # Charger les poids sauvegardés dans le modèle
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.eval()  # Mettre le modèle en mode évaluation
    print(f"Autoencodeur chargé avec succès depuis {model_path}")
    return autoencoder


def calculate_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    mse = nn.functional.mse_loss(reconstructed, original)
    if mse == 0:
        return float("inf")
    max_pixel = 1.0  # Supposons que les pixels sont normalisés entre 0 et 1
    psnr = 20 * log10(max_pixel / torch.sqrt(mse))
    return psnr


def calculate_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    # Les images doivent être en format 2D ou 3D (pour des images RGB)
    return ssim(original, reconstructed, win_size=5, channel_axis=-1, data_range=1)


def calculate_variance_explained(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    var_original = torch.var(original)
    var_error = torch.var(original - reconstructed)
    ve = 1 - (var_error / var_original)
    return ve.item()


def calculate_cosine_similarity(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    original_flat = original.view(-1)
    reconstructed_flat = reconstructed.view(-1)
    return cosine_similarity(original_flat.unsqueeze(0), reconstructed_flat.unsqueeze(0)).item()


def analyze_latent_distribution(autoencoder, data_loader):
    latents = []
    for batch in data_loader:
        with torch.no_grad():
            latent = autoencoder.encoder(batch).numpy()
            latents.append(latent)
    latents = np.concatenate(latents, axis=0)
    variance_per_dimension = np.var(latents, axis=0)
    return variance_per_dimension


def plot_reconstruction_error_histogram(original: torch.Tensor, reconstructed: torch.Tensor):
    errors = ((original - reconstructed) ** 2).mean(dim=1).sqrt()
    plt.hist(errors.detach().cpu().numpy(), bins=50)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Reconstruction Errors')
    plt.show()


def train_autoencoder(file_path: str, num_episodes: int, max_steps: int, epochs: int = 5, batch_size: int = 16, learning_rate: float = 1e-3):
    # Initialisation de l'autoencodeur et de l'optimiseur
    autoencoder = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    autoencoder.train()

    # Liste pour enregistrer la perte moyenne par epoch
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for episode_idx in range(num_episodes):
            for step_idx in range(max_steps):
                # Charger les observations pour chaque étape de l'épisode
                step_data = load_episode_step_data(
                    file_path, episode_idx, step_idx)
                if step_data is None:
                    continue

                # Concaténer les observations en une seule liste et les préparer pour le modèle
                observations = np.concatenate(
                    [np.array(obs) / 255 for obs in step_data["observations"].values()])
                # Reshape en (nombre_d_observations, 457*120*3)
                observations = observations.reshape(-1, 457 * 120 * 3)

                # Diviser en batchs pour l'entraînement
                num_samples = observations.shape[0]
                for i in range(0, num_samples, batch_size):
                    batch = observations[i:i + batch_size]
                    batch = torch.tensor(batch, dtype=torch.float32)

                    # Passage avant et calcul de la perte
                    optimizer.zero_grad()
                    reconstructed = autoencoder(batch)
                    loss = criterion(reconstructed, batch)

                    # Rétropropagation et optimisation
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

        # Calcul de la perte moyenne par epoch
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Sauvegarder le modèle
    model_path = "autoencoder_model.pth"
    torch.save(autoencoder.state_dict(), model_path)
    print(f"Modèle sauvegardé sous {model_path}")

    # Sauvegarder les statistiques de performance
    basic_stats = {
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else None
    }

    stats = validate_autoencoder(
        autoencoder, file_path, num_episodes, max_steps)
    stats.update(basic_stats)
    stats_path = "autoencoder_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    print(f"Statistiques de performance sauvegardées sous {stats_path}")

    return autoencoder, stats


def validate_autoencoder(autoencoder, file_path: str, num_episodes: int, max_steps: int):
    autoencoder.eval()
    psnr_list, ssim_list, ve_list, cosine_sim_list = [], [], [], []

    for episode_idx in range(num_episodes):
        for step_idx in range(max_steps):
            step_data = load_episode_step_data(
                file_path, episode_idx, step_idx)
            if step_data is None:
                continue

            observations = np.concatenate(
                [np.array(obs) / 255 for obs in step_data["observations"].values()])
            observations = observations.reshape(-1, 457 * 120 * 3)
            observations = torch.tensor(observations, dtype=torch.float32)

            with torch.no_grad():
                reconstructed = autoencoder(observations)

            # Calcul des métriques
            for i in range(observations.size(0)):
                psnr_list.append(calculate_psnr(
                    observations[i], reconstructed[i]))
                ssim_list.append(calculate_ssim(observations[i].reshape(
                    457, 120, 3).numpy(), reconstructed[i].reshape(457, 120, 3).numpy()))
                ve_list.append(calculate_variance_explained(
                    observations[i], reconstructed[i]))
                cosine_sim_list.append(calculate_cosine_similarity(
                    observations[i], reconstructed[i]))

    # Moyenne des métriques
    stats = {
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "Variance Explained": float(np.mean(ve_list)),
        "Cosine Similarity": float(np.mean(cosine_sim_list))
    }
    print("Statistiques de validation :", stats)
    return stats


def save_random_observation_and_reconstruction(autoencoder, file_path: str, episode_idx: int, step_idx: int, original_output_path: str, reconstructed_output_path: str):
    """
    Sélectionne une observation aléatoire, la passe dans l'autoencodeur et sauvegarde l'observation originale
    et la reconstruction en PNG.

    Parameters:
        autoencoder (torch.nn.Module): Le modèle d'autoencodeur entraîné.
        file_path (str): Chemin du fichier JSON contenant les trajectoires.
        episode_idx (int): L'indice de l'épisode à charger.
        step_idx (int): Le numéro du pas de l'épisode pour lequel les observations seront utilisées.
        original_output_path (str): Chemin pour sauvegarder l'observation originale en PNG.
        reconstructed_output_path (str): Chemin pour sauvegarder la reconstruction en PNG.
    """
    # Charger les données de l'épisode et du step
    step_data = load_episode_step_data(file_path, episode_idx, step_idx)
    if step_data is None:
        print(
            f"Les données pour l'épisode {episode_idx} et le pas {step_idx} n'ont pas pu être chargées.")
        return

    # Concaténer toutes les observations et choisir une au hasard
    observations = np.concatenate(
        [np.array(obs) for obs in step_data["observations"].values()]) / 255
    # Reshape en (nombre_observations, 457*120*3)
    observations = observations.reshape(-1, 457 * 120 * 3)
    random_idx = random.randint(0, observations.shape[0] - 1)
    random_observation = observations[random_idx]

    # Reshape pour obtenir l'image originale (457, 120, 3)
    original_image = random_observation.reshape(457, 120, 3)

    # Sauvegarder l'observation originale en PNG
    # Convertir en 0-255 si nécessaire
    original_img = Image.fromarray((original_image * 255).astype(np.uint8))
    original_img.save(original_output_path)
    print(f"Observation originale sauvegardée sous {original_output_path}")

    # Passer l'observation dans l'autoencodeur pour obtenir la reconstruction
    autoencoder.eval()  # S'assurer que le modèle est en mode évaluation
    with torch.no_grad():
        input_tensor = torch.tensor(random_observation, dtype=torch.float32).unsqueeze(
            0)  # Ajouter la dimension batch
        reconstructed_tensor = autoencoder(input_tensor).squeeze(
            0)  # Supprimer la dimension batch
        reconstructed_image = reconstructed_tensor.numpy().reshape(457, 120, 3)

    # Sauvegarder la reconstruction en PNG
    reconstructed_img = Image.fromarray(
        (reconstructed_image * 255).astype(np.uint8))
    reconstructed_img.save(reconstructed_output_path)
    print(f"Image reconstruite sauvegardée sous {reconstructed_output_path}")


def convert_to_latent_trajectories(autoencoder: torch.nn.Module, input_file: str, output_file: str = "latent_trajectories.json"):
    """
    Convertit les observations d'un fichier d'épisodes en observations latentes et sauvegarde dans un nouveau fichier JSON.

    Parameters:
        autoencoder (torch.nn.Module): Le modèle d'autoencodeur entraîné.
        input_file (str): Chemin du fichier JSON contenant les trajectoires initiales.
        output_file (str): Chemin du fichier JSON pour sauvegarder les trajectoires avec les observations latentes.

    Returns:
        str: Le chemin vers le fichier JSON contenant les trajectoires latentes.
    """
    autoencoder.eval()  # S'assurer que l'autoencodeur est en mode évaluation

    # Supprime le fichier de sortie s'il existe déjà
    if os.path.exists(output_file):
        os.remove(output_file)

    # Ouvrir le fichier de sortie pour écrire les trajectoires de manière incrémentale
    with open(output_file, 'a') as f_out:
        f_out.write("{\n")  # Début du fichier JSON

        with open(input_file, 'r') as f_in:
            trajectories = json.load(f_in)
            first_episode = True

            # Itérer sur chaque épisode
            for episode_key, steps in trajectories.items():
                if not first_episode:
                    # Séparer chaque épisode par une virgule
                    f_out.write(",\n")
                first_episode = False

                f_out.write(f'"{episode_key}": {{\n')

                first_step = True
                # Itérer sur chaque étape (step) de l'épisode
                for step_key, data in steps.items():
                    if not first_step:
                        # Séparer chaque step par une virgule
                        f_out.write(",\n")
                    first_step = False

                    # Extraire les observations et les actions
                    real_observations = data["observations"]
                    actions = data["actions"]

                    # Concaténer les observations en une seule matrice pour l'encoder
                    observations_matrix = np.stack(
                        [np.array(obs) for obs in real_observations.values()])
                    observations_tensor = torch.tensor(
                        observations_matrix, dtype=torch.float32)

                    # Passer les observations dans l'autoencodeur pour obtenir les observations latentes
                    with torch.no_grad():
                        latent_observations = autoencoder.encoder(
                            observations_tensor).numpy()

                    # Créer une structure de dictionnaire pour les observations latentes
                    latent_observations_dict = {
                        agent: latent_observations[i].tolist() for i, agent in enumerate(real_observations.keys())
                    }

                    # Écrire les données latentes dans le fichier de sortie
                    latent_step = {
                        "observations": latent_observations_dict,
                        "actions": actions
                    }
                    f_out.write(f'"{step_key}": {json.dumps(latent_step)}')

                f_out.write("\n}")  # Fin de l'épisode

        f_out.write("\n}")  # Fin du fichier JSON

    print(f"Trajectoires latentes sauvegardées dans {output_file}")
    return output_file

# Définition du modèle LSTM


class ActionToObservationLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super(ActionToObservationLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out


# Définition du modèle LSTM
class ActionToObservationLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super(ActionToObservationLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out


def train_lstm_from_latent_trajectories(latent_trajectories_file: str, lstm_model_path: str = "lstm_model.pth", stats_path: str = "lstm_stats.json"):
    # Chargement des trajectoires latentes
    with open(latent_trajectories_file, 'r') as f:
        latent_trajectories = json.load(f)

    # Préparation des données (actions, observations)
    actions, next_observations = [], []
    for episode in latent_trajectories.values():
        steps = list(episode.values())
        for i in range(len(steps) - 1):
            # Concaténer toutes les actions et observations de chaque agent pour chaque étape
            action_concat = np.concatenate(
                [steps[i]["actions"][agent] for agent in steps[i]["actions"]])
            observation_concat = np.concatenate(
                [steps[i + 1]["observations"][agent] for agent in steps[i + 1]["observations"]])

            actions.append(action_concat)
            next_observations.append(observation_concat)

    # Conversion des actions et des observations en tenseurs
    # Dimension : (batch, seq_len=1, input_dim)
    actions_tensor = torch.tensor(
        actions, dtype=torch.float32).reshape(-1, 1, len(actions[0]))
    next_observations_tensor = torch.tensor(next_observations, dtype=torch.float32).reshape(
        -1, len(next_observations[0]))  # Dimension : (batch, output_dim)

    # Création des ensembles d'entraînement et de validation
    dataset = TensorDataset(actions_tensor, next_observations_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Initialisation du modèle, de la fonction de perte et de l'optimiseur
    # Dimension de l'action concaténée pour tous les agents
    input_dim = len(actions[0])
    hidden_dim = 128  # Hyperparamètre, ajustable
    # Dimension de l'observation latente concaténée pour tous les agents
    output_dim = len(next_observations[0])
    num_layers = 2   # Hyperparamètre, ajustable

    model = ActionToObservationLSTM(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Entraînement du modèle LSTM
    num_epochs = 20  # Hyperparamètre, ajustable
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for actions_batch, observations_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(actions_batch)
            loss = criterion(predictions, observations_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for actions_batch, observations_batch in val_loader:
                predictions = model(actions_batch)
                loss = criterion(predictions, observations_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Sauvegarde du modèle LSTM
    torch.save(model.state_dict(), lstm_model_path)
    print(f"Modèle LSTM sauvegardé sous {lstm_model_path}")

    # Sauvegarde des statistiques
    stats = {
        "train_loss": train_losses,
        "validation_loss": val_losses,
        "final_train_loss": train_losses[-1],
        "final_validation_loss": val_losses[-1]
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    print(f"Statistiques d'entraînement sauvegardées sous {stats_path}")

    return model, stats


if __name__ == '__main__':

    # Collecter les trajectoires dans le format ""
    # file_path = collect_pistonball_trajectories(num_episodes=2, max_steps=2)
    # print(f"Trajectoires enregistrées dans {file_path}")

    file_path = "/home/julien/Documents/Thèse/CybMASDE/backend/src/modeling/trajectories.json"

    # # Juste pour un exemple, sauvegarder les observations concaténées en une seule image
    # obs_act_data = load_episode_step_data(file_path, 0, 1)
    # save_joint_observations_as_a_concatenated_image(
    #     obs_act_data, "observation_image.png")

    # Entraîner le modèle auto-encodeur avec toutes les observations de toutes les trajectoires collectées
    # autoencoder, statistics = train_autoencoder(
    #     file_path, num_episodes=2, max_steps=2, batch_size=32, epochs=100)

    model_path = "/home/julien/Documents/Thèse/CybMASDE/backend/src/modeling/autoencoder_model.pth"

    # Charger l'autoencodeur
    # autoencoder = load_autoencoder(model_path)

    # save_random_observation_and_reconstruction(
    #     autoencoder, file_path, 0, 1, "original_observation.png", "reconstructed_observation.png")

    # convert_to_latent_trajectories(autoencoder, file_path, "latent_trajectories.json")

    latent_trajectories_file = "/home/julien/Documents/Thèse/CybMASDE/backend/src/modeling/latent_trajectories.json"
    lstm_model_path = "lstm_model.pth"
    stats_path = "lstm_stats.json"

    # Entraîner le LSTM et sauvegarder les résultats
    trained_lstm, lstm_stats = train_lstm_from_latent_trajectories(
        latent_trajectories_file, lstm_model_path, stats_path)
    print(
        f"Modèle LSTM entraîné et statistiques sauvegardées sous {lstm_model_path} et {stats_path}")
