import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from modeling_autoencoder_LSTM import Autoencoder, LSTMTransitionModel, train_autoencoder, convert_to_latent
from torchvision import transforms

# Paramètres
latent_dim = 64
input_shape = (457, 120, 3)  # Assurez-vous que cette forme correspond aux observations réelles
num_agents = 20  # Nombre d'agents (pistons) pour le test

# Transformer pour enregistrer les images PNG
to_image = transforms.ToPILImage()

# Étape 1 : Sélectionner et afficher les premières observations et actions
def save_raw_observation_images(trajectory, agent_name="piston_0", filename_prefix="raw_obs"):
    """
    Enregistre les observations sous forme d'images PNG.
    """
    obs1, action1 = trajectory[0]
    obs2, action2 = trajectory[1]

    # On suppose ici que obs1 et obs2 sont des dictionnaires {agent_name: observation}
    for agent in obs1.keys():
        obs1_img = obs1[agent].reshape(input_shape)
        obs2_img = obs2[agent].reshape(input_shape)

        # Sauvegarder les images PNG
        to_image(torch.tensor(obs1_img).permute(2, 0, 1)).save(f"{filename_prefix}_obs1_{agent}.png")
        to_image(torch.tensor(obs2_img).permute(2, 0, 1)).save(f"{filename_prefix}_obs2_{agent}.png")

    return obs1, action1, obs2, action2

# Étape 2 : Entraîner l'autoencodeur
def train_autoencoder_and_convert_latent(trajectories, latent_dim=latent_dim):
    autoencoder, encoder = train_autoencoder(trajectories, latent_dim)
    latent_traces = convert_to_latent(encoder, trajectories)
    return autoencoder, encoder, latent_traces

# Étape 3 : Entraîner le LSTM sur les observations latentes
def train_lstm_on_latent(latent_traces, latent_dim):
    # Préparation des données
    actions, latents = [], []
    for episode in latent_traces:
        for action, latent in episode:
            actions.append(action)
            latents.append(latent)
    actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32)
    latents_tensor = torch.tensor(np.array(latents), dtype=torch.float32)

    # Créer le dataset et le DataLoader
    train_dataset = TensorDataset(actions_tensor, latents_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialiser le modèle et les paramètres
    lstm_model = LSTMTransitionModel(input_size=actions_tensor.shape[1],
                                     hidden_size=128,
                                     num_layers=2,
                                     output_size=latent_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

    # Entraîner le LSTM
    epochs = 20
    lstm_model.train()
    for epoch in range(epochs):
        for action_batch, latent_batch in train_loader:
            optimizer.zero_grad()
            output = lstm_model(action_batch.unsqueeze(1))
            loss = criterion(output, latent_batch)
            loss.backward()
            optimizer.step()

    return lstm_model

# Étape 4 : Prédire et reconstruire la prochaine observation
def predict_and_reconstruct_next_observation(lstm_model, autoencoder, action1, agent_name="piston_0", filename_prefix="predicted_obs"):
    """
    Utilise le LSTM pour prédire la prochaine observation latente, puis reconstruit cette observation
    en utilisant le décodeur de l'autoencodeur.
    """
    # Convertir action1 en tenseur
    action1_tensor = torch.tensor(np.array(action1), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Prédire la prochaine observation latente
    lstm_model.eval()
    with torch.no_grad():
        predicted_latent_observation = lstm_model(action1_tensor).squeeze(0)

    # Reconstruire l'observation à partir de l'autoencodeur
    autoencoder.eval()
    with torch.no_grad():
        predicted_observation = autoencoder.decoder(predicted_latent_observation).reshape(input_shape)

    # Sauvegarder l'image prédite en PNG
    to_image(torch.tensor(predicted_observation).permute(2, 0, 1)).save(f"{filename_prefix}_{agent_name}.png")

# Exécution du test complet
if __name__ == "__main__":
    # Charger vos trajectoires ici
    trajectories = [...]  # Charger les trajectoires réelles

    # Étape 1 : Sélection et affichage des premières observations et actions
    trajectory = trajectories[0]  # On prend la première trajectoire
    obs1, action1, obs2, action2 = save_raw_observation_images(trajectory, agent_name="piston_0")

    # Étape 2 : Entraîner l'autoencodeur et convertir les observations en latentes
    autoencoder, encoder, latent_traces = train_autoencoder_and_convert_latent(trajectories, latent_dim=latent_dim)

    # Étape 3 : Entraîner le LSTM sur les observations latentes
    lstm_model = train_lstm_on_latent(latent_traces, latent_dim=latent_dim)

    # Étape 4 : Prédiction et reconstruction de la prochaine observation
    predict_and_reconstruct_next_observation(lstm_model, autoencoder, action1, agent_name="piston_0")
