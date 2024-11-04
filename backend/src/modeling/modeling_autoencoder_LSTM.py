from typing import List, Tuple, Dict
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
import os
import optuna
from sklearn.metrics import mean_squared_error, r2_score

Trajectory = List[Tuple[np.ndarray, np.ndarray]]


class Autoencoder(nn.Module):
    """
    Autoencoder to generate a latent representation of each observation.
    """

    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class LSTMTransitionModel(nn.Module):
    """
    LSTM-based model to predict the next latent observation for all agents based on their actions.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMTransitionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Fully connected layer on last time step
        return out


def train_autoencoder(trajectories, latent_dim):
    """
    Trains an autoencoder to reduce observation dimensions.
    Parameters:
        trajectories: List of original observations to train on.
        latent_dim: Dimension of the latent space.
    Returns:
        Trained autoencoder model and the encoder part.
    """
    # Extract and flatten observations
    observations = [obs.flatten()
                    for episode in trajectories for obs, _ in episode]
    observations = torch.tensor(np.array(observations), dtype=torch.float32)
    input_dim = observations.shape[1]

    # Initialize autoencoder and train
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # DataLoader for batch training
    dataset = TensorDataset(observations, observations)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Train autoencoder
    epochs = 50
    for epoch in range(epochs):
        for obs_batch, _ in data_loader:
            optimizer.zero_grad()
            obs_recon, _ = autoencoder(obs_batch)
            loss = criterion(obs_recon, obs_batch)
            loss.backward()
            optimizer.step()

    return autoencoder, autoencoder.encoder


def convert_to_latent(encoder, trajectories):
    """
    Convert observations in trajectories to latent space.
    """
    latent_traces = []
    for episode in trajectories:
        latent_episode = []
        for obs, action in episode:
            obs_flat = torch.tensor(obs.flatten(), dtype=torch.float32)
            # Convert to latent space
            _, latent_obs = encoder(obs_flat.unsqueeze(0))
            latent_episode.append(
                (action, latent_obs.squeeze().detach().numpy()))
        latent_traces.append(latent_episode)
    return latent_traces


def get_transition_function_from_traces(trajectories, params=None, latent_dim=64):
    """
    Trains an LSTM model to predict the next latent observation vectors based on action vectors for all agents.
    """
    # Step 1: Train the autoencoder to generate latent representations
    autoencoder, encoder = train_autoencoder(trajectories, latent_dim)

    # Step 2: Convert trajectories to latent space
    latent_traces = convert_to_latent(encoder, trajectories)

    # Flatten the traces for training
    def flatten_traces(traces):
        actions = []
        latents = []
        for episode in traces:
            for action, latent in episode:
                actions.append(action)
                latents.append(latent)
        return torch.tensor(np.array(actions), dtype=torch.float32), torch.tensor(np.array(latents), dtype=torch.float32)

    actions_train, latents_train = flatten_traces(latent_traces)

    # Dataset and DataLoader
    train_dataset = TensorDataset(actions_train, latents_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # LSTM Model Training with Optuna
    def objective(trial):
        hidden_size = trial.suggest_int("hidden_size", 32, 128)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        batch_size = trial.suggest_int("batch_size", 16, 64)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        epochs = trial.suggest_int("epochs", 10, 50)

        model = LSTMTransitionModel(input_size=actions_train.shape[1], hidden_size=hidden_size,
                                    num_layers=num_layers, output_size=latent_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training Loop
        model.train()
        for epoch in range(epochs):
            for action_batch, latent_batch in train_loader:
                optimizer.zero_grad()
                output = model(action_batch.unsqueeze(1))
                loss = criterion(output, latent_batch)
                loss.backward()
                optimizer.step()

        return loss.item()

    # Optimize hyperparameters
    if params is None:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        best_params = study.best_params
    else:
        best_params = params

    # Final model training with optimal parameters
    model = LSTMTransitionModel(input_size=actions_train.shape[1],
                                hidden_size=best_params["hidden_size"],
                                num_layers=best_params["num_layers"],
                                output_size=latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"])

    model.train()
    for epoch in range(best_params["epochs"]):
        for action_batch, latent_batch in train_loader:
            optimizer.zero_grad()
            output = model(action_batch.unsqueeze(1))
            loss = criterion(output, latent_batch)
            loss.backward()
            optimizer.step()

    # Save model and statistics
    model_path = os.path.join(
        os.getcwd(), f"transition_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    torch.save(model.state_dict(), model_path)
    print("Model saved to", model_path)

    return model, {"parameters": best_params}
