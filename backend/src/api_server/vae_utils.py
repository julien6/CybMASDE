import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

# --- Définition du VAE ---


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, n_layers, activation):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]
        # Encodeur
        encoder_layers = []
        last_dim = input_dim
        for _ in range(n_layers):
            encoder_layers.append(nn.Linear(last_dim, hidden_dim))
            encoder_layers.append(act())
            last_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Décodeur
        decoder_layers = []
        last_dim = latent_dim
        for _ in range(n_layers):
            decoder_layers.append(nn.Linear(last_dim, hidden_dim))
            decoder_layers.append(act())
            last_dim = hidden_dim
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# --- Fonction de perte VAE ---


def vae_loss(recon_x, x, mu, logvar, kl_weight=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss.item()

# --- Fonction d'entraînement ---


def train_vae(model, optimizer, train_loader, val_loader, kl_weight, device, epochs=50, early_stop_patience=5):
    best_val_loss = float('inf')
    patience = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, _ = vae_loss(recon, x, mu, logvar, kl_weight)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                recon, mu, logvar = model(x)
                loss, _ = vae_loss(recon, x, mu, logvar, kl_weight)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience >= early_stop_patience:
            break
    return best_val_loss

# --- Fonction objectif Optuna ---


def objective(trial, observations, input_dim, device, max_mse, hp_space):
    # 1. Echantillonnage des hyper-paramètres à partir de hp_space
    latent_dim = trial.suggest_categorical(
        "latent_dim", hp_space["latent_dim"])
    n_layers = trial.suggest_int(
        "n_layers", hp_space["n_layers"][0], hp_space["n_layers"][1])
    hidden_dim = trial.suggest_categorical(
        "hidden_dim", hp_space["hidden_dim"])
    activation = trial.suggest_categorical(
        "activation", hp_space["activation"])
    lr = trial.suggest_loguniform("lr", hp_space["lr"][0], hp_space["lr"][1])
    batch_size = trial.suggest_categorical(
        "batch_size", hp_space["batch_size"])
    kl_weight = trial.suggest_uniform(
        "kl_weight", hp_space["kl_weight"][0], hp_space["kl_weight"][1])

    # 2. Préparation des données (inchangé)
    obs = torch.tensor(observations, dtype=torch.float32)
    n = len(obs)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    train_data = obs[train_idx]
    val_data = obs[val_idx]
    train_loader = DataLoader(TensorDataset(
        train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data),
                            batch_size=batch_size, shuffle=False)

    # 3. Construction du modèle
    model = VAE(input_dim, latent_dim, hidden_dim,
                n_layers, activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Entraînement
    val_loss = train_vae(model, optimizer, train_loader,
                         val_loader, kl_weight, device)

    # 5. Retourne la loss de validation (Optuna cherche à la minimiser)
    if val_loss < max_mse:
        torch.save(model.state_dict(), "best_vae.pth")
    return val_loss
