import torch
import torch.nn as nn


class RDLM(nn.Module):

    def __init__(self, rchs_dim, latent_joint_observation_dim, onehot_joint_action_dim, rnn_hidden_dim=128, rnn_type="LSTM", rnn_layers=2, rnn_activation="None", mlp_layers=2, mlp_hidden_dim=128, mlp_activation="relu"):
        """
        rchs_dim: dimension des rchs (recurrent component hidden state)
        latent_joint_observation_dim: dimension du latent de l'observation conjointe
        onehot_joint_action_dim: dimension de l'action conjointe en one-hot
        rnn_hidden_dim: dimension cachée du RNN
        rnn_type: "LSTM" ou "GRU"
        rnn_layers: nombre de couches du RNN
        rnn_activation: activation appliquée après le RNN ("relu", "tanh", "elu", "None")
        mlp_layers: nombre de couches du MLP
        mlp_hidden_dim: dimension cachée du MLP
        mlp_activation: activation utilisée dans le MLP ("relu", "tanh", "elu")
        Return:
            z_tp1: (batch, latent_joint_observation_dim)
            rchs_t: (batch, rnn_hidden_dim)
        """

        super().__init__()

        # Stocker les paramètres d'architecture pour la sauvegarde/chargement
        self.rchs_dim = rchs_dim
        self.latent_joint_observation_dim = latent_joint_observation_dim
        self.onehot_joint_action_dim = onehot_joint_action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers
        self.rnn_activation = rnn_activation
        self.mlp_layers = mlp_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_activation = mlp_activation

        mlp_act = {"relu": nn.ReLU, "tanh": nn.Tanh,
                   "elu": nn.ELU}[mlp_activation]

        self.rnn_action = None if rnn_activation == "None" else {
            "relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[rnn_activation]

        input_dim = rchs_dim + latent_joint_observation_dim + onehot_joint_action_dim
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, rnn_hidden_dim,
                               num_layers=rnn_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, rnn_hidden_dim,
                              num_layers=rnn_layers, batch_first=True)

        # MLP pour prédire le prochain latent
        mlp_layers_list = []
        last_dim = rnn_hidden_dim
        for _ in range(mlp_layers):
            mlp_layers_list.append(nn.Linear(last_dim, mlp_hidden_dim))
            mlp_layers_list.append(mlp_act())
            last_dim = mlp_hidden_dim
        mlp_layers_list.append(
            nn.Linear(last_dim, latent_joint_observation_dim))
        self.mlp = nn.Sequential(*mlp_layers_list)

        self.rchs_tminus1 = None  # état interne
        self.reset_internal_state()

    def reset_internal_state(self, batch_size=1, device=None):
        """Réinitialise l'état interne à zéro (à appeler en début d'épisode ou de batch)."""
        if device is None:
            device = next(self.parameters()).device
        self.rchs_tminus1 = torch.zeros(
            batch_size, self.rchs_dim, device=device)

    def forward(self, z_t, a_t, rchs_tminus1=None):
        """
        z_t: (batch, latent_joint_observation_dim)
        a_t: (batch, onehot_joint_action_dim)
        rchs_tminus1: (batch, rchs_dim) ou None
        Retourne :
            z_tp1: (batch, latent_joint_observation_dim)
            rchs_t: (batch, hidden_dim)
        """
        batch_size = z_t.shape[0]
        device = z_t.device

        # Utilisation de l'état interne si non fourni
        if rchs_tminus1 is None:
            self.reset_internal_state(batch_size, device)
            rchs_tminus1 = self.rchs_tminus1
        else:
            # Vérifie la taille de rchs_tminus1
            if rchs_tminus1.shape[1] != self.rchs_dim:
                # Si la taille ne correspond pas, on réinitialise
                rchs_tminus1 = torch.zeros(
                    batch_size, self.rchs_dim, device=device)

        # print("rchs_tminus1.shape: ", rchs_tminus1.shape)
        # print("z_t.shape: ", z_t.shape)
        # print("a_t.shape: ", a_t.shape)
        x = torch.cat([rchs_tminus1, z_t, a_t], dim=-
                      1).unsqueeze(1)  # (batch, 1, input_dim)

        rnn_out, _ = self.rnn(x)
        rchs_t = rnn_out[:, -1, :]  # (batch, hidden_dim)

        if self.rnn_action is not None:
            rchs_t = self.rnn_action(rchs_t)

        z_tp1 = self.mlp(rchs_t)

        # Mise à jour de l'état interne
        if rchs_tminus1 is self.rchs_tminus1:
            self.rchs_tminus1 = rchs_t.detach()

        return z_tp1, rchs_t

    def save(self, file_path):
        """Sauvegarde le modèle RDLM avec ses paramètres d'architecture"""
        import os
        import json

        file_path = os.path.join(file_path, "model.pth")

        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Sauvegarder les poids du modèle
        model_path = file_path if file_path.endswith(
            '.pth') else file_path + '.pth'
        torch.save(self.state_dict(), model_path)

        # Sauvegarder les paramètres d'architecture
        config_path = model_path.replace('.pth', '_config.json')
        config = {
            'rchs_dim': self.rchs_dim,
            'latent_joint_observation_dim': self.latent_joint_observation_dim,
            'onehot_joint_action_dim': self.onehot_joint_action_dim,
            'rnn_hidden_dim': self.rnn_hidden_dim,
            'rnn_type': self.rnn_type,
            'rnn_layers': self.rnn_layers,
            'rnn_activation': self.rnn_activation,
            'mlp_layers': self.mlp_layers,
            'mlp_hidden_dim': self.mlp_hidden_dim,
            'mlp_activation': self.mlp_activation
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)

    @classmethod
    def load(cls, file_path):
        """Charge un modèle RDLM sauvegardé"""
        import os
        import json

        file_path = os.path.join(file_path, "model.pth")

        # Déterminer les chemins
        model_path = file_path if file_path.endswith(
            '.pth') else file_path + '.pth'
        config_path = model_path.replace('.pth', '_config.json')

        # Charger la configuration
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Fallback: essayer de deviner les paramètres depuis le chemin ou lever une exception
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}. Cannot load RDLM without architecture parameters.")

        # Créer l'instance du modèle
        model = cls(
            rchs_dim=config['rchs_dim'],
            latent_joint_observation_dim=config['latent_joint_observation_dim'],
            onehot_joint_action_dim=config['onehot_joint_action_dim'],
            rnn_hidden_dim=config['rnn_hidden_dim'],
            rnn_type=config['rnn_type'],
            rnn_layers=config['rnn_layers'],
            rnn_activation=config['rnn_activation'],
            mlp_layers=config['mlp_layers'],
            mlp_hidden_dim=config['mlp_hidden_dim'],
            mlp_activation=config['mlp_activation']
        )

        # Charger les poids
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)

        return model


def rdlm_objective(trial, latent_obs_episodes, actions_episodes, device, max_mse, hp_space):
    # RNN hyperparameters
    rchs_dim = trial.suggest_int(
        "rchs_dim", hp_space["rchs_dim"][0], hp_space["rchs_dim"][1])
    rnn_type = trial.suggest_categorical(
        "rnn_type", hp_space["rnn_type"])
    rnn_hidden_dim = trial.suggest_int(
        "rnn_hidden_dim", hp_space["rnn_hidden_dim"][0], hp_space["rnn_hidden_dim"][1])
    rnn_layers = trial.suggest_int(
        "rnn_layers", hp_space["rnn_layers"][0], hp_space["rnn_layers"][1])
    rnn_activation = trial.suggest_categorical(
        "rnn_activation", hp_space["rnn_activation"])
    learning_rate = trial.suggest_float(
        "learning_rate", hp_space["learning_rate"][0], hp_space["learning_rate"][1], log=True)

    # MLP hyperparameters
    mlp_layers = trial.suggest_int(
        "mlp_layers", hp_space["mlp_layers"][0], hp_space["mlp_layers"][1])
    mlp_hidden_dim = trial.suggest_int(
        "mlp_hidden_dim", hp_space["mlp_hidden_dim"][0], hp_space["mlp_hidden_dim"][1])
    mlp_activation = trial.suggest_categorical(
        "mlp_activation", hp_space["mlp_activation"])

    # Instanciation du modèle
    model = RDLM(
        rchs_dim=rchs_dim,
        latent_joint_observation_dim=latent_obs_episodes.shape[2],
        onehot_joint_action_dim=actions_episodes.shape[2],
        rnn_hidden_dim=rnn_hidden_dim,
        rnn_type=rnn_type,
        rnn_layers=rnn_layers,
        rnn_activation=rnn_activation,
        mlp_layers=mlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_activation=mlp_activation
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    nb_epochs = 20  # ou adapte selon ton besoin

    for epoch in range(nb_epochs):
        total_loss = 0
        for ep in range(latent_obs_episodes.shape[0]):
            model.reset_internal_state(device=device)
            for step in range(latent_obs_episodes.shape[1] - 1):
                z_t = torch.tensor(
                    latent_obs_episodes[ep][step], dtype=torch.float32, device=device).unsqueeze(0)
                a_t = torch.tensor(
                    actions_episodes[ep][step], dtype=torch.float32, device=device).unsqueeze(0)
                z_tp1 = torch.tensor(
                    latent_obs_episodes[ep][step + 1], dtype=torch.float32, device=device).unsqueeze(0)

                optimizer.zero_grad()
                z_pred, _ = model(z_t, a_t)
                loss = loss_fn(z_pred, z_tp1)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / latent_obs_episodes.shape[0]
        # Early stopping si besoin
        if max_mse == "inf" or (avg_loss < float(max_mse)):
            break

    return avg_loss
