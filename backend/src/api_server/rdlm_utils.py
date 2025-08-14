import torch
import torch.nn as nn


class RDLM(nn.Module):
    def __init__(self, rchs_dim, latent_obs_dim, action_dim, hidden_dim, rnn_type="LSTM", rnn_layers=2, mlp_layers=2, mlp_hidden_dim=128, activation="relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]
        self.rchs_dim = rchs_dim
        self.latent_obs_dim = latent_obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers

        input_dim = rchs_dim + latent_obs_dim + action_dim
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim,
                               num_layers=rnn_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim,
                              num_layers=rnn_layers, batch_first=True)

        # MLP pour prédire le prochain latent
        mlp_layers_list = []
        last_dim = rchs_dim
        for _ in range(mlp_layers):
            mlp_layers_list.append(nn.Linear(last_dim, mlp_hidden_dim))
            mlp_layers_list.append(act())
            last_dim = mlp_hidden_dim
        mlp_layers_list.append(nn.Linear(last_dim, latent_obs_dim))
        self.mlp = nn.Sequential(*mlp_layers_list)


def forward(self, z_seq, a_seq):
    """
    z_seq: (batch, seq_len, latent_obs_dim)
    a_seq: (batch, seq_len, action_dim)
    Retourne :
        z_tp1: (batch, latent_obs_dim)  # dernière prédiction
        rchs: (batch, hidden_dim)       # dernier état caché compact
    """
    batch_size, seq_len, _ = z_seq.shape
    device = z_seq.device

    rchs = torch.zeros(batch_size, self.rchs_dim, device=device)
    for t in range(seq_len):
        z_t = z_seq[:, t, :]
        a_t = a_seq[:, t, :]
        # (batch, 1, input_dim)
        x = torch.cat([rchs, z_t, a_t], dim=-1).unsqueeze(1)
        rnn_out, _ = self.rnn(x)
        rchs = rnn_out[:, -1, :]  # (batch, hidden_dim)
    z_tp1 = self.mlp(rchs)  # prédiction finale à partir du dernier rchs
    return z_tp1, rchs


if __name__ == "__main__":
    # Exemple d'utilisation
    rchs_dim = 10
    latent_obs_dim = 5
    action_dim = 3
    hidden_dim = 64
    model = RDLM(rchs_dim, latent_obs_dim, action_dim, hidden_dim)
    print(model.forward())
