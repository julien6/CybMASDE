import torch
import torch.nn as nn

import torch
import torch.nn as nn


class JointObservationPredictionModel(nn.Module):

    def __init__(self, autoencoder, rdlm):
        super().__init__()
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        self.rdlm = rdlm
        self.rchs_internal = None  # état interne, initialisé à None

    def reset_internal_state(self, batch_size, device):
        """Réinitialise l'état interne (à appeler en début d'épisode ou de batch)."""
        self.rchs_internal = torch.zeros(
            batch_size, self.rdlm.rchs_dim, device=device)

    def predict_next_joint_observation(self, obs_t, act_t, history_obs=None, history_act=None):
        """
        Si history_obs/history_act sont fournis, fonctionne comme avant.
        Sinon, utilise et met à jour l'état interne pour traiter transition par transition.
        """
        batch_size = obs_t.shape[0]
        device = obs_t.device

        if history_obs is not None and history_act is not None:
            # Mode historique complet (comme avant)
            z_hist = []
            for t in range(history_obs.shape[1]):
                z = self.encoder(history_obs[:, t, :])
                z_hist.append(z.unsqueeze(1))
            z_hist = torch.cat(z_hist, dim=1)  # (batch, hist_len, latent_dim)
            z_t = self.encoder(obs_t)
            _, rchs = self.rdlm(z_hist, history_act)
        else:
            # Mode transition par transition, on utilise l'état interne
            if self.rchs_internal is None or self.rchs_internal.shape[0] != batch_size:
                self.reset_internal_state(batch_size, device)
            rchs = self.rchs_internal
            z_t = self.encoder(obs_t)

        # Calculer la nouvelle représentation compacte à t
        # (batch, 1, input_dim)
        x = torch.cat([rchs, z_t, act_t], dim=-1).unsqueeze(1)
        rnn_out, _ = self.rdlm.rnn(x)
        rchs_t = rnn_out[:, -1, :]  # (batch, hidden_dim)
        z_tp1 = self.rdlm.mlp(rchs_t)
        obs_pred_tp1 = self.decoder(z_tp1)

        # Mettre à jour l'état interne si mode transition
        if history_obs is None or history_act is None:
            # détaché du graphe pour éviter fuite mémoire
            self.rchs_internal = rchs_t.detach()

        return obs_pred_tp1
