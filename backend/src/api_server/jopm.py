import torch
import torch.nn as nn

from vae_utils import VAE
from rdlm_utils import RDLM


class JointObservationPredictionModel(nn.Module):

    def __init__(self, autoencoder: VAE, rdlm: RDLM):
        super().__init__()
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        self.rdlm = rdlm

    def reset_internal_state(self, batch_size, device):
        """Réinitialise l'état interne du RDLM (à appeler en début d'épisode ou de batch)."""
        self.rdlm.reset_internal_state(batch_size, device)

    def predict_next_joint_observation(self, obs_t, act_t, history_obs=None, history_act=None):
        """
        Si history_obs/history_act sont fournis, fonctionne en mode historique.
        Sinon, utilise et met à jour l'état interne du RDLM pour traiter transition par transition.
        """
        batch_size = obs_t.shape[0]
        device = obs_t.device

        if history_obs is not None and history_act is not None:
            # Mode historique complet
            z_hist = []
            for t in range(history_obs.shape[1]):
                z = self.encoder(history_obs[:, t, :])
                z_hist.append(z.unsqueeze(1))
            z_hist = torch.cat(z_hist, dim=1)  # (batch, hist_len, latent_dim)

            rchs = None
            for t in range(z_hist.shape[1]):
                _, rchs = self.rdlm(
                    z_hist[:, t, :], history_act[:, t, :], rchs)
            z_t = self.encoder(obs_t)
        else:
            # Mode transition par transition, on utilise l'état interne du RDLM
            z_t = self.encoder(obs_t)
            rchs = None  # RDLM utilisera son état interne

        # Utilisation du forward du RDLM (qui gère l'état interne si besoin)
        z_tp1, _ = self.rdlm(z_t, act_t, rchs)

        obs_pred_tp1 = self.decoder(z_tp1)
        return obs_pred_tp1
