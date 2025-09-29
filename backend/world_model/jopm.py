import os
import torch
import torch.nn as nn

from vae_utils import VAE
from rdlm_utils import RDLM


class JOPM(nn.Module):

    def __init__(self, autoencoder: VAE, rdlm: RDLM, initial_joint_observations: torch.Tensor):
        super().__init__()
        self.autoencoder = autoencoder
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        self.rdlm = rdlm
        self.initial_joint_observations = initial_joint_observations

    def save(self, file_path):
        """Sauvegarde le modèle complet (autoencodeur + RDLM + observations initiales)."""

        # Autoencodeur
        torch.save(self.autoencoder.state_dict(), os.path.join(
            file_path, "autoencoder", "model.pth"))
        print("Autoencoder saved in ", os.path.join(
            file_path, "autoencoder", "model.pth"))

        # RDLM
        torch.save(self.rdlm.state_dict(), os.path.join(
            file_path, "rdlm", "model.pth"))
        print("RDLM saved in ", os.path.join(file_path, "rdlm", "model.pth"))

        # Initial joint-observations
        torch.save(self.initial_joint_observations, os.path.join(
            file_path, 'initial_joint_observations.json'))
        print("Initial joint-observations saved in ",
              os.path.join(file_path, 'initial_joint_observations.json'))

    @classmethod
    def load(self, file_path):
        """Charge le modèle complet (autoencodeur + RDLM + observations initiales)."""
        # Autoencodeur
        autoencoder = VAE.load(os.path.join(
            file_path, "autoencoder", "model.pth"))

        # RDLM
        rdlm = RDLM.load(os.path.join(file_path, "rdlm", "model.pth"))

        # Initial joint-observations
        initial_joint_observations = torch.load(os.path.join(
            file_path, 'initial_joint_observations.json'))

        return JOPM(autoencoder, rdlm, initial_joint_observations)

    def reset_internal_state(self, batch_size, device) -> torch.Tensor:
        """Réinitialise l'état interne du RDLM (à appeler en début d'épisode ou de batch)."""
        # Choisir un épisode au hasard et prendre l'observation conjointe initiale associée
        idx = torch.randint(0, self.initial_joint_observations.shape[0])
        initial_obs = self.initial_joint_observations[idx].to(device)
        self.rdlm.reset_internal_state(batch_size, device)
        return initial_obs

    def predict_next_joint_observation(self, obs_t, act_t, history_obs=None, history_act=None) -> torch.Tensor:
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
