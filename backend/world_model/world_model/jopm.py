import os
import torch
import torch.nn as nn

from world_model.vae_utils import VAE
from world_model.rdlm_utils import RDLM


class JOPM(nn.Module):

    def __init__(self, autoencoder: VAE, rdlm: RDLM, initial_joint_observations: torch.Tensor):
        super().__init__()
        self.autoencoder = autoencoder
        self.rdlm = rdlm
        self.initial_joint_observations = initial_joint_observations

    def save(self, file_path):
        """Sauvegarde le modèle complet (autoencodeur + RDLM + observations initiales)."""

        # Créer les dossiers si nécessaire
        os.makedirs(os.path.join(file_path, "autoencoder"), exist_ok=True)
        os.makedirs(os.path.join(file_path, "rdlm"), exist_ok=True)

        # Autoencodeur
        self.autoencoder.save(os.path.join(file_path, "autoencoder", "model"))
        print("Autoencoder saved in ", os.path.join(
            file_path, "autoencoder/model"))

        # RDLM
        self.rdlm.save(os.path.join(file_path, "rdlm", "model"))
        print("RDLM saved in ", os.path.join(file_path, "rdlm/model"))

        # Initial joint-observations
        torch.save(self.initial_joint_observations, os.path.join(
            file_path, 'initial_joint_observations.pth'))
        print("Initial joint-observations saved in ",
              os.path.join(file_path, 'initial_joint_observations.pth'))

    @classmethod
    def load(cls, file_path):
        """Charge le modèle complet (autoencodeur + RDLM + observations initiales)."""
        # Autoencodeur
        autoencoder = VAE.load(os.path.join(
            file_path, "autoencoder", "model"))

        # RDLM
        rdlm = RDLM.load(os.path.join(file_path, "rdlm", "model"))

        # Initial joint-observations
        initial_joint_observations = torch.load(os.path.join(
            file_path, 'initial_joint_observations.pth'))

        return cls(autoencoder, rdlm, initial_joint_observations)

    def reset_internal_state(self, batch_size, device) -> torch.Tensor:
        """Réinitialise l'état interne du RDLM (à appeler en début d'épisode ou de batch)."""
        # Choisir un épisode au hasard et prendre l'observation conjointe initiale associée
        idx = torch.randint(
            0, self.initial_joint_observations.shape[0], (1,)).item()
        initial_obs = self.initial_joint_observations[idx]
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
                z, _ = self.autoencoder.encode(history_obs[:, t, :])
                z_hist.append(z.unsqueeze(1))
            z_hist = torch.cat(z_hist, dim=1)  # (batch, hist_len, latent_dim)

            rchs = None
            for t in range(z_hist.shape[1]):
                _, rchs = self.rdlm(
                    z_hist[:, t, :], history_act[:, t, :], rchs)
            z_t, _ = self.autoencoder.encode(obs_t)
        else:
            # Mode transition par transition, on utilise l'état interne du RDLM
            z_t, _ = self.autoencoder.encode(obs_t)
            rchs = None  # RDLM utilisera son état interne

        # Utilisation du forward du RDLM (qui gère l'état interne si besoin)
        z_tp1, _ = self.rdlm(z_t, act_t, rchs)

        obs_pred_tp1 = self.autoencoder.decode(z_tp1)
        return obs_pred_tp1
