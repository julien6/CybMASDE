from copy import copy
import os
import json
import time
import math
import gym
import random
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna

from torch.utils.data import DataLoader, TensorDataset
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray import tune
import yaml
from environment_api import EnvironmentAPI
from project import Configuration, Modelling, Training, Analyzing
from project import current_project
from flask import request, jsonify, Response
from project import Project
from multiprocessing import Process
from random import randint
from typing import Any, List
from marllib import marl
from mma_wrapper.label_manager import label_manager
from mma_wrapper.organizational_model import deontic_specification, organizational_model, structural_specifications, functional_specifications, deontic_specifications, time_constraint_type
from mma_wrapper.organizational_specification_logic import role_logic, goal_factory, role_factory, goal_logic
from mma_wrapper.utils import label, observation, action, trajectory
from marllib.envs.base_env.wmt import RLlibWMT
from collections import OrderedDict
from ray.tune.stopper import Stopper
from vae_utils import VAE, vae_loss, objective


class MeanStdStopper(Stopper):
    def __init__(self, mean_threshold, std_threshold, window_size=100):
        self.mean_threshold = mean_threshold
        self.std_threshold = std_threshold
        self.window_size = window_size
        self.episode_rewards = []

    def __call__(self, trial_id, result):
        rewards = result.get("hist_stats", {}).get("episode_reward", None)

        if rewards is not None:
            self.episode_rewards.extend(rewards)
            if len(self.episode_rewards) >= self.window_size:
                self.episode_rewards = self.episode_rewards[-self.window_size:]
                mean = np.mean(self.episode_rewards)
                std = np.std(self.episode_rewards)
                print(
                    f"Average over {len(self.episode_rewards)} episodes : {mean:.2f}")
                print(
                    f"Standard deviation over {len(self.episode_rewards)} episodes : {std:.2f}")
                if mean >= self.mean_threshold and std <= self.std_threshold:
                    print(
                        f"Early stopping : average >= {self.mean_threshold} and standard deviation <= {self.std_threshold}")
                    return True  # Stop this trial
        return False  # Continue


cybmasde_conf = json.load(
    open(os.path.join(os.path.expanduser("~"), ".cybmasde")), "r")

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MTAProcess(Process):

    def __init__(self, configuration: Configuration, modelling: Modelling, training: Training, analyzing: Analyzing):
        super().__init__()
        self.configuration = configuration
        self.modelling = modelling
        self.training = training
        self.analyzing = analyzing

    def run(self):
        """Run the MTA process."""
        print("Running MTA process with configuration:", self.configuration)

        # Run each activity of the MTA process
        self.run_modelling_activity()

        for i in range(self.configuration.max_refinement_cycle):
            print(
                f"Refinement cycle {i + 1}/{self.configuration.max_refinement_cycle}")
            self.run_training_activity()
            self.run_analyzing_activity()

    def run_modelling_activity(self):
        """Run the modelling activity."""
        print("Running modelling activity...")

        if not self.modelling.simulated_environment.environment_path:
            print(
                "No ready-to-use simulated environment path provided, generating a new one with World Models...")

            # Check if traces and reward are provided
            if not self.modelling.simulated_environment.generated_environment.world_model.used_traces_path:
                raise ValueError(
                    "Traces path is not provided to generate the world model.")

            if not self.self.modelling.simulated_environment.generated_environment.reward_function:
                raise ValueError(
                    "Reward function is not provided to generate the simulated environment.")

            if not self.self.modelling.simulated_environment.generated_environment.reward_function:
                raise ValueError(
                    "Reward function is not provided to generate the simulated environment.")

            if not self.self.modelling.simulated_environment.generated_environment.rendering_function:
                logger.warning(
                    "No ready-to-use rendering function provided, using raw rendering function...")

            ##############################################################
            # Generate the world model
            ##############################################################

            # Check if the world model hyperparameters are provided else use default ones
            if self.modelling.simulated_environment.generated_environment.world_model.hyperparameters is None:
                print(
                    "No world model hyperparameters research space provided, using default ones...")
                self.modelling.simulated_environment.generated_environment.world_model.hyperparameters = cybmasde_conf[
                    "default_world_model_hyperparameters"]

            # Run the world model training with hyperparameter optimization (HPO)
            print("Running world model hyperparameter optimization...")
            self.run_world_model_with_hpo()

            # Assemble the reward function, world model (and optionally the rendering function) into the simulated environment
            print("Assembling the simulated environment...")
            self.assemble_simulated_environment()

    def run_world_model_with_hpo(self):
        """Run the world model training with hyperparameter optimization (HPO)."""
        print("Running world model training with hyperparameter optimization...")

        self.run_autoencoder_with_hpo()

        self.run_rdlm_with_hpo()

    def load_joint_observations(self, traces_path):
        """
        Charge toutes les observations conjointes à partir des fichiers de trajectoires dans traces_path.
        Retourne une liste d'observations conjointes (chaque observation conjointe = concaténation des observations de tous les agents à un pas de temps donné).
        """
        joint_observations = []

        # Parcours de tous les fichiers trajectories_*.json dans le dossier
        for fname in os.listdir(traces_path):
            if fname.startswith("trajectories_") and fname.endswith(".json"):
                with open(os.path.join(traces_path, fname), "r") as f:
                    data = json.load(f)
                # On suppose que data est un dict {agent_id: [ [obs, ...], ... ]}
                # On récupère la longueur de la trajectoire (nombre de pas)
                agent_ids = list(data.keys())
                traj_len = len(data[agent_ids[0]])
                # Pour chaque pas de temps, on concatène les observations de tous les agents
                for t in range(traj_len):
                    obs_t = []
                    for agent in agent_ids:
                        # [obs, ...], on prend la première entrée (l'observation)
                        obs = data[agent][t][0]
                        obs_t.extend(obs)
                    joint_observations.append(
                        np.array(obs_t, dtype=np.float32))
        return joint_observations

    def run_autoencoder_with_hpo(self):
        """Run the autoencoder training with hyperparameter optimization (HPO)."""
        print("Running autoencoder training with hyperparameter optimization...")

        hp_space = self.modelling.simulated_environment.generated_environment.autoencoder.hyperparameters
        max_mse = self.modelling.simulated_environment.generated_environment.autoencoder.max_mean_square_error
        traces_path = self.modelling.simulated_environment.generated_environment.world_model.used_traces_path

        observations = self.load_joint_observations(traces_path)
        input_dim = len(observations[0])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hp_space = {
        #     "latent_dim": [8, 16, 32, 64],
        #     "n_layers": [1, 3],  # borne min et max pour suggest_int
        #     "hidden_dim": [64, 128, 256, 512],
        #     "activation": ["relu", "tanh", "elu"],
        #     "lr": [1e-4, 1e-2],  # borne min et max pour loguniform
        #     "batch_size": [32, 64, 128],
        #     "kl_weight": [0.1, 1.0]  # borne min et max pour uniform
        # }

        def optuna_objective(trial):
            return objective(trial, observations, input_dim, device, max_mse, hp_space)

        study = optuna.create_study(direction="minimize")
        # Ajuste n_trials selon tes ressources
        study.optimize(optuna_objective, n_trials=30)

        print("Best hyperparameters:", study.best_params)
        print("Best validation MSE:", study.best_value)

        # Recharge le meilleur modèle
        best_model = VAE(input_dim, **{k: study.best_params[k] for k in [
            "latent_dim", "hidden_dim", "n_layers", "activation"]}).to(device)
        best_model.load_state_dict(torch.load("best_vae.pth"))

        # Mets à jour les hyperparamètres dans l'objet de config
        self.modelling.simulated_environment.generated_environment.autoencoder.hyperparameters = study.best_params

        # Sauvegarde finale du modèle
        torch.save(best_model.state_dict(), "autoencoder_model.pth")
        print("Best autoencoder saved as autoencoder_model.pth")

    def run_rdlm_with_hpo(self):
        """Run the RLDM (RNN+MLP) training with hyperparameter optimization (HPO)."""
        print("Running RLDM training with hyperparameter optimization...")

        # 1. Charger l'espace de recherche des hyperparamètres
        hp_space = self.modelling.simulated_environment.generated_environment.rdlm.hyperparameters
        max_mse = self.modelling.simulated_environment.generated_environment.rdlm.max_mean_square_error

        # 2. Charger les trajectoires latentes et les actions conjointes
        latent_trajectories = load_latent_trajectories(
            "latent_trajectories.pkl")  # À implémenter
        actions_trajectories = [traj["joint_actions"]
                                for traj in latent_trajectories]

        # 3. Initialiser les variables pour le meilleur modèle
        best_hp = None
        best_mse = float('inf')
        best_model = None

        # 4. Générer toutes les combinaisons d'hyperparamètres (grid search)
        for hp_config in generate_grid(hp_space):  # À implémenter
            # 4.a. Initialiser et entraîner le RLDM (RNN+MLP)
            rdlm = train_rdlm(
                latent_trajectories=latent_trajectories,
                actions=actions_trajectories,
                hyperparameters=hp_config
            )  # À implémenter

            # 4.b. Évaluer la performance (MSE prédiction)
            mse = evaluate_rdlm(
                rdlm,
                latent_trajectories=latent_trajectories,
                actions=actions_trajectories
            )  # À implémenter

            print(f"Tested config: {hp_config} -> MSE: {mse:.4f}")

            # 4.c. Garder le meilleur modèle sous le seuil
            if mse < best_mse and mse <= max_mse:
                best_mse = mse
                best_hp = hp_config
                best_model = rdlm

        if best_model is None:
            raise RuntimeError("No RLDM reached the required MSE threshold.")

        # 5. Mettre à jour les hyperparamètres avec la meilleure config
        self.modelling.simulated_environment.generated_environment.rdlm.hyperparameters = best_hp

        # 6. Sauvegarder le modèle RLDM entraîné (RNN + MLP)
        save_rdlm(best_model, "rdlm_model.pth")  # À implémenter

        print(f"Best RLDM config: {best_hp} with MSE: {best_mse:.4f}")

    def run_training_activity(self):
        """Run the training activity."""
        print("Running training activity...")

        # Check if hyperparameters intervals are provided, else use default ones
        if not self.training.hyperparameters:
            print(
                "No training hyperparameters research space provided, using default ones...")
            self.training.hyperparameters = cybmasde_conf["default_training_hyperparameters"]

        best_result = None
        best_algorithm = None
        best_checkpoint_path = None
        for algorithm in self.training.hyperparameters["algorithms"]:

            env = marl.make_env(
                environment_name="simulated_environment", map_name="default", force_coop=False, organizational_model=self.modelling.organizational_specifications)

            mappo = marl.algos.getattr(algorithm)(
                hyperparam_source="common", **{k: v for k, v in self.training.hyperparameters["algorithms"][algorithm]["algorithm"].items(
                ) if k in list(self.load_algorithm_default_hp(algorithm).keys())})

            model = marl.build_model(
                env, mappo, self.training.hyperparameters[algorithm]["model"])

            experiment_analysis: ExperimentAnalysis = mappo.fit(
                env,
                model,
                stop=MeanStdStopper(mean_threshold=self.training.configuration.mean_threshold,
                                    std_threshold=self.training.configuration.std_threshold, window_size=self.training.configuration.window_size),
                local_mode=False,
                num_gpus=self.training.configuration.num_gpus,
                num_workers=self.training.configuration.num_workers,
                share_policy='all',
                checkpoint_freq=self.training.configuration.checkpoint_freq,
                checkpoint_end=True)

            best_config = experiment_analysis.get_best_config(
                metric="episode_reward_mean", mode="max")
            print("Best hyperparameters found with algorithm {}: {}".format(
                algorithm, best_config))
            best_trial = experiment_analysis.get_best_trial(
                metric="episode_reward_mean", mode="max")
            if best_result is None or best_trial.best_result > best_result:
                best_result = best_trial.metric_analysis["episode_reward_mean"]["max"]
                best_algorithm = algorithm
                best_checkpoint_path = best_trial.checkpoint.value
            self.training.hyperparameters["algorithms"][algorithm] = best_config

        best_hp = copy.deepcopy(
            self.training.hyperparameters["algorithms"][best_algorithm])
        self.training.hyperparameters["algorithms"] = {}
        self.training.hyperparameters["algorithms"][best_algorithm] = {}
        self.training.hyperparameters["algorithms"][best_algorithm]["algorithm"] = {
            k: v for k, v in best_hp.itmes if k in list(self.load_algorithm_default_hp(best_algorithm).keys())}
        self.training.hyperparameters["algorithms"][best_algorithm]["model"] = {
            k: v for k, v in best_hp["model"]["model_arch_args"].items() if k in ["core_arch", "mixer_arch", "encode_layer"]}
        self.training["best_checkpoint"] = best_checkpoint_path

    def load_algorithm_default_hp(self, name: str) -> dict:
        """Load the algorithm with the best hyperparameters."""
        rel_path = "algos/hyperparams/common/{}.yaml".format(name)

        with open(os.path.join(os.path.dirname(__file__), rel_path), "r") as f:
            algo_config_dict = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        return algo_config_dict["algo_args"]

    def run_analyzing_activity(self):
        """Run the analyzing activity."""
        print("Running analyzing activity...")

        best_algo = list(self.training.hyperparameters["algorithms"].keys())[0]
        best_hp = self.training.hyperparameters["algorithms"][best_algo]

        env = marl.make_env(
            environment_name="simulated_environment", map_name="default", force_coop=False, organizational_model=self.modelling.organizational_specifications)

        mappo = marl.algos.getattr()(
            hyperparam_source="common", **best_hp["algorithm"])

        model = marl.build_model(
            env, mappo, best_hp["model"])

        # TODO: Add HPO for TEMM
        mappo.render(env, model,
                     restore_path={
                         'params_path': os.path.join(self.training["best_checkpoint"], "../params.json"),
                         'model_path': self.training["best_checkpoint"],
                         'render': True if self.modelling.simulated_environment.rendering_function is (not None or "") else False,
                         #  'record_env': True,
                         'render_env': True
                     },
                     enable_temm=True,
                     local_mode=True,
                     share_policy="group",
                     stop_timesteps=1,
                     timesteps_total=1,
                     checkpoint_freq=10000,
                     stop_iters=1,
                     checkpoint_end=False)
