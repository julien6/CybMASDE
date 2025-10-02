import importlib
import os
import json
import signal
import sys
import threading
import time
import numpy as np
import logging
import torch
import optuna
import yaml
import shutil

from typing import List
from mma_wrapper.label_manager import label_manager
from copy import copy, deepcopy
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from pettingzoo.utils.env import ParallelEnv
from world_model.rdlm_utils import rdlm_objective, RDLM
from world_model.vae_utils import VAE, objective
from world_model.component_functions import ComponentFunctions
from world_model.project_configuration import JOPM, RDLM_conf, Configuration
from multiprocessing import Process
from marllib import marl
from ray.tune.stopper import Stopper
from ray import tune
from mma_wrapper.organizational_model import organizational_model
from distutils.dir_util import copy_tree
from mma_wrapper.label_manager import label_manager
from mma_wrapper.organizational_model import deontic_specification, organizational_model, structural_specifications, functional_specifications, deontic_specifications, time_constraint_type
from mma_wrapper.organizational_specification_logic import role_logic, goal_factory, role_factory, goal_logic
from mma_wrapper.utils import label, observation, action, trajectory


class MeanStdStopper(Stopper):
    def __init__(self, mean_threshold, std_threshold, window_size=100, max_timesteps_total=1e7):
        self.mean_threshold = mean_threshold
        self.std_threshold = std_threshold
        self.window_size = window_size
        self.episode_rewards = []
        self.max_timesteps_total = max_timesteps_total

    def __call__(self, trial_id, result):
        if result.get("timesteps_total", 0) > self.max_timesteps_total:
            print(
                f"Trial {trial_id} exceeded max timesteps {self.max_timesteps_total}. Stopping.")
            return True  # Stop this trial if it exceeds max timesteps
        rewards = result.get("hist_stats", {}).get("episode_reward", None)
        if rewards is not None:
            self.episode_rewards.extend(rewards)
            if len(self.episode_rewards) >= self.window_size:
                self.episode_rewards = self.episode_rewards[-self.window_size:]
                mean = np.mean(self.episode_rewards)
                std = np.std(self.episode_rewards)
                # print(f"Average over {len(self.episode_rewards)} episodes : {mean:.2f}")
                #  print(f"Standard deviation over {len(self.episode_rewards)} episodes : {std:.2f}")
                if mean >= self.mean_threshold and std <= self.std_threshold:
                    print(
                        f"Early stopping : average >= {self.mean_threshold} and standard deviation <= {self.std_threshold}")
                    return True  # Stop this trial
        return False  # Continue

    def stop_all(self):
        return False


if not os.path.exists(os.path.join(os.path.expanduser("~"), ".cybmasde", "configuration.json")):
    json.dump({}, open(os.path.join(os.path.expanduser(
        "~"), ".cybmasde", "configuration.json"), "w+"))

cybmasde_conf = json.load(
    open(os.path.join(os.path.expanduser("~"), ".cybmasde", "configuration.json")))

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MTAProcess(Process):

    def __init__(self, configuration: Configuration, componentFunctions: ComponentFunctions):
        super().__init__()
        self.configuration = configuration
        self.componentFunctions = componentFunctions

    def _signal_handler(self, signum, frame):
        print(f"Received signal {signum}, stopping MTAProcess...")
        sys.exit(0)

    def run(self):
        """Run the MTA process."""
        print("Running MTA process with configuration:", self.configuration)

        # Installer le handler SIGTERM
        signal.signal(signal.SIGTERM, self._signal_handler)
        # Optionnel, pour Ctrl-C aussi
        signal.signal(signal.SIGINT, self._signal_handler)

        # Run each activity of the MTA process
        # self.run_modelling_activity()

        # # Run refinement cycles
        # for i in range(self.configuration.max_refinement_cycle):
        #     print(
        #         f"Refinement cycle {i + 1}/{self.configuration.max_refinement_cycle}")
        # self.run_training_activity()
        self.run_analyzing_activity()
        print("Finished MTA process")

    def load_handcrafted_environment(self):
        """Load the handcrafted environment."""
        spec = importlib.util.spec_from_file_location(
            "CustomParallelEnv", os.path.join(
                self.configuration.common.project_path, self.configuration.modelling.environment_path))

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Recherche la classe CustomParallelEnv dans le module
        self.handcrafted_environment: ParallelEnv = module
        if hasattr(module, "CustomParallelEnv"):
            self.handcrafted_environment = getattr(
                module, "CustomParallelEnv")(label_manager=self.label_manager)
            return True
        else:
            return False

    def check_method_impemented(self, function: callable, arguments: List):
        try:
            function(*arguments)
        except NotImplementedError:
            print(f"Method {function.__name__} is not implemented.")
            return False
        except Exception as e:
            return True

    def run_modelling_activity(self):
        """Run the modelling activity."""
        print("Running modelling activity...")

        if not self.load_handcrafted_environment():
            print(
                "No ready-to-use simulated environment path provided, generating a new one with World Models...")

            # Check if traces and reward are provided
            if not os.path.exists(os.path.join(self.configuration.common.project_path, self.configuration.modelling.generated_environment.world_model.used_traces_path)):
                raise ValueError(
                    "Traces path is not provided to generate the world model.")

            if not self.check_method_impemented(self.componentFunctions.reward_fn, [None, None, None]):
                raise ValueError(
                    "Reward function is not provided to generate the simulated environment.")

            if not self.check_method_impemented(self.componentFunctions.done_fn, [None, None, None]):
                raise ValueError(
                    "Stop condition function is not provided to generate the simulated environment.")

            if not self.check_method_impemented(self.componentFunctions.render_fn, [None, None, None]):
                logger.warning(
                    "No ready-to-use rendering function provided, using raw rendering function...")

            ##############################################################
            # Generate the world model
            ##############################################################

            # Check if the world model hyperparameters are provided else use default ones
            if os.path.exists(os.path.join(self.configuration.common.project_path, self.configuration.modelling.generated_environment.world_model.jopm.autoencoder.hyperparameters)):
                self.autoencoder_hyperparameters = json.load(open(os.path.join(
                    self.configuration.common.project_path, self.configuration.modelling.generated_environment.world_model.jopm.autoencoder.hyperparameters)))
            else:
                print(
                    "No Autoencoder hyperparameters research space provided, using default ones...")
                self.autoencoder_hyperparameters = cybmasde_conf["autoencoder"]

            if os.path.exists(os.path.join(self.configuration.common.project_path, self.configuration.modelling.generated_environment.world_model.jopm.rdlm.hyperparameters)):
                self.rdlm_hyperparameters = json.load(open(os.path.join(
                    self.configuration.common.project_path, self.configuration.modelling.generated_environment.world_model.jopm.rdlm.hyperparameters)))
            else:
                print(
                    "No RDLM hyperparameters research space provided, using default ones...")
                self.rdlm_hyperparameters = cybmasde_conf["rdlm"]

            # if os.path.exists(os.path.join(self.configuration.common.project_path, self.configuration.modelling.generated_environment.world_model.jopm.initial_joint_observations)):
            #     self.initial_joint_observations = json.load(open(os.path.join(
            #         self.configuration.common.project_path, self.configuration.modelling.generated_environment.world_model.jopm.initial_joint_observations)))
            # else:
            #     raise Exception(
            #         "No initial joint observation provided...")

            # Run the world model training with hyperparameter optimization (HPO)
            print("Running world model hyperparameter optimization...")
            self.run_autoencoder_with_hpo()

            self.run_rdlm_with_hpo()

            # Assemble the reward function, world model (and optionally the rendering function) into the simulated environment
            print("Assembling the simulated environment...")
            self.assemble_simulated_environment()

    def assemble_simulated_environment(self):
        self.jopm = JOPM(self.autoencoder, self.rdlm,
                         self.initial_joint_observations)
        self.jopm.save(os.path.dirname(os.path.join(self.configuration.common.project_path,
                       self.configuration.modelling.generated_environment.world_model.jopm.initial_joint_observations)))

    def load_traces(self, traces_path):
        """
        Charge toutes les traces à partir des fichiers de trajectoires dans traces_path.
        Retourne une liste de traces conjointes
        """
        joint_observations = []
        joint_actions = []

        # for each episode
        for fname in sorted(os.listdir(traces_path)):
            joint_observations_ep = []
            joint_actions_ep = []
            if fname.startswith("joint_history_") and fname.endswith(".json"):
                with open(os.path.join(traces_path, fname), "r") as f:
                    history_data = json.load(f)
                    # for each time step
                    for (joint_observation, joint_action) in history_data:
                        joint_observations_ep.append(np.array(
                            joint_observation, dtype=np.float32))
                        joint_actions_ep.append(
                            np.array(joint_action, dtype=np.float32))
                    joint_observations.append(
                        np.array(joint_observations_ep, dtype=np.float32))
                    joint_actions.append(
                        np.array(joint_actions_ep, dtype=np.float32))
        joint_observations = np.array(joint_observations, dtype=np.float32)
        joint_actions = np.array(joint_actions, dtype=np.float32)

        return joint_observations, joint_actions

    def get_joint_observations(self, joint_histories):
        """
        Extrait les observations conjointes à partir des historiques de traces.
        """

        joint_observations = []

        for history in joint_histories:
            for (joint_observation, joint_action) in history:
                print("joint_observation:", joint_observation)
                joint_observations.append(joint_observation.flatten())

        print("Joint observations shape:",
              np.array(joint_observations).shape)
        return joint_observations

    def run_autoencoder_with_hpo(self):
        """Run the autoencoder training with hyperparameter optimization (HPO)."""
        print("Running autoencoder training with hyperparameter optimization...")

        hp_space = self.autoencoder_hyperparameters
        max_mse = self.configuration.modelling.generated_environment.world_model.jopm.autoencoder.max_mean_square_error
        traces_path = os.path.join(self.configuration.common.project_path,
                                   self.configuration.modelling.generated_environment.world_model.used_traces_path)

        # joint_observations.shape = (nb_episode, nb_time_step, nb_agents, obs_dim)
        # joint_actions.shape = (nb_episode, nb_time_step, nb_agents, action_dim)
        joint_observations, joint_actions = self.load_traces(traces_path)

        nb_episodes = joint_observations.shape[0]
        nb_steps = joint_observations.shape[1]
        nb_agents = joint_observations.shape[2]
        observation_vector_dim = joint_observations.shape[3]

        # Extract all joint observations at time_step = 0 for each episode
        self.initial_joint_observations = joint_observations[:, 0, :, :]

        # We reshape them to get 2D tensors
        joint_observations = joint_observations.reshape(
            joint_observations.shape[0] * joint_observations.shape[1], joint_observations.shape[2] * joint_observations.shape[3])
        joint_actions = joint_actions.reshape(
            joint_actions.shape[0] * joint_actions.shape[1], joint_actions.shape[2])

        input_dim = joint_observations.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def optuna_objective(trial):
            return objective(trial, joint_observations, input_dim, device, max_mse, hp_space)

        study = optuna.create_study(direction="minimize")
        # Ajuste n_trials selon tes ressources
        study.optimize(optuna_objective, n_trials=30)

        print("Best hyperparameters:", study.best_params)
        print("Best validation MSE:", study.best_value)

        # Recharge le meilleur modèle
        self.autoencoder = VAE(input_dim=input_dim, **{k: study.best_params[k] for k in [
            "latent_dim", "hidden_dim", "n_layers", "activation"]}).to(device)

        # Mets à jour les hyperparamètres
        json.dump({
            k: [study.best_params[k], study.best_params[k]] for k in ["latent_dim", "hidden_dim", "n_layers", "activation", "lr", "batch_size", "kl_weight"]
        }, open(os.path.join(self.configuration.common.project_path,  self.configuration.modelling.generated_environment.world_model.jopm.autoencoder.hyperparameters), "w+"))

        # 1. Passe le modèle en mode évaluation
        self.autoencoder.eval()

        # 2. Compresse les observations
        with torch.no_grad():
            obs_tensor = torch.tensor(
                joint_observations, dtype=torch.float32, device=device)
            # On suppose que autoencoder.encoder retourne le latent (sinon adapte selon ta classe VAE)
            # shape: (nb_episode, latent_dim)
            mu, logvar = self.autoencoder.encode(obs_tensor)
            compressed_joint_observations = mu.cpu().numpy()

        # 3. Crée le dossier compressed si besoin
        compressed_dir = os.path.join(self.configuration.common.project_path,
                                      self.configuration.modelling.generated_environment.world_model.used_traces_path, "compressed")
        os.makedirs(compressed_dir, exist_ok=True)

        # Trouver le dernier X existant
        existing_files = [f for f in os.listdir(compressed_dir) if f.startswith(
            "compressed_joint_history_") and f.endswith(".json")]
        if existing_files:
            ids = []
            for fname in existing_files:
                try:
                    num = int(fname.replace(
                        "compressed_joint_history_", "").replace(".json", ""))
                    ids.append(num)
                except ValueError:
                    continue
            last_id = max(ids) + 1 if ids else 0
        else:
            last_id = 0

        # Sauvegarder chaque historique compressé
        for ep in range(nb_episodes):
            compressed_history = []
            for t in range(nb_steps):
                # On stocke chaque transition (latent_obs, joint_action)
                compressed_history.append([
                    compressed_joint_observations[ep * nb_steps + t].tolist(),
                    joint_actions[ep * nb_steps + t].tolist()
                ])
            filename = f"compressed_joint_history_{last_id}.json"
            file_path = os.path.join(compressed_dir, filename)
            with open(file_path, "w") as f:
                json.dump(compressed_history, f)
            last_id += 1

    def run_rdlm_with_hpo(self):
        """Run the RLDM (RNN+MLP) training with hyperparameter optimization (HPO)."""
        print("Running RLDM training with hyperparameter optimization...")

        rdlm_hyperparameters = None

        # Charger les hyperparamètres RDLM
        if os.path.exists(os.path.join(self.configuration.common.project_path, self.configuration.modelling.generated_environment.world_model.jopm.rdlm.hyperparameters)):
            rdlm_hyperparameters = json.load(open(os.path.join(self.configuration.common.project_path,
                                             self.configuration.modelling.generated_environment.world_model.jopm.rdlm.hyperparameters)))
        else:
            print(
                "No RDLM hyperparameters research space provided, using default ones...")
            rdlm_hyperparameters = cybmasde_conf["rdlm"]

        max_mse = self.configuration.modelling.generated_environment.world_model.jopm.rdlm.max_mean_square_error

        # Charger les historiques compressés
        compressed_dir = os.path.join(
            self.configuration.common.project_path,
            self.configuration.modelling.generated_environment.world_model.used_traces_path,
            "compressed"
        )

        if not os.path.exists(compressed_dir):
            os.makedirs(compressed_dir, exist_ok=True)

        compressed_files = sorted([f for f in os.listdir(compressed_dir) if f.startswith(
            "compressed_joint_history_") and f.endswith(".json")])
        print(
            f"Found {len(compressed_files)} compressed trace files.", compressed_files)

        latent_obs_episodes = []
        actions_episodes = []

        for fname in compressed_files:
            with open(os.path.join(compressed_dir, fname), "r") as f:
                episode = json.load(f)
                latent_obs = []
                actions = []
                for latent_obs_t, action_t in episode:
                    latent_obs.append(latent_obs_t)
                    # One-hot encode actions
                    one_hot_actions = []
                    for i, a in enumerate(action_t):
                        # Suppose action_space_n est le nombre de valeurs possibles pour chaque agent
                        one_hot = [
                            0] * self.componentFunctions.label_manager.action_space.n
                        one_hot[int(a)] = 1
                        one_hot_actions.extend(one_hot)
                    actions.append(one_hot_actions)
                latent_obs_episodes.append(latent_obs)
                actions_episodes.append(actions)

        # (n_episodes, n_steps, latent_dim)
        latent_obs_episodes = np.array(latent_obs_episodes, dtype=np.float32)
        # (n_episodes, n_steps, action_space_n * n_agents)
        actions_episodes = np.array(actions_episodes, dtype=np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def optuna_objective(trial):
            return rdlm_objective(trial, latent_obs_episodes, actions_episodes, device, max_mse, rdlm_hyperparameters)

        study = optuna.create_study(direction="minimize")
        study.optimize(optuna_objective, n_trials=30)
        print("Best RDLM hyperparameters:", study.best_params)

        # Recharge le meilleur modèle
        self.rdlm = RDLM(latent_joint_observation_dim=latent_obs_episodes.shape[2],
                         onehot_joint_action_dim=actions_episodes.shape[2],
                         **{k: v for k, v in study.best_params.items() if k != "learning_rate"}).to(device)

        # Mets à jour les hyperparamètres
        json.dump({k: [v, v] for k, v in study.best_params.items()}, open(os.path.join(self.configuration.common.project_path,
                  self.configuration.modelling.generated_environment.world_model.jopm.rdlm.hyperparameters), "w+"))

    def run_training_activity(self):
        """Run the training activity."""
        print("Running training activity...")

        # Check if hyperparameters intervals are provided, else use default ones
        training_hyperparameters = json.load(
            open(self.configuration.training.hyperparameters, "r"))
        if not self.configuration.training.hyperparameters or training_hyperparameters == {}:
            print(
                "No training hyperparameters research space provided, using default ones...")
            training_hyperparameters = cybmasde_conf[
                "training_hyperparameters"]

        organizational_specifications = organizational_model.from_dict(json.load(open(os.path.join(
            self.configuration.training.organizational_specifications, "organizational_specifications.json"), "r")))

        best_result = float('-inf')
        best_algorithm = None
        best_checkpoint_path = None
        for algorithm in training_hyperparameters["algorithms"]:

            env = marl.make_env(
                environment_name="cybmasde",
                map_name="default",
                force_coop=False,
                organizational_model=organizational_specifications,
                jopm_path=os.path.join(self.configuration.common.project_path, os.path.dirname(
                    self.configuration.modelling.generated_environment.world_model.jopm.initial_joint_observations)),
                component_functions_path=os.path.join(
                    self.configuration.common.project_path, self.configuration.modelling.generated_environment.component_functions_path),
                label_manager_path=os.path.join(
                    self.configuration.common.project_path, self.configuration.common.label_manager)
            )

            algo = marl.algos.__getattribute__(algorithm)(
                hyperparam_source="common", **{k: tune.grid_search(v) if isinstance(v, list) else v for k, v in training_hyperparameters["algorithms"][algorithm]["algorithm"].items(
                ) if k in list(self.load_algorithm_default_hp(algorithm).keys())})

            model = marl.build_model(
                env, algo, {k: tune.grid_search(v) if isinstance(v, list) else v for k, v in training_hyperparameters["algorithms"][algorithm]["model"].items() if k in ["core_arch", "mixer_arch", "encode_layer"]})

            experiment_analysis: ExperimentAnalysis = algo.fit(
                env,
                model,
                stop=MeanStdStopper(mean_threshold=training_hyperparameters["mean_threshold"],
                                    std_threshold=training_hyperparameters[
                                        "std_threshold"],
                                    window_size=training_hyperparameters["window_size"],
                                    max_timesteps_total=training_hyperparameters["max_timesteps_total"]),
                local_mode=False,
                num_gpus=training_hyperparameters["num_gpus"],
                num_workers=training_hyperparameters["num_workers"],
                share_policy='all',
                checkpoint_freq=training_hyperparameters["checkpoint_freq"],
                checkpoint_end=True)

            best_config = experiment_analysis.get_best_config(
                metric="episode_reward_mean", mode="max")
            print(f"Best hyperparameters found with algorithm {algorithm}")
            print("\tmodel: ", {k: v for k, v in best_config["model"]["custom_model_config"]["model_arch_args"].items(
            ) if k in ["core_arch", "mixer_arch", "encode_layer"]})
            print("\talgorithm: ", {k: v for k, v in best_config.items() if k in list(
                self.load_algorithm_default_hp(algorithm).keys())})

            best_trial = experiment_analysis.get_best_trial(
                metric="episode_reward_mean", mode="max")
            result = best_trial.metric_analysis["episode_reward_mean"]["max"]
            if best_result is None or result > best_result:
                best_result = result
                best_algorithm = algorithm
                best_checkpoint_path = best_trial.checkpoint.value
            training_hyperparameters["algorithms"][algorithm] = {
                "algorithm": {k: v for k, v in best_config.items() if k in list(
                    self.load_algorithm_default_hp(algorithm).keys())},
                "model": {k: v for k, v in best_config["model"]["custom_model_config"]["model_arch_args"].items(
                ) if k in ["core_arch", "mixer_arch", "encode_layer"]}
            }

        best_hp = deepcopy(
            training_hyperparameters["algorithms"][best_algorithm])
        training_hyperparameters = {}
        training_hyperparameters["algorithms"] = {}
        training_hyperparameters["algorithms"][best_algorithm] = best_hp

        json.dump(training_hyperparameters, open(os.path.join(
            self.configuration.common.project_path, self.configuration.training.hyperparameters), "w"), indent=4)

        # save the model parameters
        json.dump(training_hyperparameters, open(os.path.join(
            self.configuration.common.project_path, self.configuration.training.joint_policy, "model_config.json"), "w"), indent=4)

        # Remove all folders except the one containing the best checkpoint
        checkpoint_parent_dir = os.path.dirname(
            os.path.dirname(best_checkpoint_path))
        checkpoint_dir = os.path.dirname(best_checkpoint_path)
        for item in os.listdir(checkpoint_parent_dir):
            item_path = os.path.join(checkpoint_parent_dir, item)
            # Only remove directories, and skip the checkpoint_dir
            if os.path.isdir(item_path) and item_path != checkpoint_dir:
                try:
                    shutil.rmtree(item_path)
                except Exception as e:
                    logger.warning(f"Could not remove {item_path}: {e}")

        checkpoint_path_src = os.path.dirname(
            os.path.dirname(best_checkpoint_path))
        checkpoint_path_dest = os.path.join(
            self.configuration.common.project_path, self.configuration.training.joint_policy, "model")
        try:
            shutil.rmtree(checkpoint_path_dest)
        except Exception as e:
            logger.warning(f"Could not remove {checkpoint_path_dest}: {e}")

        os.makedirs(checkpoint_path_dest, exist_ok=True)
        copy_tree(checkpoint_path_src, checkpoint_path_dest)
        print("Joint policy saved in ", checkpoint_path_dest)

        shutil.rmtree(os.path.dirname(os.path.dirname(checkpoint_parent_dir)))

    def load_algorithm_default_hp(self, name: str) -> dict:
        """Load the algorithm with the best hyperparameters."""
        rel_path = "../../MARLlib/marllib/marl/algos/hyperparams/common/{}.yaml".format(
            name)

        with open(os.path.join(os.path.dirname(__file__), rel_path), "r") as f:
            algo_config_dict = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        return algo_config_dict["algo_args"]

    def run_analyzing_activity(self):
        """Run the analyzing activity."""
        print("Running analyzing activity...")

        best_hp = json.load(open(os.path.join(
            self.configuration.common.project_path, self.configuration.training.joint_policy, "model_config.json"), "r"))

        algorithm = list(best_hp["algorithms"].keys())[0]

        organizational_specifications = organizational_model.from_dict(json.load(open(os.path.join(
            self.configuration.training.organizational_specifications, "organizational_specifications.json"), "r")))

        env = marl.make_env(
            environment_name="cybmasde",
            map_name="default",
            force_coop=False,
            organizational_model=organizational_specifications,
            jopm_path=os.path.join(self.configuration.common.project_path, os.path.dirname(
                self.configuration.modelling.generated_environment.world_model.jopm.initial_joint_observations)),
            component_functions_path=os.path.join(
                self.configuration.common.project_path, self.configuration.modelling.generated_environment.component_functions_path),
            label_manager_path=os.path.join(
                self.configuration.common.project_path, self.configuration.common.label_manager)
        )

        algo = marl.algos.__getattribute__(algorithm)(
            hyperparam_source="common", **best_hp["algorithms"][algorithm]["algorithm"])

        model = marl.build_model(
            env, algo, best_hp["algorithms"][algorithm]["model"])

        params_path = os.path.join(self.configuration.common.project_path,
                                   self.configuration.training.joint_policy, "model/params.json")

        checkpoint_name = [f for f in os.listdir(os.path.join(
            self.configuration.common.project_path, self.configuration.training.joint_policy, "model")) if f.startswith("checkpoint_")][0]

        model_path = os.path.join(self.configuration.common.project_path, self.configuration.training.joint_policy,
                                  "model", checkpoint_name, f"checkpoint-{int(checkpoint_name.split('_')[1])}")

        # TODO: Add HPO for TEMM
        algo.render(env, model,
                    restore_path={
                        'params_path': params_path,
                        'model_path': model_path,
                        # self.check_method_impemented(self.componentFunctions.render_fn, [None, None, None]),
                        'render': False,
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

        # Remove old analysis results if any
        shutil.rmtree(os.path.join(self.configuration.common.
                                   project_path, self.configuration.analyzing.post_training_trajectories_path), ignore_errors=True)
        shutil.rmtree(os.path.join(self.configuration.common.
                                   project_path, self.configuration.analyzing.figures_path), ignore_errors=True)
        shutil.rmtree(os.path.join(self.configuration.common.
                                   project_path, self.configuration.analyzing.post_training_trajectories_path), ignore_errors=True)

        # Copy the figures to the project folder
        copy_tree(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_results", "figures"), os.path.join(
            self.configuration.common.project_path, self.configuration.analyzing.figures_path))

        # Copy the trajectories to the project folder
        copy_tree(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_results", "trajectories"), os.path.join(self.configuration.common.
                                                                                                                             project_path, self.configuration.analyzing.post_training_trajectories_path))

        # Copy the inferred organizational specifications to the project folder
        copy_tree(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_results", "inferred_organizational_specifications"),
                  os.path.join(self.configuration.common.project_path, self.configuration.analyzing.inferred_organizational_specifications))

        inferred_roles_summary = organizational_model.from_dict(json.load(open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "analysis_results", "inferred_organizational_specifications", "inferred_roles_summary.json"), "r")))
        inferred_goals_summary = organizational_model.from_dict(json.load(open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "analysis_results", "inferred_organizational_specifications", "inferred_goals_summary.json"), "r")))

        # Delete the temporary analysis_results folder
        shutil.rmtree(os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "analysis_results"), ignore_errors=True)
        shutil.rmtree(os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "exp_results"), ignore_errors=True)

    # def generate_organizational_model(self, inferred_roles_summary, inferred_goals_summary):
    #     """Generate the organizational model from the inferred roles and goals summaries."""

    #     inferred_roles = {}
    #     for inferred_role_name in inferred_roles_summary:
    #         # print("Inferred role:", inferred_role_name, " with rules:", inferred_roles_summary[inferred_role_name]["rules"])
    #         if len(inferred_roles_summary[inferred_role_name]["rules"]) > 0:
    #             inferred_roles[inferred_role_name] = inferred_roles_summary[inferred_role_name]["rules"]

    #     inferred_goals = {}
    #     for inferred_goal_name in inferred_goals_summary:
    #         # print("Inferred goal:", inferred_goal_name, " with observations:", inferred_goals_summary[inferred_goal_name]["observations"])
    #         if len(inferred_goals_summary[inferred_goal_name]["observations"]) > 0:
    #             inferred_goals[inferred_goal_name] = inferred_goals_summary[inferred_goal_name]["rules"]

    #     label_mngr = self.componentFunctions.label_manager()

    #     def primary_fun(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:
    #         chosen_action =
    #         return

    #     organizational_model = organizational_model(
    #         structural_specifications(
    #             roles={
    #                 "role_primary": role_logic(label_manager=label_mngr).registrer_script_rule(primary_fun),
    #                 "role_secondary": role_logic(label_manager=label_mngr).registrer_script_rule(primary_fun)},
    #             role_inheritance_relations={}, root_groups={}),
    #         functional_specifications=functional_specifications(
    #             goals={}, social_scheme={}, mission_preferences=[]),
    #         deontic_specifications=deontic_specifications(permissions=[], obligations=[
    #             deontic_specification(
    #                 "role_primary", ["agent_0"], [], time_constraint_type.ANY),
    #             deontic_specification(
    #                 "role_secondary", ["agent_1"], [], time_constraint_type.ANY)
    #         ]))

    #     # Save the generated organizational model
    #     org_model_path = os.path.join(self.configuration.common.project_path,
    #                                   self.configuration.analyzing.inferred_organizational_specifications, "organizational_specifications.json")
    #     os.makedirs(os.path.dirname(org_model_path), exist_ok=True)
    #     json.dump(org_model.to_dict(), open(org_model_path, "w"), indent=4)
    #     print("Generated organizational model saved to:", org_model_path)

    #     return org_model


if __name__ == "__main__":

    def load_component_functions(configuration: Configuration, lbl_manager: label_manager) -> ComponentFunctions:
        spec = importlib.util.spec_from_file_location(
            "ComponentFunctions", os.path.join(
                configuration.common.project_path, configuration.modelling.generated_environment.component_functions_path))

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Recherche la classe ComponentFunctions dans le module
        component_functions: ComponentFunctions = module
        if hasattr(module, "ComponentFunctions"):
            component_functions = getattr(
                module, "ComponentFunctions")(label_manager=lbl_manager)
        else:
            raise ImportError(
                "No ComponentFunctions class found in ", os.path.join(
                    configuration.common.project_path, configuration.modelling.generated_environment.component_functions_path))
        return component_functions

    def load_label_manager(configuration: Configuration) -> label_manager:
        spec = importlib.util.spec_from_file_location(
            "label_manager", os.path.join(
                configuration.common.project_path, configuration.common.label_manager))

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Recherche la classe label_manager dans le module
        lbl_manager: label_manager = module
        if hasattr(module, "label_manager"):
            lbl_manager = getattr(
                module, "label_manager")()
        else:
            raise ImportError(
                "No label_manager class found in ", os.path.join(
                    configuration.common.project_path, configuration.common.label_manager))
        return lbl_manager

    project_configuration = configuration = Configuration.from_json(os.path.join(
        os.path.expanduser("~"), "Documents/new_test/project_configuration.json"))

    lbl_manager = load_label_manager(project_configuration)
    component_functions = load_component_functions(
        project_configuration, lbl_manager)
    mta_process = MTAProcess(
        configuration=project_configuration, componentFunctions=component_functions)
    mta_process.run()
