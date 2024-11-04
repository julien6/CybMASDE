import os
import subprocess
import sys
import importlib
import optuna
import math
import numpy as np
from marllib import marl

# from custom_envs.movingcompany.moving_company_v0 import parallel_env


def install_package(env_config):
    package_info = env_config.get("package_environment_installation", {})

    if "via_pip" in package_info:
        package_name = package_info["via_pip"]
        subprocess.run([sys.executable, "-m", "pip",
                       "install", package_name], check=True)

    if "via_script" in package_info:
        script_path = package_info["via_script"]
        if os.path.exists(script_path):
            subprocess.run(["bash", script_path], check=True)
        else:
            raise FileNotFoundError(
                f"Installation script '{script_path}' not found.")

    if "via_setuptools" in package_info:
        package_dir = package_info["via_setuptools"]
        if os.path.exists(os.path.join(package_dir, "setup.py")):
            subprocess.run([sys.executable, "-m", "pip",
                           "install", package_dir], check=True)
        else:
            raise FileNotFoundError(f"No 'setup.py' found in '{package_dir}'.")


def instantiate_environment(env_config):
    env_instantiation = env_config.get("environment_instantiation", {})
    module_name = env_instantiation.get("module_name")
    creation_function = env_instantiation.get("creation_function")
    creation_options = env_instantiation.get("creation_options", {})

    if not module_name or not creation_function:
        raise ValueError(
            "Both 'module_name' and 'creation_function' must be provided.")

    # Import the module dynamically
    module = importlib.import_module(module_name)
    env_creation_func = getattr(module, creation_function)

    # Instantiate the environment with the provided options
    env = env_creation_func(**creation_options)
    return env


def solve(env_config_json, algorithm_config):
    # Step 1: Install the package environment
    install_package(env_config_json)

    # Step 2: Instantiate the environment
    env = instantiate_environment(env_config_json)

    # Step 3: Process each algorithm in the configuration
    for algo_name, params in algorithm_config.items():
        use_hpo = params.get("use_HPO_else_custom", False)
        custom_params = params.get("custom_hyper_parameters", {})
        hpo_config = params.get("grid_search_HPO", {})

        if use_hpo and hpo_config:
            best_config = run_hpo(env, algo_name, hpo_config.get(
                "use_empirical_rules_for_intervals", False), hpo_config.get("custom_intervals", {}))
            custom_params = best_config  # Replace custom_params with the best configuration found

        # # Always train with custom_params at the end
        # train_algorithm(env, algo_name, custom_params)

    print("Training completed.")


def flatten_space(space):
    """ Retourne la taille aplatie de l'espace d'observation """
    return int(np.prod(space.shape))


def calculate_empirical_intervals(observation_size, action_size):
    # Définir les intervalles en fonction des règles empiriques
    intervals = {
        "hidden_layer_size": (observation_size * 0.25, observation_size * 0.75),
        "latent_size": (observation_size * 0.01, observation_size * 0.05),
        "learning_rate": (1e-5, 1e-2),
        "num_layers": (2, 5),
        "activation_function": ["ReLU", "LeakyReLU", "ELU"]
    }
    return intervals


def get_algorithm_hyperparameters(algo_name):
    """ Retourne la liste des hyperparamètres spécifiques pour un algorithme donné """
    algo_hyperparams = {
        "Multi-Agent Q-Learning": ["learning_rate", "gamma", "epsilon", "buffer_size"],
        "MADDPG": ["actor_lr", "critic_lr", "gamma", "tau", "batch_size"],
        "REINFORCE": ["learning_rate", "batch_size"],
        "MAPPO": ["learning_rate", "clip_range", "num_mini_batches", "entropy_coeff", "gae_lambda"],
        "A3C": ["learning_rate", "entropy_coeff", "value_loss_coeff", "max_grad_norm"],
        "IQL": ["learning_rate", "gamma", "epsilon", "buffer_size"],
        "COMA": ["learning_rate", "critic_lr", "batch_size", "gamma", "tau"],
        "QMIX": ["learning_rate", "mixing_embed_dim", "batch_size", "gamma"],
        "VDN": ["learning_rate", "batch_size", "gamma"]
    }
    return algo_hyperparams.get(algo_name, [])


def run_hpo(env, algo_name, use_empirical_rules, custom_intervals):
    # Définir la taille aplatie des observations et des actions
    observation_size = flatten_space(
        env.observation_space(env.possible_agents[0]))
    action_size = flatten_space(env.action_space(env.possible_agents[0]))
    # print(f"Observation size: {observation_size}, Action size: {action_size}")

    if use_empirical_rules:
        hyperparam_intervals = calculate_empirical_intervals(
            observation_size, action_size)
    else:
        hyperparam_intervals = custom_intervals

    # Obtenir les hyperparamètres spécifiques pour l'algorithme choisi
    algo_hyperparams = get_algorithm_hyperparameters(algo_name)

    def objective(trial):
        # Définir les hyperparamètres à tester en fonction de l'algorithme
        trial_params = {}
        for param in algo_hyperparams:
            if param == "learning_rate":
                trial_params[param] = trial.suggest_loguniform("learning_rate",
                                                               hyperparam_intervals["learning_rate"][0],
                                                               hyperparam_intervals["learning_rate"][1])
            elif param == "hidden_layer_size" and "hidden_layer_size" in hyperparam_intervals:
                trial_params[param] = trial.suggest_int("hidden_layer_size",
                                                        int(
                                                            hyperparam_intervals["hidden_layer_size"][0]),
                                                        int(hyperparam_intervals["hidden_layer_size"][1]))
            elif param == "latent_size" and "latent_size" in hyperparam_intervals:
                trial_params[param] = trial.suggest_int("latent_size",
                                                        int(
                                                            hyperparam_intervals["latent_size"][0]),
                                                        int(hyperparam_intervals["latent_size"][1]))
            elif param == "num_layers" and "num_layers" in hyperparam_intervals:
                trial_params[param] = trial.suggest_int("num_layers",
                                                        int(
                                                            hyperparam_intervals["num_layers"][0]),
                                                        int(hyperparam_intervals["num_layers"][1]))
            elif param == "activation_function" and "activation_function" in hyperparam_intervals:
                trial_params[param] = trial.suggest_categorical("activation_function",
                                                                hyperparam_intervals["activation_function"])

        # Simuler l'entraînement et retourner une métrique (exemple : reward moyen)
        performance = simulate_training(env, algo_name, **trial_params)

        return performance

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    return study.best_params


def simulate_training(env, algo_name, **trial_params):
    # Initialisation de l'agent avec la bibliothèque MARLlib
    # model = marl.algos.get_model(algo_name, env, **trial_params)
    # agent = model.build_agent()

    print("==> ", marl.algos.mappo)

    # Simulation de l'entraînement
    num_episodes = 10  # Peut être ajusté selon les besoins
    total_reward = 0
    # for episode in range(num_episodes):
    #     obs = env.reset()
    #     done = {agent: False for agent in env.possible_agents}
    #     while not all(done.values()):
    #         actions = {agent: agent.compute_action(obs[agent]) for agent in env.possible_agents if not done[agent]}
    #         obs, rewards, done, _ = env.step(actions)
    #         total_reward += sum(rewards.values())

    # Retourner la récompense moyenne sur les épisodes
    return total_reward / num_episodes


if __name__ == "__main__":

    # env_config = {
    #     "package_environment_installation": {
    #         "via_pip": "pettingzoo"
    #     },
    #     "environment_instantiation": {
    #         "module_name": "pettingzoo.butterfly.cooperative_pong_v5",
    #         "creation_function": "env",
    #         "creation_options": {
    #             "ball_speed": 9,
    #             "left_paddle_speed": 12,
    #             "right_paddle_speed": 12
    #         }
    #     }
    # }

    env_config = {
        "package_environment_installation": {
            "via_setuptools": "/home/julien/Documents/Thèse/CybMASDE/backend/src/modeling/generated_envs/custom_envs"
        },
        "environment_instantiation": {
            "module_name": "custom_envs.movingcompany.moving_company_v0",
            "creation_function": "parallel_env",
            "creation_options": {
                "size": 6
            }
        }
    }

    algorithm_config = {
        "PPO": {
            "custom_hyper_parameters": {
                "batch_size": 64,
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "clip_range": 0.2
            },
            "grid_search_HPO": {
                "use_empirical_rules_for_intervals": True,
                "custom_intervals": {
                    "learning_rate": [0.0001, 0.01],
                    "batch_size": [32, 128]
                }
            },
            "use_HPO_else_custom": True  # HPO sera lancé en priorité si présent
        },
        "DQN": {
            "custom_hyper_parameters": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "buffer_size": 100000,
                "epsilon_decay": 0.995
            }
            # Pas de HPO ici, donc seulement les hyper-paramètres personnalisés seront utilisés
        }
    }

    # Appel de la fonction solve avec les configurations JSON
    solve(env_config, algorithm_config)
    # parallel_env(size=6)
