const default_config = {
  "common": {
    "project_name": "new_test",
    "project_description": "Project description",
    "label_manager": "label_manager.py",
    "project_path": "/home/soulej/Documents/new_test"
  },
  "modelling": {
    "environment_path": "modelling/handcrafted_environment.py",
    "generated_environment": {
      "world_model": {
        "jopm": {
          "autoencoder": {
            "statistics": {},
            "model": "modelling/generated_environment/world_model/jopm/autoencoder/model",
            "hyperparameters": {
              "latent_dim": [
                32,
                32
              ],
              "hidden_dim": [
                256,
                256
              ],
              "n_layers": [
                2,
                2
              ],
              "activation": [
                "elu",
                "elu"
              ],
              "lr": [
                0.0014153166657866285,
                0.0014153166657866285
              ],
              "batch_size": [
                128,
                128
              ],
              "kl_weight": [
                0.2765579976596325,
                0.2765579976596325
              ]
            },
            "max_mean_square_error": "inf"
          },
          "rdlm": {
            "statistics": {},
            "model": "modelling/generated_environment/world_model/jopm/rdlm/model",
            "hyperparameters": {
              "rchs_dim": [
                50,
                50
              ],
              "rnn_type": [
                "GRU",
                "GRU"
              ],
              "rnn_hidden_dim": [
                152,
                152
              ],
              "rnn_layers": [
                2,
                2
              ],
              "rnn_activation": [
                "None",
                "None"
              ],
              "learning_rate": [
                0.002881626168985456,
                0.002881626168985456
              ],
              "mlp_layers": [
                3,
                3
              ],
              "mlp_hidden_dim": [
                149,
                149
              ],
              "mlp_activation": [
                "elu",
                "elu"
              ]
            },
            "max_mean_square_error": "inf"
          },
          "initial_joint_observations": "modelling/generated_environment/world_model/jopm/initial_joint_observations.pth"
        },
        "statistics": {},
        "used_traces_path": "modelling/generated_environment/world_model/traces"
      },
      "component_functions_path": "modelling/generated_environment/component_functions.py"
    }
  },
  "training": {
    "hyperparameters": {
      "algorithms": {
        "mappo": {
          "algorithm": {
            "batch_mode": "truncate_episodes",
            "lr": 0.0005,
            "entropy_coeff": 0.01,
            "num_sgd_iter": 5,
            "clip_param": 0.3,
            "use_gae": true,
            "lambda": 1,
            "vf_loss_coeff": 1,
            "kl_coeff": 0.2,
            "vf_clip_param": 10
          },
          "model": {
            "core_arch": "mlp",
            "encode_layer": "64-128"
          }
        }
      },
      "mean_threshold": 200,
      "max_timesteps_total": 2000,
      "std_threshold": 50,
      "window_size": 100,
      "num_gpus": 0,
      "num_workers": 1,
      "checkpoint_freq": 10
    },
    "organizational_specifications": "training/organizational_specifications",
    "joint_policy": "training/joint_policy",
    "statistics": {}
  },
  "analyzing": {
    "hyperparameters": {},
    "statistics": {},
    "figures_path": "analyzing/figures",
    "post_training_trajectories_path": "analyzing/trajectories",
    "inferred_organizational_specifications": "analyzing/inferred_organizational_specifications"
  },
  "transferring": {
    "configuration": {
      "trajectory_retrieve_frequency": 4,
      "trajectory_batch_size": 8,
      "deploy_mode": "REMOTE",
      "environment_api": "transferring/environment_api.py",
      "max_nb_iteration": 100
    }
  },
  "refining": {
    "max_refinement_cycles": 2,
    "auto_continue_refinement": false
  }
};

export { default_config };
