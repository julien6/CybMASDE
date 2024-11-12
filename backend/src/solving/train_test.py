from marllib import marl

# # prepare the environment academy_pass_and_shoot_with_keeper
# #env = marl.make_env(environment_name="hanabi", map_name="Hanabi-Very-Small")
# env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)

# # initialize algorithm and load hyperparameters
# mappo = marl.algos.mappo(hyperparam_source="mpe")

# # build agent model based on env + algorithms + user preference if checked available
# model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# # start learning + extra experiment settings if needed. remember to check ray.yaml before use
# mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 20000000}, local_mode=False, num_gpus=0,
#           num_workers=1, share_policy='all', checkpoint_freq=500)


import sys
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from agents_opc import leader_opc, normal_opc, good_opc
from mm_env import make_env
from mm_wrapper.osr_model import deontic_specifications, time_constraint_type, obligation, organizational_model, structural_specifications
from datetime import datetime
from pathlib import Path
from ray.tune import Analysis
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from mcy import RLlibMCY, RLlibMCY_FCOOP
from ray import tune
from btf import RLlibBTF, RLlibBTF_FCOOP

# osr = organizational_model(
#     structural_specifications(
#         {"r_leader": leader_opc, "r_normal": normal_opc, "r_good": good_opc}, None, None),
#     None,
#     deontic_specifications(None, {
#         obligation("r_leader", None, time_constraint_type.ANY): ["leadadversary_0"],
#         obligation("r_normal", None, time_constraint_type.ANY): ["adversary_0", "adversary_1", "adversary_2"],
#         obligation("r_good", None, time_constraint_type.ANY): ["agent_0", "agent_1"]}))

# Adding the BTF environment to the registries
try:
    ENV_REGISTRY["btf"] = RLlibBTF
except Exception as e:
    ENV_REGISTRY["btf"] = str(e)

try:
    COOP_ENV_REGISTRY["btf"] = RLlibBTF_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["btf"] = str(e)

env_conf = {
    "env": "btf",
    "env_args": {
        "map_name": "pistonball"
    },
    "mask_flag": False,
    "global_state_flag": False,
    "opp_action_in_cc": True
}

# Adding the MCY environment to the registries
# try:
#     ENV_REGISTRY["mcy"] = RLlibMCY
# except Exception as e:
#     ENV_REGISTRY["mcy"] = str(e)

# try:
#     COOP_ENV_REGISTRY["mcy"] = RLlibMCY_FCOOP
# except Exception as e:
#     COOP_ENV_REGISTRY["mcy"] = str(e)

# env_conf = {
#     "env": "mcy",
#     "env_args": {
#         "map_name": "moving_company",
#         "size": 6,
#         "seed": 42,
#         "max_cycles": 30
#     },
#     "mask_flag": False,
#     "global_state_flag": False,
#     "opp_action_in_cc": True
# }

###################################
# # HPO
# env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)

# mappo = marl.algos.mappo(hyperparam_source="test", lr=tune.grid_search([0.0005, 0.001]))

# model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": tune.grid_search(["8-16", "16-32", "32-128-256"])})

# mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 20000000}, local_mode=False, num_gpus=0,
#           num_workers=1, share_policy='all', checkpoint_freq=500)

# # more examples on ray search spaces can be found at this link:
# # https://docs.ray.io/en/latest/tune/tutorials/tune-search-spaces.html
###################################

# env = make_env(environment_name="mcy",
#                map_name="moving_company", force_coop=False, organizational_model=None, env_config_dict=env_conf, render_mode=None)

env = make_env(environment_name="btf",
               map_name="pistonball", force_coop=True, organizational_model=None, env_config_dict=env_conf, render_mode=None)

# initialize algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source="test")


# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(
    env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})


checkpoint_freq = 10

if len(sys.argv) > 1 and sys.argv[1] == "--test":

    checkpoint_path = None

    mode = "max"
    metric = 'episode_reward_mean'

    algorithm = mappo.name
    map_name = env[1]["env_args"]["map_name"]
    arch = model[1]["model_arch_args"]["core_arch"]
    running_directory = '_'.join([algorithm, arch, map_name])
    running_directory = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "exp_results", running_directory))

    print(f"running_directory: {running_directory}")

    if (os.path.exists(running_directory)):

        # Trouver tous les fichiers de checkpoint dans le dossier
        checkpoint_trial_folders = [f for f in os.listdir(
            running_directory) if f.startswith(algorithm.upper())]

        # 2024-06-16_15-38-06
        date_format = '%Y-%m-%d_%H-%M-%S'

        checkpoint_trial_folders.sort(
            key=lambda f: datetime.strptime(str(f[-19:]), date_format))

        checkpoint_path = os.path.join(
            running_directory, checkpoint_trial_folders[-1])

    print(f"checkpoint_path: {checkpoint_path}")

    analysis = Analysis(
        checkpoint_path, default_metric=metric, default_mode=mode)
    df = analysis.dataframe()

    idx = df[metric].idxmax()

    training_iteration = df.iloc[idx].training_iteration

    best_logdir = df.iloc[idx].logdir

    best_checkpoint_dir = [p for p in Path(best_logdir).iterdir(
    ) if "checkpoint_" in p.name and (int(p.name.split("checkpoint_")[1]) <= training_iteration and training_iteration <= int(p.name.split("checkpoint_")[1]) + checkpoint_freq)][0]

    checkpoint_number = str(
        int(best_checkpoint_dir.name.split("checkpoint_")[1]))
    best_checkpoint_file_path = os.path.join(
        best_checkpoint_dir, f'checkpoint-{checkpoint_number}')

    print(f"best_checkpoint_file_path: {best_checkpoint_file_path}")

    # rendering
    mappo.render(env, model,
                 stop={'timesteps_total': 40},
                 restore_path={'params_path': f"{checkpoint_path}/params.json",  # experiment configuration
                               'model_path': best_checkpoint_file_path,  # checkpoint path
                               'render': True},  # render
                 local_mode=False,
                 num_gpus=0,
                 num_workers=1,
                 share_policy="all",
                 checkpoint_end=False)

else:

    # start learning + extra experiment settings if needed. remember to check ray.yaml before use
    # mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 20000000}, local_mode=False, num_gpus=1,
    #           num_workers=10, share_policy='group', checkpoint_freq=checkpoint_freq)

    mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 20000000}, local_mode=False, num_gpus=0,
              num_workers=1, share_policy='all', checkpoint_freq=checkpoint_freq)
