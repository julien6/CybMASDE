import math
import gym
import random
import numpy as np
from random import randint
from typing import Any, List
from marllib import marl
from mma_wrapper.label_manager import label_manager
from mma_wrapper.organizational_model import deontic_specification, organizational_model, structural_specifications, functional_specifications, deontic_specifications, time_constraint_type
from mma_wrapper.organizational_specification_logic import role_logic, goal_factory, role_factory, goal_logic
from mma_wrapper.utils import label, observation, action, trajectory
from marllib.envs.base_env.wmt import RLlibWMT
from collections import OrderedDict


def decompose_feature_vector(feature_vector, num_agents=2, num_pots=2):
    """
    Décompose un vecteur d'observation en un dictionnaire de sous-vecteurs associés aux noms des features,
    tout en conservant l'ordre d'apparition des features.

    Arguments :
        feature_vector (np.array) : Vecteur d'observation produit par OvercookedAI.
        num_agents (int) : Nombre d'agents dans l'environnement (par défaut 2).
        num_pots (int) : Nombre de pots pris en compte (par défaut 2).

    Retourne :
        OrderedDict : Dictionnaire ordonné contenant les sous-vecteurs associés aux noms des features.
    """
    index = 0
    features = OrderedDict()

    # Orientation (4 valeurs one-hot)
    features["orientation"] = feature_vector[index:index+4]
    index += 4

    # Objet tenu (4 valeurs one-hot)
    features["held_object"] = feature_vector[index:index+4]
    index += 4

    # Distances aux objets clés (6 paires dx, dy)
    obj_names = ["onion", "tomato", "dish", "soup", "serving", "empty_counter"]
    for obj in obj_names:
        features[f"dist_{obj}"] = feature_vector[index:index+2]
        index += 2

    # Nombre d'ingrédients dans la soupe la plus proche
    features["soup_num_onions"] = feature_vector[index]
    index += 1
    features["soup_num_tomatoes"] = feature_vector[index]
    index += 1

    # Caractéristiques des casseroles proches
    for pot_idx in range(num_pots):
        features[f"pot_{pot_idx}_exists"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_is_empty"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_is_full"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_is_cooking"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_is_ready"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_num_onions"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_num_tomatoes"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_cook_time"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_dist"] = feature_vector[index:index+2]
        index += 2

    # Murs adjacents (4 valeurs booléennes)
    for direction in range(4):
        features[f"wall_{direction}"] = feature_vector[index]
        index += 1

    # Caractéristiques des autres joueurs
    other_players_features_length = (num_agents - 1) * (num_pots * 10 + 26)
    features["other_player_features"] = feature_vector[index:index +
                                                       other_players_features_length]
    index += other_players_features_length

    # Distances relatives aux autres joueurs
    features["relative_distances_to_others"] = feature_vector[index:index +
                                                              2*(num_agents-1)]
    index += 2*(num_agents-1)

    # Position absolue
    features["absolute_position"] = feature_vector[index:index+2]
    index += 2

    return dict((k, v) for k, v in features.items())


class overcooked_label_manager(label_manager):

    def __init__(self, action_space: gym.Space = None, observation_space: gym.Space = None, view_scope=None):
        super().__init__(action_space, observation_space)

        self.view_scope = view_scope

        self.action_encode = {
            "nothing": 4,
            "up": 0,
            "down": 1,
            "left": 3,
            "right": 2,
            "interact": 5
        }

        self.action_decode = {v: k for k, v in self.action_encode.items()}

    def one_hot_decode_observation(self, observation: observation, agent: str = None) -> Any:
        return decompose_feature_vector(observation)

    def one_hot_encode_action(self, action: Any, agent: str = None) -> action:
        return self.action_encode[action]

    def one_hot_decode_action(self, action: action, agent: str = None) -> Any:
        return self.action_decode[action]


def go_towards(dx, dy, facing_walls):
    actions = {}
    if abs(dx) >= abs(dy):
        if dx < 0:
            actions["left"] = False
            if facing_walls[3] == 0:
                actions["left"] = True
        if dx > 0:
            actions["right"] = False
            if facing_walls[2] == 0:
                actions["right"] = True
    if abs(dx) <= abs(dy):
        if dy < 0:
            actions["up"] = False
            if facing_walls[0] == 0:
                actions["up"] = True
        if dy > 0:
            actions["down"] = False
            if facing_walls[1] == 0:
                actions["down"] = True
    if len(actions.keys()) == 1 and list(actions.values())[0] == False:
        if (abs(dx) == 1 and abs(dy) == 0) or (abs(dx) == 0 and abs(dy) == 1):
            return list(actions.keys())[0]
        if (abs(dx) > 1 or abs(dy) > 1):
            directions = ["up", "down", "left", "right"]
            free_dir = [directions[index] for index, blocked_wall in enumerate(
                facing_walls) if blocked_wall == 0]
            return random.choice(free_dir)
    else:
        for action, free in actions.items():
            if free:
                return action


ori = {
    "up": np.asarray([1, 0, 0, 0]),
    "down": np.asarray([0, 1, 0, 0]),
    "right": np.asarray([0, 0, 1, 0]),
    "left": np.asarray([0, 0, 0, 1])
}


def go_and_interact(dist, orientation, facing_walls):
    if abs(dist[0]) > 0 or abs(dist[1]) > 0:
        towards_item = go_towards(dist[0], dist[1], facing_walls)
        if ((abs(dist[0]) == 0 and abs(dist[1]) == 1) or (abs(dist[0]) == 1 and abs(dist[1]) == 0)) and (ori[towards_item] == orientation).all():
            return "interact"
        return towards_item
    return None


def is_next_and_facing_to(dist, orientation):
    towards_item = go_towards(dist[0], dist[1])
    return ((abs(dist[0]) == 0 and abs(dist[1]) == 1) or (abs(dist[0]) == 1 and abs(dist[1]) == 0)) and (ori[towards_item] == orientation).all()


def primary_fun(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:
    data = label_manager.one_hot_decode_observation(
        observation=observation, agent=agent_name)

    # |-----------> x
    # |
    # |
    # |
    # V
    # y
    #
    #
    # [
    #  X   Onion
    #  X   Soup
    #  X   Dish
    #  X   Tomato
    # ]

    orientation = data["orientation"]
    held_object = np.array(data["held_object"])
    dist_onion = data["dist_onion"]
    dist_soup = data["dist_soup"]
    dist_dish = data["dist_dish"]
    dist_serving = data["dist_serving"]
    facing_walls = [data["wall_0"], data["wall_1"],
                    data["wall_2"], data["wall_3"]]
    dist_empty_counter = data["dist_empty_counter"]

    pot_0_is_empty = data[f"pot_0_is_empty"]
    pot_0_is_full = data[f"pot_0_is_full"]
    pot_0_is_cooking = data[f"pot_0_is_cooking"]
    pot_0_is_ready = data[f"pot_0_is_ready"]
    pot_0_num_onions = data[f"pot_0_num_onions"]
    pot_0_cook_time = data[f"pot_0_cook_time"]
    pot_0_dist = data[f"pot_0_dist"]

    pot_1_is_empty = data[f"pot_1_is_empty"]
    pot_1_is_full = data[f"pot_1_is_full"]
    pot_1_is_cooking = data[f"pot_1_is_cooking"]
    pot_1_is_ready = data[f"pot_1_is_ready"]
    pot_1_num_onions = data[f"pot_1_num_onions"]
    pot_1_cook_time = data[f"pot_1_cook_time"]
    pot_1_dist = data[f"pot_1_dist"]

    # If it has a soup then bring it to the serving zone
    if held_object[1] == 1:
        action = go_and_interact(dist_empty_counter, orientation, facing_walls)
        if action is not None:
            return label_manager.one_hot_encode_action(action)

    # If the agent has a dish plate and one of the pots is ready, then go to the concerned pot and take a soup
    if held_object[2] == 1:
        if pot_0_is_ready == 1:
            action = go_and_interact(pot_0_dist, orientation, facing_walls)
            if action is not None:
                return label_manager.one_hot_encode_action(action)
        if pot_1_is_ready == 1:
            action = go_and_interact(pot_1_dist, orientation, facing_walls)
            if action is not None:
                return label_manager.one_hot_encode_action(action)

    # If any pot is cooking or has finished cooking and the agent has nothing in hands, then take a dish plate
    if ((pot_1_is_ready == 1 or pot_1_is_cooking == 1) and (pot_0_is_ready == 1 or pot_0_is_cooking == 1)) and np.all(held_object == 0):
        action = go_and_interact(dist_dish, orientation, facing_walls)
        if action is not None:
            return label_manager.one_hot_encode_action(action)
    # If pot 0 is full and non-cooking and if agent has nothing in hands, then go and interact to launch the cooking
    if (pot_0_is_full == 1 and pot_0_is_cooking == 0 and pot_0_is_ready == 0) and np.all(held_object == 0):
        action = go_and_interact(pot_0_dist, orientation, facing_walls)
        if action is not None:
            return label_manager.one_hot_encode_action(action)
    # If pot 1 is full and non-cooking and if agent has nothing in hands, then go and interact to launch the cooking
    if (pot_1_is_full == 1 and pot_1_is_cooking == 0 and pot_1_is_ready == 0) and np.all(held_object == 0):
        action = go_and_interact(pot_1_dist, orientation, facing_walls)
        if action is not None:
            return label_manager.one_hot_encode_action(action)

    # If there is a non full pot, then take an onion and bring it to a pot
    if ((pot_0_is_empty == 1 or pot_0_is_full == 0) and pot_0_is_cooking == 0 and pot_0_is_ready == 0) or ((pot_1_is_empty == 1 or pot_1_is_full == 0) and pot_1_is_cooking == 0 and pot_1_is_ready == 0):
        if held_object[0] == 0 and held_object[2] == 0 and held_object[1] == 0:
            action = go_and_interact(dist_onion, orientation, facing_walls)
            if action is not None:
                return label_manager.one_hot_encode_action(action)
        if held_object[0] == 1 and held_object[2] == 0 and held_object[1] == 0:
            if ((pot_0_is_empty == 1 or pot_0_is_full == 0) and pot_0_is_cooking == 0 and pot_0_is_ready == 0):
                action = go_and_interact(pot_0_dist, orientation, facing_walls)
                if action is not None:
                    return label_manager.one_hot_encode_action(action)
            if ((pot_1_is_empty == 1 or pot_1_is_full == 0) and pot_1_is_cooking == 0 and pot_1_is_ready == 0):
                action = go_and_interact(pot_1_dist, orientation, facing_walls)
                if action is not None:
                    return label_manager.one_hot_encode_action(action)
    return label_manager.one_hot_encode_action("nothing")


oa_label_mngr = overcooked_label_manager()

ao_model = organizational_model(
    structural_specifications(
        roles={
            "role_primary": role_logic(label_manager=oa_label_mngr).registrer_script_rule(primary_fun),
            "role_secondary": role_logic(label_manager=oa_label_mngr).registrer_script_rule(primary_fun)},
        role_inheritance_relations={}, root_groups={}),
    functional_specifications=functional_specifications(
        goals={}, social_scheme={}, mission_preferences=[]),
    deontic_specifications=deontic_specifications(permissions=[], obligations=[
        deontic_specification(
            "role_primary", ["agent_0"], [], time_constraint_type.ANY),
        deontic_specification(
            "role_secondary", ["agent_1"], [], time_constraint_type.ANY)
    ]))


# prepare env
env = marl.make_env(environment_name="overcooked",
                    map_name="asymmetric_advantages", organizational_model=ao_model)

# initialize algorithm with appointed hyper-parameters
# mappo = marl.algos.mappo(hyperparam_source="mpe")
mappo = marl.algos.mappo(hyperparam_source="test")

# build agent model based on env + algorithms + user preference
model = marl.build_model(
    env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# start training
# mappo.fit(env, model, stop={'episode_reward_mean': 6000, 'timesteps_total': 20000000}, local_mode=False, num_gpus=0, num_gpus_per_worker=0,
#           num_workers=1, share_policy='group', checkpoint_freq=20)

# rendering
mappo.render(env, model,
             restore_path={
                 'params_path': "./exp_results/mappo_mlp_asymmetric_advantages_copy/MAPPOTrainer_overcooked_asymmetric_advantages_08cc3_00000_0_2025-03-19_10-25-01/params.json",
                 'model_path': "./exp_results/mappo_mlp_asymmetric_advantages_copy/MAPPOTrainer_overcooked_asymmetric_advantages_08cc3_00000_0_2025-03-19_10-25-01/checkpoint_000020/checkpoint-20",
                #  'render': True,
                 #  'record_env': True,
                 'render_env': True
             },
             enable_temm = True,
             local_mode=True,
             share_policy="group",
             stop_timesteps=1,
             timesteps_total=1,
             checkpoint_freq=1,
             stop_iters=1,
             checkpoint_end=True)
