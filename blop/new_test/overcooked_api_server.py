# Endpoint pour obtenir la dernière observation
import gym
import requests
import numpy as np
import json
from flask import Flask, request, jsonify
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, OvercookedState):
            return obj.to_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        return super().default(obj)


app = Flask(__name__)
app.json_encoder = NumpyEncoder

layout_mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
core_env = OvercookedEnv.from_mdp(layout_mdp, horizon=20)
config_dict = {'base_env': core_env,
               'featurize_fn': core_env.featurize_state_mdp}

env = gym.make('Overcooked-v0', **config_dict)
last_observation = None
last_action = None


@app.route('/agents', methods=['GET'])
def get_agents():
    global last_observation
    if last_observation is None:
        global env
        obs = env.reset()["both_agent_obs"]
        obs = json.loads(json.dumps(obs, indent=4, cls=NumpyEncoder))
        last_observation = obs
    return jsonify({'agents': list(range(0, len(last_observation)))})


@app.route('/reset', methods=['GET'])
def reset_env():
    obs = env.reset()["both_agent_obs"]
    obs = json.loads(json.dumps(obs, indent=4, cls=NumpyEncoder))
    global last_observation
    last_observation = obs
    return jsonify({'observation': obs})


@app.route('/last_observation', methods=['GET'])
def get_last_observation():
    global last_observation
    if last_observation is None or last_observation == {}:
        obs = env.reset()["both_agent_obs"]
        obs = json.loads(json.dumps(obs, indent=4, cls=NumpyEncoder))
        last_observation = obs
        return jsonify({'last_observation': obs})
    return jsonify({'last_observation': last_observation})


@app.route('/last_action', methods=['GET'])
def get_last_action():
    global last_action
    if last_action is None:
        return jsonify({'error': 'Aucune action disponible'}), 404
    return jsonify({'last_action': last_action})


@app.route('/step', methods=['POST'])
def step_env():
    actions = request.get_json()
    if actions is None:
        return jsonify({'error': 'actions field is required'}), 400
    try:
        obs, reward, done, info = env.step(actions)
        obs = json.loads(json.dumps(obs, indent=4, cls=NumpyEncoder))[
            "both_agent_obs"]
        reward = json.loads(json.dumps(reward, indent=4, cls=NumpyEncoder))
        done = json.loads(json.dumps(done, indent=4, cls=NumpyEncoder))
        info = json.loads(json.dumps(info, indent=4, cls=NumpyEncoder))
    except AssertionError as e:
        obs = {}

    global last_observation
    last_observation = obs
    global last_action
    last_action = actions

    return jsonify(obs)


@app.route('/close', methods=['GET'])
def close_env():
    env.close()
    return jsonify({'status': 'closed'})


@app.route('/action_space', methods=['GET'])
def get_action_space():
    try:
        action_space = env.action_space
        if hasattr(action_space, 'to_json'):
            data = action_space.to_json()
        elif hasattr(action_space, 'to_dict'):
            data = action_space.to_dict()
        else:
            data = str(action_space)
        return jsonify({'action_space': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/observation_space', methods=['GET'])
def get_observation_space():
    try:
        observation_space = env.observation_space
        if hasattr(observation_space, 'to_json'):
            data = observation_space.to_json()
        elif hasattr(observation_space, 'to_dict'):
            data = observation_space.to_dict()
        else:
            data = str(observation_space)
        return jsonify({'observation_space': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5030)
