from flask import Flask, Response, request
from flask_cors import CORS
import os
import time
import json
from typing import Dict, List, Any
from simulation_models.cyborg.CyborgEnvironment import CyborgEnvironment
# from simulation_models.cyst.cyst_environment import CystEnvironment
# from simulation_models.CybMASDE._CybMASDE_environment import CybMASDEEnvironment
from simulation_models.SimulationModel import SimulationModel

app = Flask(__name__)
CORS(app)


# for simulation
env: SimulationModel = None
simulation_engine: str = None
number_of_episodes: int = None
number_of_iterations: int = None
iteration_pause_duration: int = None


@app.post('/set-simulation')
def set_simulation_scenario():
    simulation_engine = request.args.get("simulation_engine", "CybORG")
    number_of_episodes = int(request.args.get("number_of_episodes", 1))
    number_of_iterations = int(request.args.get("number_of_iterations", 1))
    iteration_pause_duration = int(
        request.args.get("iteration_pause_duration", 0))
    scenario_plan = {}
    try:
        scenario_plan = request.json
    except Exception as e:
        raise e

    if simulation_engine == "CYST":
        # env = CystEnvironment(scenario_plan)
        pass

    if simulation_engine == "MCAS":
        # env = CybMASDEEnvironment(scenario_plan)
        pass

    if simulation_engine == "CybORG":
        global env
        env = CyborgEnvironment(
            scenario_plan, max_episode=number_of_episodes, max_iteration=number_of_iterations)

    return Response(json.dumps({"logs": "Successfully set up simulation environment"}), status=200, mimetype='application/json')


@app.get("/simulation-next")
def next_iteration():
    """Update environment to next state (next iteration and/or episode)
    Returns None when max episode and max iteration is reached.
    """

    global env

    requested_info = []

    if request.args.get("episode_number", False):
        requested_info += ["episode_number"]
    if request.args.get("iteration_number", False):
        requested_info += ["iteration_number"]
    if request.args.get("observation_spaces", False):
        requested_info += ["observation_spaces"]
    if request.args.get("action_spaces", False):
        requested_info += ["action_spaces"]
    if request.args.get("agents_observations", False):
        requested_info += ["agents_observations"]
    if request.args.get("agents_actions", False):
        requested_info += ["agents_actions"]
    if request.args.get("agents_rewards", False):
        requested_info += ["agents_rewards"]
    if request.args.get("true_states", False):
        requested_info += ["true_states"]
    if request.args.get("network_graph", False):
        requested_info += ["network_graph"]
    if request.args.get("team_agent_mapping", False):
        requested_info += ["team_agent_mapping"]
    if request.args.get("pz_cyborg_actions", False):
        requested_info += ["pz_cyborg_actions"]
    if request.args.get("cyborg_actions", False):
        requested_info += ["cyborg_actions"]
    if request.args.get("cyborg_observations", False):
        requested_info += ["cyborg_observations"]
    if request.args.get("agents_cyborg_comms", False):
        requested_info += ["agents_cyborg_comms"]
    if request.args.get("agents_pz_comms", False):
        requested_info += ["agents_pz_comms"]

    res = env.next(requested_info)

    if res == None:
        return Response(json.dumps({"logs": "Max iteration reached"}), status=204, mimetype='application/json')
    else:
        return res
