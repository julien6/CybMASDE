import os
import json
import time

from environment_api import EnvironmentAPI
from project import Configuration, Modelling, Training, Analyzing
from project import current_project
from flask import request, jsonify, Response
from project import Project
from multiprocessing import Process
import logging

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

        # Initialize the environment API
        environment_api = EnvironmentAPI(
            self.configuration.environment_api_url)

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
                print("No world model hyperparameters provided, using default ones...")
                self.modelling.simulated_environment.generated_environment.world_model.hyperparameters = cybmasde_conf[
                    "default_world_model_hyperparameters"]

            # Run the world model hyperparameter optimization (HPO)
            print("Running world model hyperparameter optimization...")
            self.run_world_model_hpo()

            # Generate the world model with the best hyperparameters
            print("Generating world model with best hyperparameters...")
            self.generate_world_model()

            # Assemble the reward function, world model (and optionally the rendering function) into the simulated environment
            print("Assembling the simulated environment...")
            self.assemble_simulated_environment()

    def run_training_activity(self):
        """Run the training activity."""
        print("Running training activity...")
        # Implement training activity logic here

    def run_analyzing_activity(self):
        """Run the analyzing activity."""
        print("Running analyzing activity...")
        # Implement analyzing activity logic here
