import importlib
import os
import json
import signal
import time
import threading

import gym
from mma_wrapper.label_manager import label_manager
from random_joint_policy import RandomJointPolicy
from component_functions import ComponentFunctions
from joint_policy import JointPolicy
from mta_process import MTAProcess
from environment_api import EnvironmentAPI
from project_configuration import Configuration, DeployMode
from multiprocessing import Process


class TransferringProcess(Process):

    def __init__(self, configuration: Configuration, environment_api: EnvironmentAPI, joint_policy: JointPolicy = None):
        super().__init__()
        self._stop_event = threading.Event()
        self.configuration = configuration
        self.environment_api = environment_api
        self.agents = self.environment_api.agents()
        self.load_label_manager()
        self.load_component_functions()
        if joint_policy is not None:
            self.joint_policy = joint_policy
        else:
            self.joint_policy = self.create_random_joint_policy(
                self.configuration)
        self.mta_thread = None
        self.mta_lock = threading.Lock()

    def _signal_handler(self, signum, frame):
        print(f"Received signal {signum}, stopping TransferringProcess...")
        self._stop_event.set()
        self.mta_process.terminate()
        self.mta_process.join()
        self.mta_process.close()

    def run(self):
        """Run the transferring process."""

        # Installer le handler SIGTERM
        signal.signal(signal.SIGTERM, self._signal_handler)
        # Optionnel, pour Ctrl-C aussi
        signal.signal(signal.SIGINT, self._signal_handler)

        print("Running transferring process with deploy mode:",
              self.configuration.transferring.deploy_mode)

        # Initialisation correcte pour éviter les erreurs d'index
        nb_agents = len(self.environment_api.agents())
        last_joint_histories = []
        last_joint_history = []
        nb_iteration = self.configuration.transferring.max_nb_iteration
        it = 0

        if self.configuration.transferring.deploy_mode == "REMOTE":
            print("Transferring joint-policy remotely...")
            last_joint_observation = None
            while not self._stop_event.is_set():

                if it >= nb_iteration or last_joint_observation == {}:
                    print(
                        "Maximum number of iterations reached, adding the history to the history list")
                    last_joint_histories.append(last_joint_history)
                    last_joint_history = []
                    last_joint_observation = None
                    it = 0

                if last_joint_observation is None:
                    last_joint_observation = self.environment_api.retrieve_joint_observation()

                joint_action = self.joint_policy.next_action(
                    last_joint_observation)

                last_joint_history.append(
                    [last_joint_observation, joint_action])

                time.sleep(
                    float(1 / self.configuration.transferring.trajectory_retrieve_frequency))

                last_joint_observation = self.environment_api.apply_joint_action(
                    joint_action)

                if len(last_joint_histories) >= self.configuration.transferring.trajectory_batch_size:
                    self.create_mta_process(last_joint_histories)
                    last_joint_histories = []

                it += 1

            print("TransferringProcess stopped.")
            if self.mta_thread is not None and self.mta_thread.is_alive():
                self.mta_thread.join()
                print("MTAProcess stopped.")

        if self.configuration.transferring.deploy_mode == "DIRECT":
            print("Transferring joint-policy directly...")
            deployable_joint_policy = self.prepare_deployable_joint_policy(
                self.joint_policy, self.configuration.transferring.trajectory_retrieve_frequency)
            self.environment_api.deploy_joint_policy(
                deployable_joint_policy)
            while True:
                time.sleep(
                    float(1 / self.configuration.transferring.trajectory_retrieve_frequency))
                retrieved_histories = self.environment_api.retrieve_joint_histories()
                last_joint_histories.append(retrieved_histories)
                if len(last_joint_histories) >= self.configuration.transferring.trajectory_batch_size:
                    self.create_mta_process(last_joint_histories)
                    last_joint_histories = []

    def load_component_functions(self):
        spec = importlib.util.spec_from_file_location(
            "ComponentFunctions", os.path.join(
                self.configuration.common.project_path, self.configuration.modelling.simulated_environment.generated_environment.component_functions_path))

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Recherche la classe ComponentFunctions dans le module
        self.component_functions: ComponentFunctions = module
        if hasattr(module, "ComponentFunctions"):
            self.component_functions = getattr(
                module, "ComponentFunctions")(label_manager=self.label_manager)
        else:
            raise ImportError(
                "No ComponentFunctions class found in ", os.path.join(
                    self.configuration.common.project_path, self.configuration.modelling.simulated_environment.generated_environment.component_functions_path))

    def load_label_manager(self):
        spec = importlib.util.spec_from_file_location(
            "label_manager", os.path.join(
                self.configuration.common.project_path, self.configuration.common.label_manager))

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Recherche la classe label_manager dans le module
        self.label_manager: label_manager = module
        if hasattr(module, "label_manager"):
            self.label_manager = getattr(
                module, "label_manager")()
        else:
            raise ImportError(
                "No label_manager class found in ", os.path.join(
                    self.configuration.common.project_path, self.configuration.common.label_manager))

    def create_random_joint_policy(self, configuration: Configuration):
        """Create a random joint policy."""
        action_space = self.label_manager.action_space
        return RandomJointPolicy(action_space, self.agents)

    def load_latest_joint_policy(self):
        """Load the latest joint policy from the checkpoint."""
        return RandomJointPolicy(gym.spaces.Discrete(2), self.agents)

    def create_mta_process(self, last_joint_histories):
        """Asynchronous save and launch of the MTA if needed."""

        saving_process = threading.Thread(
            target=self.save_joint_histories, args=(last_joint_histories,))
        saving_process.start()

        with self.mta_lock:
            if self.mta_thread is None or not self.mta_thread.is_alive():
                def run_mta():
                    self.mta_process = MTAProcess(
                        self.configuration, self.component_functions)
                    self.mta_process.start()
                    self.mta_process.join()
                    self.mta_process.close()

                    print("MTA finished, loading the new joint policy...")
                    new_joint_policy = self.load_latest_joint_policy()
                    self.joint_policy = new_joint_policy
                    if self.configuration.transferring.deploy_mode == "DIRECT":
                        deployable_joint_policy = self.prepare_deployable_joint_policy(
                            self.joint_policy, self.configuration.transferring.trajectory_retrieve_frequency)
                        self.environment_api.deploy_joint_policy(
                            deployable_joint_policy)
                        print("New joint policy deployed (DIRECT).")
                    elif self.configuration.transferring.deploy_mode == "REMOTE":
                        print("New joint policy loaded (REMOTE).")
                self.mta_thread = threading.Thread(target=run_mta)
                self.mta_thread.start()
                print("MTA process started.")
            else:
                print("MTA process already running.")

    def save_joint_histories(self, last_joint_histories):
        """Save each episode in a separate file in the specified traces directory, with incremented IDs."""

        print("Joint histories saving process started.")

        traces_dir = os.path.join(self.configuration.common.project_path,
                                  self.configuration.modelling.simulated_environment.generated_environment.world_model.used_traces_path)
        os.makedirs(traces_dir, exist_ok=True)

        existing_files = [f for f in os.listdir(traces_dir) if f.startswith(
            "joint_history_") and f.endswith(".json")]
        if existing_files:
            # Extraire les numéros X
            ids = []
            for fname in existing_files:
                try:
                    num = int(fname.replace(
                        "joint_history_", "").replace(".json", ""))
                    ids.append(num)
                except ValueError:
                    continue
            last_joint_history_id = max(ids) + 1 if ids else 0
        else:
            last_joint_history_id = 0

        # Sauvegarder chaque épisode avec un numéro unique
        for episode_data in last_joint_histories:
            filename = f"joint_history_{last_joint_history_id}.json"
            file_path = os.path.join(traces_dir, filename)
            with open(file_path, "w") as f:
                json.dump(episode_data, f)
            last_joint_history_id += 1
        time.sleep(5)
        print("Joint histories saving process finished.")

    def prepare_deployable_joint_policy(self, joint_policy, frequency):
        """MMethod to implement to prepare the deployable policy."""
        joint_policy = self.load_latest_joint_policy()
        self.environment_api.deploy_joint_policy(joint_policy)
