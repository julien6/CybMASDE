import os
import json
import time
import threading

from joint_policy import JointPolicy
from mta_process import MTAProcess
from environment_api import EnvironmentAPI
from project import DeployMode, Transferring
from project import current_project
from flask import request, jsonify, Response
from project import Project
from multiprocessing import Process


class TransferringProcess(Process):

    def __init__(self, transferring: Transferring, environment_api: EnvironmentAPI, joint_policy: JointPolicy):
        super().__init__()
        self.transferring = transferring
        self.environment_api = environment_api
        self.joint_policy = joint_policy
        self.mta_thread = None
        self.mta_lock = threading.Lock()

    def run(self):
        """Run the transferring process."""

        print("Running transferring process with deploy mode:",
              self.transferring.deploy_mode)

        # Initialisation correcte pour éviter les erreurs d'index
        nb_agents = self.environment_api.nb_agents if hasattr(
            self.environment_api, "nb_agents") else 1
        last_joint_histories = [[] for _ in range(nb_agents)]

        if self.transferring.deploy_mode == DeployMode.REMOTE:
            print("Transferring joint-policy remotely...")
            last_joint_observation = None
            while True:
                if last_joint_observation is None:
                    last_joint_observation = self.environment_api.retrieve_joint_observation()
                joint_action = self.joint_policy.next_action(
                    last_joint_observation)
                last_joint_histories = [
                    last_joint_histories[i] +
                    [[last_joint_observation[i], joint_action[i]]]
                    for i in range(nb_agents)
                ]
                time.sleep(
                    float(1 / self.transferring.trajectory_retrieve_frequency))
                last_joint_observation = self.environment_api.apply_joint_action(
                    joint_action)

                if len(last_joint_histories[0]) >= self.transferring.trajectory_batch_size:
                    self.create_mta_process(last_joint_histories)
                    last_joint_histories = [[] for _ in range(nb_agents)]

        if self.transferring.deploy_mode == DeployMode.DIRECT:
            print("Transferring joint-policy directly...")
            deployable_joint_policy = self.prepare_deployable_joint_policy(
                self.joint_policy, self.transferring.trajectory_retrieve_frequency)
            self.environment_api.deploy_joint_policy(
                deployable_joint_policy)
            while True:
                time.sleep(
                    float(1 / self.transferring.trajectory_retrieve_frequency))
                retrieved_histories = self.environment_api.retrieve_joint_histories()
                last_joint_histories = [
                    last_joint_histories[i] + [retrieved_histories[i]]
                    for i in range(nb_agents)
                ]
                if len(last_joint_histories[0]) >= self.transferring.trajectory_batch_size:
                    self.create_mta_process(last_joint_histories)
                    last_joint_histories = [[] for _ in range(nb_agents)]

    def load_latest_joint_policy(self):
        """Charge la dernière politique conjointe depuis le checkpoint."""
        # À adapter selon ton format de sauvegarde/checkpoint
        checkpoint_path = os.path.join(
            current_project.checkpoints_dir, "latest_joint_policy.json")
        with open(checkpoint_path, "r") as f:
            policy_data = json.load(f)
        # À adapter selon la structure de ta politique
        return self.joint_policy.from_dict(policy_data)

    def create_mta_process(self, last_joint_histories):
        """Sauvegarde asynchrone et lancement du MTA si besoin."""

        threading.Thread(target=self.save_joint_histories,
                         args=(last_joint_histories,)).start()

        with self.mta_lock:
            if self.mta_thread is None or not self.mta_thread.is_alive():
                def run_mta():
                    mta = MTAProcess(
                        configuration=current_project.configuration,
                        modelling=current_project.modelling,
                        training=current_project.training,
                        analyzing=current_project.analyzing
                    )
                    mta.start()
                    mta.join()
                    mta.close()
                    print("MTA terminé, chargement de la nouvelle politique...")
                    new_joint_policy = self.load_latest_joint_policy()
                    self.joint_policy = new_joint_policy
                    if self.transferring.deploy_mode == DeployMode.DIRECT:
                        deployable_joint_policy = self.prepare_deployable_joint_policy(
                            self.joint_policy, self.transferring.trajectory_retrieve_frequency)
                        self.environment_api.deploy_joint_policy(
                            deployable_joint_policy)
                        print("Nouvelle politique conjointe déployée (DIRECT).")
                    elif self.transferring.deploy_mode == DeployMode.REMOTE:
                        print("Nouvelle politique conjointe chargée (REMOTE).")
                self.mta_thread = threading.Thread(target=run_mta)
                self.mta_thread.start()
                print("MTA process started.")
            else:
                print("MTA process already running.")

    def save_joint_histories(self, last_joint_histories):
        """Méthode à implémenter pour sauvegarder les trajectoires sur disque."""
        # ...implémentation de la sauvegarde...
        pass

    def prepare_deployable_joint_policy(self, joint_policy, frequency):
        """Méthode à implémenter pour préparer la politique déployable."""
        # ...implémentation...
        pass
