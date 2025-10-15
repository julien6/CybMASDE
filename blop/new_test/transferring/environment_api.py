import time
import numpy as np


class EnvironmentAPI:
    """Class representing the environment API for interacting with the environment."""

    def __init__(self):
        self.api_url = "http://127.0.0.1:5030/".strip('/')
        self.session = None
        try:
            import requests
            self.session = requests.Session()
        except ImportError:
            raise RuntimeError(
                'Le module requests est requis pour utiliser cette API.')

    def agents(self) -> int:
        """Return the agents in the environment."""
        url = f"{self.api_url}/agents"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()

    def retrieve_joint_observation(self):
        """Réinitialise l'environnement et retourne l'observation initiale."""
        url = f"{self.api_url}/last_observation"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json().get('last_observation')

    def retrieve_joint_histories(self):
        """Non implémenté : dépend de l'API serveur."""
        raise NotImplementedError(
            "L'API serveur ne fournit pas d'historiques.")

    def apply_joint_action(self, joint_action):
        """Envoie une action conjointe à l'environnement et retourne l'observation suivante, reward, done, info."""
        url = f"{self.api_url}/step"
        resp = self.session.post(url, json=joint_action)
        resp.raise_for_status()
        return resp.json()

    def deploy_joint_policy(self, joint_policy):
        """Non implémenté : dépend de la logique de la politique."""
        raise NotImplementedError("Déploiement de politique non implémenté.")

    def get_action_space(self):
        url = f"{self.api_url}/action_space"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json().get('action_space')

    def get_observation_space(self):
        url = f"{self.api_url}/observation_space"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json().get('observation_space')

    def get_agents(self):
        url = f"{self.api_url}/agents"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json().get('agents')


if __name__ == "__main__":
    api = EnvironmentAPI()
    print("--- Test get_agents() ---")
    try:
        agents = api.get_agents()
        print("Agents:", agents)
    except Exception as e:
        print("Erreur get_agents:", e)

    print("--- Test get_action_space() ---")
    try:
        action_space = api.get_action_space()
        print("Action space:", action_space)
    except Exception as e:
        print("Erreur get_action_space:", e)

    print("--- Test get_observation_space() ---")
    try:
        obs_space = api.get_observation_space()
        print("Observation space:", obs_space)
    except Exception as e:
        print("Erreur get_observation_space:", e)

    print("--- Test retrieve_joint_observation() ---")
    try:
        obs = api.retrieve_joint_observation()
        print("Observation initiale:", obs)
    except Exception as e:
        print("Erreur retrieve_joint_observation:", e)

    print("--- Test apply_joint_action() ---")
    try:
        # Exemple d'action conjointe : à adapter selon l'espace d'action réel
        joint_action = [0, 0]
        result = api.apply_joint_action(joint_action)
        print("Résultat step:", result)
    except Exception as e:
        print("Erreur apply_joint_action:", e)
