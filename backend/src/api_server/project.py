import json
import os

from dataclasses import dataclass
from enum import Enum
import time
from typing import Dict, Optional
from transferring_process import TransferringProcess
from mta_process import MTAProcess
from environment_api import EnvironmentAPI


class DeployMode(str, Enum):
    """Enum representing the deployment mode of the project."""
    REMOTE = "remote"
    DIRECT = "direct"


@dataclass
class Configuration:
    """Class representing the configuration of the project."""
    environment_api_url: str
    action_space: str  # Path or inline Python code
    observation_space: str  # Path or inline Python code
    file_path: Optional[str] = None  # Path to save the project configuration
    max_refinement_cycle: int = 10  # Maximum number of refinement cycles


@dataclass
class WorldModel:
    """Class representing the world model."""
    statistics: str  # Path or JSON
    model: str  # Path to best saved model
    hyperparameters: str  # Path or JSON
    used_traces_path: str  # Path to folder containing JSON trajectories


@dataclass
class Autoencoder:
    """Class representing the autoencoder."""
    model: str  # Path to best saved model
    hyperparameters: str  # Path or JSON


@dataclass
class GeneratedEnvironment:
    """Class representing the generated environment."""
    world_model: WorldModel
    reward_function: str  # Path or inline Python code
    autoencoder: Autoencoder  # Path or JSON


@dataclass
class SimulatedEnvironment:
    """Class representing the simulated environment."""
    environment_path: str  # Path to PettingZoo or RLlib Python file
    generated_environment: GeneratedEnvironment
    # Path or inline Python code for rendering
    rendering_function: Optional[str] = None


@dataclass
class Modelling:
    """Class representing the modelling phase of the project."""
    simulated_environment: SimulatedEnvironment
    organizational_specifications: str  # Path, JSON or Python for MOISE+MARL model


@dataclass
class Training:
    """Class representing the training phase of the project."""
    hyperparameters: str  # Path or JSON
    joint_policy: str  # Path to best trained joint policy
    statistics: str  # Path or JSON
    configuration: Dict  # Object representing the configuration for training


@dataclass
class Analyzing:
    """Class representing the analyzing phase of the project."""
    hyperparameters: str  # Path or JSON
    statistics: str  # Path or JSON
    figures_path: str  # Path to generated figures
    # Path to folder of agent trajectories (JSON)
    post_training_trajectories_path: str
    # Path, JSON or Python for inferred MOISE+MARL
    inferred_organizational_specifications: str


@dataclass
class Transferring:
    """Class representing the transferring phase of the project."""
    # Number or string depending on usage (e.g., seconds)
    trajectory_retrieve_frequency: int
    trajectory_batch_size: int
    deploy_mode: DeployMode


class Project:
    """Class representing a project with all components."""

    configuration: Configuration
    modelling: Modelling
    training: Training
    analyzing: Analyzing
    transferring: Transferring

    def save(self, filePath: Optional[str] = None):
        """Save the project to a file or the default path."""
        if filePath:
            with open(filePath, 'w') as f:
                json.dump(self.__dict__, f, indent=4)
        else:
            # Save to the file path specified in the configuration
            if self.configuration.file_path:
                with open(os.path.join(self.configuration.file_path, "project.json"), 'w') as f:
                    json.dump(self.to_dict(), f, indent=4)
            else:
                # Default save logic (e.g., save to a predefined path)
                default_path = "default_project.json"
                with open(default_path, 'w') as f:
                    json.dump(self.__dict__, f, indent=4)

    def load(self, filePath: str):
        """Load the project from a file."""
        with open((os.path.join(filePath, "project.json")), 'r') as f:
            data = json.load(f)
            self.configuration = Configuration(**data['configuration'])
            self.modelling = Modelling(**data['modelling'])
            self.training = Training(**data['training'])
            self.analyzing = Analyzing(**data['analyzing'])
            self.transferring = Transferring(**data['transferring'])

    def to_dict(self):
        """Convert the project to a dictionary."""
        return {
            "configuration": self.configuration.__dict__,
            "modelling": self.modelling.__dict__,
            "training": self.training.__dict__,
            "analyzing": self.analyzing.__dict__,
            "transferring": self.transferring.__dict__
        }

    def from_dict(cls, data: dict):
        """Create a Project instance from a dictionary."""
        configuration = Configuration(**data['configuration'])
        modelling = Modelling(**data['modelling'])
        training = Training(**data['training'])
        analyzing = Analyzing(**data['analyzing'])
        transferring = Transferring(**data['transferring'])
        return cls(configuration, modelling, training, analyzing, transferring)

    def run(self):
        """Run the project, executing activities."""
        # This method can be implemented to run the project
        self.environment_api = EnvironmentAPI(
            self.configuration.environment_api_url)
        self.run_transferring_process()

    def run_transferring_process(self):
        """Run the transferring process of the project."""
        process = TransferringProcess(config_path="project_config.json")
        process.start()
        print("Process transferring started independently.")

    def __str__(self):
        """String representation of the project."""
        return f"Project(configuration={self.configuration}, modelling={self.modelling}, training={self.training}, analyzing={self.analyzing}, transferring={self.transferring})"

    def __repr__(self):
        """Representation of the project for debugging."""
        return f"Project(configuration={self.configuration!r}, modelling={self.modelling!r}, training={self.training!r}, analyzing={self.analyzing!r}, transferring={self.transferring!r})"

    def __eq__(self, other):
        """Check equality of two Project instances."""
        if not isinstance(other, Project):
            return False
        return (self.configuration == other.configuration and
                self.modelling == other.modelling and
                self.training == other.training and
                self.analyzing == other.analyzing and
                self.transferring == other.transferring)

    def __hash__(self):
        """Hash function for the Project class."""
        return hash((self.configuration, self.modelling, self.training, self.analyzing, self.transferring))

    def validate(self):
        """Validate the project data."""
        if not isinstance(self.configuration, Configuration):
            raise ValueError("Invalid configuration")
        if not isinstance(self.modelling, Modelling):
            raise ValueError("Invalid modelling")
        if not isinstance(self.training, Training):
            raise ValueError("Invalid training")
        if not isinstance(self.analyzing, Analyzing):
            raise ValueError("Invalid analyzing")
        if not isinstance(self.transferring, Transferring):
            raise ValueError("Invalid transferring")
        # Additional validation logic can be added here
        return True

    def reset(self):
        """Reset the project to its initial state."""
        self.configuration = Configuration("", "", "")
        self.modelling = Modelling(SimulatedEnvironment(
            "", GeneratedEnvironment(WorldModel("", "", "", ""))), "")
        self.training = Training("", "", "")
        self.analyzing = Analyzing("", "", "", "", "")
        self.transferring = Transferring("", "", "")
