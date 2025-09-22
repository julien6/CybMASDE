import json

from dataclasses import dataclass, field, asdict
import os
from typing import Optional, Union, Dict, Any, Literal
from pathlib import Path
from typing import Any, Iterable, Tuple

# ---------- Leaf structures ----------


@dataclass
class RDLM_conf:
    """Artifacts and metadata for the learned recurrent dynamics model."""
    model: str  # path to the trained model
    statistics: Union[str, Dict[str, Any]]  # path/JSON
    hyperparameters: Union[str, Dict[str, Any]]  # path/JSON
    max_mean_square_error: float  # threshold for acceptable reconstruction error


@dataclass
class Autoencoder_conf:
    """Artifacts and metadata for the learned autoencoder."""
    model: str  # path to the trained model
    statistics: Union[str, Dict[str, Any]]  # path/JSON
    hyperparameters: Union[str, Dict[str, Any]]  # path/JSON
    max_mean_square_error: float  # threshold for acceptable reconstruction error


@dataclass
class JOPM:
    """Joint Observation Prediction Model."""
    autoencoder: Autoencoder_conf
    rdlm: RDLM_conf
    initial_joint_observation: str


@dataclass
class WorldModel:
    """Artifacts and metadata for the learned world model."""
    statistics: Union[str, Dict[str, Any]]
    jopm: JOPM
    used_traces_path: str


@dataclass
class GeneratedEnvironment:
    """Configuration of the generated (modeled) environment."""
    world_model: WorldModel
    component_functions_path: str


@dataclass
class SimulatedEnvironment:
    """Pointer to a handcrafted PettingZoo/RLlib env or to a generated one."""
    environment_path: Optional[str] = None
    generated_environment: Optional[GeneratedEnvironment] = None


class Modelling:
    """Phase 1: modeling."""
    simulated_environment: SimulatedEnvironment
    organizational_specifications: Union[str,
                                         Dict[str, Any]]  # path/JSON/Python

    def __init__(self, simulated_environment: SimulatedEnvironment,
                 organizational_specifications: Union[str, Dict[str, Any]]):
        self.simulated_environment = simulated_environment
        self.organizational_specifications = organizational_specifications


@dataclass
class Common:
    """Common metadata and label manager hookup."""
    project_name: str
    project_description: str
    label_manager: Union[str, Dict[str, Any]]  # path or Python code/spec
    project_path: str


class Training:
    """Phase 2: training of the joint policy."""
    hyperparameters: Union[str, Dict[str, Any]]
    joint_policy: str
    statistics: Union[str, Dict[str, Any]]

    def __init__(self, hyperparameters: Union[str, Dict[str, Any]],
                 statistics: Union[str, Dict[str, Any]], joint_policy: str):
        self.hyperparameters = hyperparameters
        self.statistics = statistics
        self.joint_policy = joint_policy


class Analyzing:
    """Phase 3: post-training analysis and outputs."""
    hyperparameters: Union[str, Dict[str, Any]]
    statistics: Union[str, Dict[str, Any]]
    figures_path: str
    post_training_trajectories_path: str
    inferred_organizational_specifications: Union[str, Dict[str, Any]]

    def __init__(self, hyperparameters: Union[str, Dict[str, Any]],
                 statistics: Union[str, Dict[str, Any]], figures_path: str,
                 post_training_trajectories_path: str,
                 inferred_organizational_specifications: Union[str, Dict[str, Any]]):
        self.hyperparameters = hyperparameters
        self.statistics = statistics
        self.figures_path = figures_path
        self.post_training_trajectories_path = post_training_trajectories_path
        self.inferred_organizational_specifications = inferred_organizational_specifications


DeployMode = Literal["REMOTE", "DIRECT"]


class Transferring:
    """Phase 4: deployment & online data retrieval."""

    trajectory_retrieve_frequency: str
    trajectory_batch_size: int
    deploy_mode: DeployMode
    environment_api: str
    last_checkpoint: str
    max_nb_iteration: int

    def __init__(self, configuration: str, last_checkpoint: str):
        configuration = json.load(open(configuration, 'r'))
        self.trajectory_retrieve_frequency = configuration["trajectory_retrieve_frequency"]
        self.trajectory_batch_size = configuration["trajectory_batch_size"]
        self.deploy_mode = configuration["deploy_mode"]
        self.environment_api = configuration["environment_api"]
        self.max_nb_iteration = configuration.get(
            "max_nb_iteration", 1000)
        self.last_checkpoint = last_checkpoint

# ---------- Root structure ----------


@dataclass
class Configuration:
    """Root configuration matching the provided JSON schema."""
    common: Common = field(metadata={"json_name": "common"})
    modelling: Modelling
    training: Training
    analyzing: Analyzing
    transferring: Transferring

    # -------- Helper methods --------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Configuration":
        """Construct a Configuration from a (possibly nested) dict."""

        project_folder_path = data["common"]["project_path"]

        if project_folder_path == "":
            project_folder_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "project_example")

        def build_autoencoder(d: Dict[str, Any]) -> Autoencoder_conf:
            return Autoencoder_conf(max_mean_square_error=d["max_mean_square_error"], **{k: os.path.join(project_folder_path, v) for k, v in d.items() if k != "max_mean_square_error"})

        def build_rdlm(d: Dict[str, Any]) -> RDLM_conf:
            return RDLM_conf(max_mean_square_error=d["max_mean_square_error"], **{k: os.path.join(project_folder_path, v) for k, v in d.items() if k != "max_mean_square_error"})

        def build_jopm(d: Dict[str, Any]) -> JOPM:
            d["autoencoder"] = build_autoencoder(d["autoencoder"])
            d["rdlm"] = build_rdlm(d["rdlm"])
            return JOPM(autoencoder=d["autoencoder"], rdlm=d["rdlm"], **{k: os.path.join(project_folder_path, v) for k, v in d.items() if k != "autoencoder" and k != "rdlm"})

        def build_world_model(d: Dict[str, Any]) -> WorldModel:
            d["jopm"] = build_jopm(d["jopm"])
            return WorldModel(jopm=d["jopm"], **{k: os.path.join(project_folder_path, v) for k, v in d.items() if k != "jopm"})

        def build_generated_env(d: Dict[str, Any]) -> GeneratedEnvironment:
            d = dict(d)
            d["world_model"] = build_world_model(d["world_model"])
            d["component_functions_path"] = os.path.join(
                project_folder_path, d["component_functions_path"])
            return GeneratedEnvironment(**d)

        def build_sim_env(d: Dict[str, Any]) -> SimulatedEnvironment:
            d = dict(d)
            # generated_environment est optionnel
            if "generated_environment" in d and d["generated_environment"] is not None:
                d["generated_environment"] = build_generated_env(
                    d["generated_environment"])
            d["environment_path"] = os.path.join(
                project_folder_path, d["environment_path"])
            return SimulatedEnvironment(**d)

        def build_modelling(d: Dict[str, Any]) -> Modelling:
            d = dict(d)
            d["simulated_environment"] = build_sim_env(
                d["simulated_environment"])
            d["organizational_specifications"] = os.path.join(
                project_folder_path, d["organizational_specifications"])
            return Modelling(**d)

        def build_common(d: Dict[str, Any]) -> Common:
            return Common(**d)

        def build_training(d: Dict[str, Any]) -> Training:
            return Training(**{k: os.path.join(project_folder_path, v) for k, v in d.items()})

        def build_analyzing(d: Dict[str, Any]) -> Analyzing:
            return Analyzing(**{k: os.path.join(project_folder_path, v) for k, v in d.items()})

        def build_transferring(d: Dict[str, Any]) -> Transferring:
            return Transferring(**{k: os.path.join(project_folder_path, v) for k, v in d.items()})

        return cls(
            common=build_common(data["common"]),
            modelling=build_modelling(data["modelling"]),
            training=build_training(data["training"]),
            analyzing=build_analyzing(data["analyzing"]),
            transferring=build_transferring(data["transferring"]),
        )

    def serialize_configuration(self) -> dict:
        """
        Recursively serialize a Configuration object (and all its nested objects)
        into a JSON-compatible dictionary.
        """

        def serialize(self):
            # Dataclass
            if hasattr(self, "__dataclass_fields__"):
                return {k: serialize(getattr(self, k)) for k in self.__dataclass_fields__}
            # Custom class with __dict__
            elif hasattr(self, "__dict__"):
                return {k: serialize(v) for k, v in self.__dict__.items() if not k.startswith("_")}
            # List or tuple
            elif isinstance(self, (list, tuple)):
                return [serialize(v) for v in self]
            # Dict
            elif isinstance(self, dict):
                return {k: serialize(v) for k, v in self.items()}
            # Path
            elif isinstance(self, Path):
                return str(self)
            # Literal, str, int, float, bool, None
            else:
                return self

        return serialize(self)

    @classmethod
    def from_json(cls, src: Union[str, Path]) -> "Configuration":
        """Load a Configuration from a JSON file path."""
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self, dst: str = None) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict, restoring the 'common' key name."""

        project_configuration_dict = self.serialize_configuration()

        dst = Path(
            dst) if dst else project_configuration_dict["common"]["project_path"]

        del project_configuration_dict["transferring"]["environment_api"]
        del project_configuration_dict["transferring"]["deploy_mode"]
        del project_configuration_dict["transferring"]["trajectory_batch_size"]
        del project_configuration_dict["transferring"]["trajectory_retrieve_frequency"]
        project_configuration_dict["transferring"]["configuration"] = os.path.join(
            dst, "configuration.json")
        return project_configuration_dict

    def to_json(self, dst: Union[str, Path], indent: int = 2) -> None:
        """Dump the configuration to a JSON file."""
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(dst=dst), f,
                      ensure_ascii=False, indent=indent)

   # -------- Minimal validation (optional but handy) --------

    def validate(self) -> None:
        """Lightweight checks that required fields look sane."""
        # example checks (extend as needed)
        if self.transferring.deploy_mode not in ("remote", "direct"):
            raise ValueError(
                "transferring.deploy_mode must be 'remote' or 'direct'")
        if not isinstance(self.transferring.trajectory_batch_size, int) or self.transferring.trajectory_batch_size <= 0:
            raise ValueError(
                "transferring.trajectory_batch_size must be a positive integer")
        # Ensure at least one env source is provided
        simenv = self.modelling.simulated_environment
        if not simenv.environment_path and not simenv.generated_environment:
            raise ValueError(
                "modelling.simulated_environment must define either 'environment_path' or 'generated_environment'")
