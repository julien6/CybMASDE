import os
import json

from dataclasses import dataclass, field, asdict
from typing import Optional, Union, Dict, Any, Literal
from pathlib import Path
from typing import Any, Iterable, Tuple
from world_model.jopm import JOPM

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
class JOPM_conf:
    """Artifacts and metadata for the learned joint observation prediction model."""
    autoencoder: Autoencoder_conf
    rdlm: RDLM_conf
    # path to the initial joint observations used to reset the RDLM
    initial_joint_observations: str


@dataclass
class WorldModel:
    """Artifacts and metadata for the learned world model."""
    statistics: Union[str, Dict[str, Any]]
    jopm: JOPM_conf
    used_traces_path: str


@dataclass
class GeneratedEnvironment:
    """Configuration of the generated (modeled) environment."""
    world_model: WorldModel
    component_functions_path: str


@dataclass
class Modelling:
    """Phase 1: modeling."""
    environment_path: Optional[str] = None
    generated_environment: Optional[GeneratedEnvironment] = None


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
    organizational_specifications: Union[str,
                                         Dict[str, Any]]  # path/JSON/Python

    def __init__(self, project_folder_path: str, hyperparameters: Union[str, Dict[str, Any]],
                 statistics: Union[str, Dict[str, Any]], joint_policy: str,
                 organizational_specifications: Union[str, Dict[str, Any]]):
        self.hyperparameters = os.path.join(
            project_folder_path, hyperparameters)
        self.statistics = statistics
        self.joint_policy = joint_policy
        self.organizational_specifications = os.path.join(
            project_folder_path, organizational_specifications)


class Refining:
    """Phase 3.5: transfer learning configuration.  Optional."""
    max_refinement_cycles: int
    auto_continue_refinement: bool = False

    def __init__(self, max_refinement_cycles: int = 1, auto_continue_refinement: bool = False):
        self.max_refinement_cycles = max_refinement_cycles
        self.auto_continue_refinement = auto_continue_refinement


class Analyzing:
    """Phase 3: post-training analysis and outputs."""
    hyperparameters: Union[str, Dict[str, Any]]
    statistics: Union[str, Dict[str, Any]]
    figures_path: str
    post_training_trajectories_path: str
    inferred_organizational_specifications: Union[str, Dict[str, Any]]

    def __init__(self, project_folder_path: str, hyperparameters: Union[str, Dict[str, Any]],
                 statistics: Union[str, Dict[str, Any]], figures_path: str,
                 post_training_trajectories_path: str,
                 inferred_organizational_specifications: Union[str, Dict[str, Any]]):
        self.hyperparameters = json.load(
            open(os.path.join(project_folder_path, hyperparameters), 'r'))

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

    def __init__(self, project_folder_path: str, configuration: str, last_checkpoint: str):
        configuration = json.load(
            open(os.path.join(project_folder_path, configuration), 'r'))
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
    refining: Refining

    # -------- Helper methods --------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Configuration":
        """Construct a Configuration from a (possibly nested) dict."""

        project_folder_path = data["common"]["project_path"]

        if project_folder_path == "":
            project_folder_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "project_example")

        def build_autoencoder(d: Dict[str, Any]) -> Autoencoder_conf:
            return Autoencoder_conf(**d)

        def build_rdlm(d: Dict[str, Any]) -> RDLM_conf:
            return RDLM_conf(**d)

        def build_jopm(d: Dict[str, Any]) -> JOPM_conf:
            d["autoencoder"] = build_autoencoder(d["autoencoder"])
            d["rdlm"] = build_rdlm(d["rdlm"])
            return JOPM_conf(**d)

        def build_world_model(d: Dict[str, Any]) -> WorldModel:
            d["jopm"] = build_jopm(d["jopm"])
            return WorldModel(**d)

        def build_generated_env(d: Dict[str, Any]) -> GeneratedEnvironment:
            d = dict(d)
            d["world_model"] = build_world_model(d["world_model"])
            return GeneratedEnvironment(**d)

        def build_common(d: Dict[str, Any]) -> Common:
            return Common(**d)

        def build_modelling(d: Dict[str, Any]) -> Modelling:
            d = dict(d)
            if "generated_environment" in d and d["generated_environment"] is not None:
                d["generated_environment"] = build_generated_env(
                    d["generated_environment"])
            return Modelling(**d)

        def build_training(d: Dict[str, Any]) -> Training:
            return Training(project_folder_path, **d)

        def build_analyzing(d: Dict[str, Any]) -> Analyzing:
            return Analyzing(project_folder_path, **d)

        def build_transferring(d: Dict[str, Any]) -> Transferring:
            return Transferring(project_folder_path, **d)

        def build_refining(d: Dict[str, Any]) -> Refining:
            return Refining(**d)

        return cls(
            common=build_common(data["common"]),
            modelling=build_modelling(data["modelling"]),
            training=build_training(data["training"]),
            analyzing=build_analyzing(data["analyzing"]),
            transferring=build_transferring(data["transferring"]),
            refining=build_refining(data["refining"]),
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

        del project_configuration_dict["transferring"]["max_nb_iteration"]
        del project_configuration_dict["transferring"]["environment_api"]
        del project_configuration_dict["transferring"]["deploy_mode"]
        del project_configuration_dict["transferring"]["trajectory_batch_size"]
        del project_configuration_dict["transferring"]["trajectory_retrieve_frequency"]
        project_configuration_dict["transferring"]["configuration"] = "transferring/configuration.json"
        return project_configuration_dict

    def to_json(self, dst: Union[str, Path], indent: int = 2) -> None:
        """Dump the configuration to a JSON file."""
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(dst=dst), f,
                      ensure_ascii=False, indent=indent)
