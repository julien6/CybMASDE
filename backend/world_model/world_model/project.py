import copy
import importlib
import json
import os
from pathlib import Path
import sys
import time
import shutil
import argparse

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
from world_model.transferring_process import TransferringProcess
from world_model.environment_api import EnvironmentAPI
from world_model.project_configuration import Configuration
from world_model.mta_process import copy_folder


class Project:
    """Class representing a project with all components."""

    configuration: Configuration

    def __init__(self, configuration: Configuration = None):

        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = Configuration.from_json(os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "project_example", "project_configuration.json"))

        self.transferring_process = None

    def save(self, filePath: Optional[str] = None, name: Optional[str] = "unnamed_project"):
        """Save the project to a file or the default path."""

        self.configuration.common.project_name = name

        src = Path(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "project_example"))
        dst = None
        if filePath is not None:
            dst = Path(os.path.join(filePath, name))
            self.configuration.common.project_path = dst
        else:
            if not self.configuration.common.project_path.endswith("project_example"):
                dst = Path(self.configuration.common.project_path)
            else:
                dst = Path(os.path.join(os.path.expanduser("~"), name))
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Project saved to {dst}")
        self.configuration.to_json(os.path.join(
            dst, "project_configuration.json"))

    def create_project(self, name: str, description: str, output: Optional[str] = None, template: str = "handcrafted"):
        """Create a new project with the given parameters."""
        if output is not None:
            project_path = os.path.join(output, name)
        else:
            project_path = os.path.join(os.path.expanduser("~"), name)

        self.configuration.common.project_name = name
        self.configuration.common.project_description = description
        self.configuration.common.project_path = project_path
        self.configuration.modelling.mode = template

        src = Path(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "project_example"))
        dst = Path(project_path)
        if os.path.exists(dst):
            shutil.rmtree(dst)

        copy_folder(src, dst)
        print(f"Project '{name}' created at {dst} using template '{template}'")
        self.configuration.to_json(os.path.join(
            dst, "project_configuration.json"))

    def load(self, filePath: str):
        """Load the project from a file."""
        filePath = os.path.join(filePath, "project_configuration.json")
        if not os.path.exists(filePath):
            raise FileNotFoundError(
                f"Project configuration file not found: {filePath}")
        self.configuration = Configuration.from_json(filePath)
        print(f"Project loaded from {Path(filePath).parent}")

    def validate(self, config: Configuration, args):
        """Validate the project configuration."""
        print("Validating project configuration...")
        errors = config.validate(strict=getattr(args, "strict", False))
        if errors:
            print("Validation errors found:")
            for error in errors:
                print(f"- {error}")
            if not getattr(args, "quiet", False):
                print("Please fix the above errors.")
        else:
            print("Project configuration is valid.")

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(configuration=Configuration.from_dict(data))

    def run(self):
        """Run the project, executing activities."""

        print("Running project:", os.path.join(
            self.configuration.common.project_path, "transferring/environment_api.py"))
        spec = importlib.util.spec_from_file_location(
            "EnvironmentAPI", os.path.join(
                self.configuration.common.project_path, "transferring/environment_api.py"))

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Recherche la classe EnvironmentAPI dans le module
        self.environment_api: EnvironmentAPI = None
        if hasattr(module, "EnvironmentAPI"):
            self.environment_api = getattr(module, "EnvironmentAPI")()
        else:
            raise ImportError(
                f"No EnvironmentAPI class found in {self.configuration.transferring.environment_api}")

        self.run_transferring_process()

    def run_transferring_process(self):
        """Run the transferring process of the project."""
        self.transferring_process = TransferringProcess(
            self.configuration, self.environment_api, joint_policy=None)
        self.transferring_process.start()
        print("Process transferring started independently.")

    def stop(self):
        """Stop the transferring process cleanly."""
        if self.transferring_process is not None:
            print("Stopping transferring process...")
            self.transferring_process.terminate()  # SIGTERM
            self.transferring_process.join()
            print("Transferring process stopped.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run a project with optional project path.")
    parser.add_argument("--project_path", type=str,
                        help="Path to the project directory")
    args = parser.parse_args()

    if args.project_path:
        project_path = args.project_path
    else:
        project_path = os.path.join(
            os.path.expanduser("~"), "Documents", "new_test")

    project = Project()
    project.load(project_path)

    try:
        project.run()
        if project.transferring_process is not None:
            project.transferring_process.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
        project.stop()
        sys.exit(0)
