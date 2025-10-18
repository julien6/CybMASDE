import copy
import importlib
import json
import os
from pathlib import Path
import sys
import time
import shutil
import argparse
import jsonschema

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
from world_model.transferring_process import TransferringProcess
from world_model.environment_api import EnvironmentAPI
from world_model.project_configuration import Configuration
from world_model.mta_process import copy_folder, MTAProcess


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

    def validate(self, args):
        """Validate the project configuration."""
        print("Validating project configuration...")

        with open(self.configuration.common.project_path, "r") as f:
            config = json.load(f)

        if getattr(args, "quiet", True):
            with open("cybmasde_project_schema.json", "r") as f:
                schema = json.load(f)
            try:
                jsonschema.validate(instance=config, schema=schema)
                print(
                    "✅ Project configuration JSON does comply with the expected schema.")
            except jsonschema.ValidationError as e:
                print("❌ Validation error:", e.message)

        if getattr(args, "strict", False):
            # Check each string value in the config that looks like a path actually exists.
            project_dir = Path(self.configuration.common.project_path)
            # If project_path points to a file (like a config file), use its parent as base dir
            if project_dir.is_file():
                project_dir = project_dir.parent

            missing = []

            def _check_paths(obj, key_path=""):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        _check_paths(v, f"{key_path}.{k}" if key_path else k)
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        _check_paths(v, f"{key_path}[{i}]")
                elif isinstance(obj, str):
                    # Heuristics to decide if a string should be treated as a file path
                    looks_like_path = (
                        os.path.sep in obj
                        or obj.startswith(".")
                        or obj.endswith((".py", ".json", ".yaml", ".yml", ".txt", ".csv", ".xml", ".ini"))
                    )
                    if looks_like_path:
                        p = Path(obj)
                        if not p.is_absolute():
                            p = project_dir / p
                        if not p.exists():
                            missing.append((key_path, str(p)))

            _check_paths(config)

            if missing:
                for key, path in missing:
                    print(
                        f"❌ Missing file path referenced in config at '{key}': {path}")
                raise FileNotFoundError(
                    "One or more referenced file paths in the project configuration do not exist.")
            else:
                print("✅ All referenced file paths exist.")
            print("Project configuration is valid under strict mode.")

        print("Project configuration is valid.")

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(configuration=Configuration.from_dict(data))

    def run(self, args=None):
        """Run the project, executing activities.

        If `args` is provided (parsed CLI args), delegate to `run2` which
        orchestrates the pipeline according to the CLI flags. If `args` is
        None, preserve the previous behaviour (start transferring using the
        configuration already loaded in the Project instance).
        """

        if args is not None:
            return self.run2(args)

        # Backwards-compatible behaviour: start transferring from current configuration
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

    def run2(self, args):
        """Run entrypoint used by the CLI (cybmasde run ...).

        This function loads the project configuration from CLI args when
        provided, applies a few run-time parameters (max refinement cycles,
        thresholds, inference flags) and starts the transferring process.

        The function is intentionally conservative: it orchestrates the
        pipeline and starts the transferring process which itself will spawn
        the MTA processes when trajectories are collected. This avoids
        complex cross-process orchestration in the parent process.
        """

        # Load project configuration from CLI args (project path + config)
        project_path = getattr(args, "project", None)
        config_file = getattr(args, "config", None)

        if project_path:
            # if a custom config filename is provided and exists under project path
            if config_file:
                candidate = config_file if os.path.isabs(
                    config_file) else os.path.join(project_path, config_file)
                if os.path.exists(candidate):
                    # load explicitly the provided configuration file
                    self.configuration = Configuration.from_json(candidate)
                else:
                    # fallback: try default file in project path
                    try:
                        self.load(project_path)
                    except Exception:
                        raise FileNotFoundError(
                            f"Could not find configuration file: {candidate}")
            else:
                # no explicit config filename -> use default behaviour
                self.load(project_path)

        # Apply CLI run parameters into the configuration where relevant
        try:
            max_refine = int(getattr(args, "max_refine", None)) if getattr(
                args, "max_refine", None) is not None else None
        except Exception:
            max_refine = None

        if max_refine is not None:
            # best-effort: update refining configuration
            try:
                self.configuration.refining.max_refinement_cycles = max_refine
            except Exception:
                # attach dynamically if needed
                setattr(self.configuration.refining,
                        "max_refinement_cycles", max_refine)

        # thresholds for early stopping during training/analysis
        reward_th = getattr(args, "reward_threshold", None)
        std_th = getattr(args, "std_threshold", None)

        if reward_th is not None:
            setattr(self.configuration, "reward_threshold", reward_th)
        if std_th is not None:
            setattr(self.configuration, "std_threshold", std_th)

        # inference / refinement flags
        accept_inferred = getattr(args, "accept_inferred", False)
        interactive_infer = getattr(args, "interactive_infer", True)
        setattr(self.configuration, "accept_inferred", accept_inferred)
        setattr(self.configuration, "interactive_infer", interactive_infer)

        # Mode selection
        mode = "full-auto"
        if getattr(args, "semi_auto", False):
            mode = "semi-auto"
        if getattr(args, "manual", False):
            mode = "manual"

        print(f"[PROJECT] Running in mode: {mode}")
        if getattr(args, "skip_model", False):
            print("[PROJECT] Skipping modelling phase as requested (--skip-model).")
        if getattr(args, "skip_analyze", False):
            print("[PROJECT] Skipping analyzing phase as requested (--skip-analyze).")

        # Load EnvironmentAPI
        print("[PROJECT] Loading EnvironmentAPI from project transferring folder...")
        env_api_path = os.path.join(
            self.configuration.common.project_path, "transferring", "environment_api.py")
        if not os.path.exists(env_api_path):
            raise FileNotFoundError(
                f"Environment API not found at {env_api_path}")

        spec = importlib.util.spec_from_file_location(
            "EnvironmentAPI", env_api_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "EnvironmentAPI"):
            self.environment_api = getattr(module, "EnvironmentAPI")()
        else:
            raise ImportError(
                f"No EnvironmentAPI class found in {env_api_path}")

        # Start the transferring process which will orchestrate the online loop
        # and spawn MTA processes when required.
        print("[PROJECT] Starting transferring process...")
        self.run_transferring_process()

        # Inform user about next steps. For manual mode we do not block here.
        if mode == "manual":
            print("[PROJECT] Manual mode: transferring process started. Use signals or the API to control refinement cycles.")
            return

        # For full-auto and semi-auto, the transferring process will do the heavy work.
        # We simply wait for the transferring process to finish (join) unless the
        # user requested to run it in background.
        try:
            if self.transferring_process is not None:
                print(
                    "[PROJECT] Transferring process running (parent will wait). Press Ctrl-C to stop.")
                self.transferring_process.join()
        except KeyboardInterrupt:
            print("[PROJECT] KeyboardInterrupt received, stopping...")
            self.stop()

    def run_modeling(self, args):
        """Run the modelling phase from the CLI.

        Supports --auto and --manual modes, optional --traces path to preexisting
        traces, and hints for model sizes (--vae-dim, --lstm-hidden).
        """

        # Load project if specified
        project_path = getattr(args, "project", None)
        config_file = getattr(args, "config", None)
        if project_path:
            if config_file:
                candidate = config_file if os.path.isabs(
                    config_file) else os.path.join(project_path, config_file)
                if os.path.exists(candidate):
                    self.configuration = Configuration.from_json(candidate)
                else:
                    self.load(project_path)
            else:
                self.load(project_path)

        # Determine mode
        if getattr(args, "auto", False) and getattr(args, "manual", False):
            raise ValueError(
                "Please specify only one of --auto or --manual for modelling.")
        mode = "auto" if getattr(args, "auto", False) else "manual"

        print(
            f"[PROJECT] Starting modelling phase (mode={mode}) for project: {self.configuration.common.project_path}")

        # If traces provided, copy them into the project's traces folder
        traces_src = getattr(args, "traces", None)
        if traces_src:
            dst_traces_dir = os.path.join(self.configuration.common.project_path,
                                          self.configuration.modelling.generated_environment.world_model.used_traces_path)
            os.makedirs(dst_traces_dir, exist_ok=True)
            if os.path.isdir(traces_src):
                for fname in os.listdir(traces_src):
                    srcf = os.path.join(traces_src, fname)
                    if os.path.isfile(srcf):
                        shutil.copy2(srcf, os.path.join(dst_traces_dir, fname))
                print(
                    f"[PROJECT] Copied traces from {traces_src} to {dst_traces_dir}")
            elif os.path.isfile(traces_src):
                shutil.copy2(traces_src, os.path.join(
                    dst_traces_dir, os.path.basename(traces_src)))
                print(
                    f"[PROJECT] Copied trace file {traces_src} to {dst_traces_dir}")
            else:
                print(
                    f"[PROJECT] Warning: traces path {traces_src} does not exist.")

        # Load label manager
        try:
            lm_path = os.path.join(
                self.configuration.common.project_path, self.configuration.common.label_manager)
            spec = importlib.util.spec_from_file_location(
                "label_manager", lm_path)
            lm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lm_module)
            if hasattr(lm_module, "label_manager"):
                label_manager_cls = getattr(lm_module, "label_manager")
                lbl_manager = label_manager_cls()
            else:
                raise ImportError(
                    "No label_manager found in project label_manager file.")
        except Exception as e:
            raise ImportError(f"Failed to load label manager: {e}")

        # Load component functions (if present)
        component_functions_obj = None
        try:
            comp_rel = self.configuration.modelling.generated_environment.component_functions_path
            comp_path = os.path.join(
                self.configuration.common.project_path, comp_rel)
            spec = importlib.util.spec_from_file_location(
                "ComponentFunctions", comp_path)
            comp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(comp_module)
            if hasattr(comp_module, "ComponentFunctions"):
                component_functions_obj = getattr(
                    comp_module, "ComponentFunctions")(label_manager=lbl_manager)
            else:
                print("[PROJECT] No ComponentFunctions class found in component functions file; modelling may fail if required methods are missing.")
        except Exception as e:
            print(f"[PROJECT] Could not load component functions: {e}")

        # Instantiate MTAProcess and run modelling activity (in-process)
        try:
            mta = MTAProcess(self.configuration, component_functions_obj)

            # Attach CLI hints (may be used by internal HPO or helpers)
            vae_dim = getattr(args, "vae_dim", None) or getattr(
                args, "vae-dim", None)
            lstm_hidden = getattr(args, "lstm_hidden", None) or getattr(
                args, "lstm-hidden", None)
            if vae_dim is not None:
                setattr(mta, "cli_vae_dim", int(vae_dim))
                print(f"[PROJECT] Hint: VAEs latent dim set to {vae_dim}")
            if lstm_hidden is not None:
                setattr(mta, "cli_lstm_hidden", int(lstm_hidden))
                print(
                    f"[PROJECT] Hint: RDLN/LSTM hidden size set to {lstm_hidden}")

            if mode == "manual":
                print(
                    "[PROJECT] Manual modelling mode selected: please implement or run modelling steps manually.")
                return

            # Run modelling activity directly (not as a separate process) so user sees logs here
            mta.run_modelling_activity()
            print("[PROJECT] Modelling phase completed.")

        except Exception as e:
            print(f"[PROJECT] Error while running modelling phase: {e}")
            raise

    def run_training(self, args):
        """Run the training phase from the CLI.

        Reads CLI hyperparameters (algo, batch-size, lr, gamma, clip, seed, epochs),
        loads project/config if provided, prepares label manager and component
        functions, instantiates an MTAProcess and runs the training activity in-process.
        """

        # Load project if specified
        project_path = getattr(args, "project", None)
        config_file = getattr(args, "config", None)
        if project_path:
            if config_file:
                candidate = config_file if os.path.isabs(
                    config_file) else os.path.join(project_path, config_file)
                if os.path.exists(candidate):
                    self.configuration = Configuration.from_json(candidate)
                else:
                    self.load(project_path)
            else:
                self.load(project_path)

        print(
            f"[PROJECT] Starting training phase for project: {self.configuration.common.project_path}")

        # Load label manager (required by some component functions)
        try:
            lm_path = os.path.join(
                self.configuration.common.project_path, self.configuration.common.label_manager)
            spec = importlib.util.spec_from_file_location(
                "label_manager", lm_path)
            lm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lm_module)
            if hasattr(lm_module, "label_manager"):
                label_manager_cls = getattr(lm_module, "label_manager")
                lbl_manager = label_manager_cls()
            else:
                raise ImportError(
                    "No label_manager found in project label_manager file.")
        except Exception as e:
            raise ImportError(f"Failed to load label manager: {e}")

        # Load component functions (optional)
        component_functions_obj = None
        try:
            comp_rel = self.configuration.modelling.generated_environment.component_functions_path
            comp_path = os.path.join(
                self.configuration.common.project_path, comp_rel)
            spec = importlib.util.spec_from_file_location(
                "ComponentFunctions", comp_path)
            comp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(comp_module)
            if hasattr(comp_module, "ComponentFunctions"):
                component_functions_obj = getattr(
                    comp_module, "ComponentFunctions")(label_manager=lbl_manager)
            else:
                print(
                    "[PROJECT] No ComponentFunctions class found in component functions file; training may still proceed with defaults.")
        except Exception as e:
            print(f"[PROJECT] Could not load component functions: {e}")

        # Instantiate MTAProcess and run training activity
        try:
            mta = MTAProcess(self.configuration, component_functions_obj)

            # Attach CLI hints so MTA can optionally consume them
            algo = getattr(args, "algo", None)
            batch_size = getattr(args, "batch_size", None) or getattr(
                args, "batch-size", None)
            lr = getattr(args, "lr", None)
            gamma = getattr(args, "gamma", None)
            clip = getattr(args, "clip", None)
            seed = getattr(args, "seed", None)
            epochs = getattr(args, "epochs", None)

            if algo is not None:
                setattr(mta, "cli_algo", algo)
                print(f"[PROJECT] Hint: training algorithm set to {algo}")
            if batch_size is not None:
                setattr(mta, "cli_batch_size", int(batch_size))
                print(f"[PROJECT] Hint: batch size set to {batch_size}")
            if lr is not None:
                setattr(mta, "cli_lr", float(lr))
                print(f"[PROJECT] Hint: learning rate set to {lr}")
            if gamma is not None:
                setattr(mta, "cli_gamma", float(gamma))
                print(f"[PROJECT] Hint: gamma set to {gamma}")
            if clip is not None:
                setattr(mta, "cli_clip", float(clip))
                print(f"[PROJECT] Hint: clip set to {clip}")
            if seed is not None:
                setattr(mta, "cli_seed", int(seed))
                print(f"[PROJECT] Hint: random seed set to {seed}")
            if epochs is not None:
                setattr(mta, "cli_epochs", int(epochs))
                print(f"[PROJECT] Hint: training epochs set to {epochs}")

            # Run training activity synchronously so logs appear in caller
            mta.run_training_activity()
            print("[PROJECT] Training phase completed.")

        except Exception as e:
            print(f"[PROJECT] Error while running training phase: {e}")
            raise

    def run_analyzing(self, args):
        """Run the analyzing phase (Auto-TEMM) from the CLI.

        Supports --auto-temm (boolean), --metrics (list of metric names) and
        --representativity (float). Loads project/config if provided, prepares
        label manager and component functions, instantiates MTAProcess and runs
        the analyzing activity in-process.
        """

        # Load project if specified
        project_path = getattr(args, "project", None)
        config_file = getattr(args, "config", None)
        if project_path:
            if config_file:
                candidate = config_file if os.path.isabs(
                    config_file) else os.path.join(project_path, config_file)
                if os.path.exists(candidate):
                    self.configuration = Configuration.from_json(candidate)
                else:
                    self.load(project_path)
            else:
                self.load(project_path)

        auto_temm = getattr(args, "auto_temm", False) or getattr(
            args, "auto-temm", False)
        metrics = getattr(args, "metrics", None)
        representativity = getattr(args, "representativity", None)

        print(
            f"[PROJECT] Starting analyzing phase (auto-temm={auto_temm}) for project: {self.configuration.common.project_path}")
        if metrics is not None:
            print(f"[PROJECT] Metrics requested: {metrics}")
        if representativity is not None:
            print(f"[PROJECT] Representativity threshold: {representativity}")

        # Load label manager (required by analyzer routines)
        try:
            lm_path = os.path.join(
                self.configuration.common.project_path, self.configuration.common.label_manager)
            spec = importlib.util.spec_from_file_location(
                "label_manager", lm_path)
            lm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lm_module)
            if hasattr(lm_module, "label_manager"):
                label_manager_cls = getattr(lm_module, "label_manager")
                lbl_manager = label_manager_cls()
            else:
                raise ImportError(
                    "No label_manager found in project label_manager file.")
        except Exception as e:
            raise ImportError(f"Failed to load label manager: {e}")

        # Load component functions (optional)
        component_functions_obj = None
        try:
            comp_rel = self.configuration.modelling.generated_environment.component_functions_path
            comp_path = os.path.join(
                self.configuration.common.project_path, comp_rel)
            spec = importlib.util.spec_from_file_location(
                "ComponentFunctions", comp_path)
            comp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(comp_module)
            if hasattr(comp_module, "ComponentFunctions"):
                component_functions_obj = getattr(
                    comp_module, "ComponentFunctions")(label_manager=lbl_manager)
            else:
                print(
                    "[PROJECT] No ComponentFunctions class found in component functions file; analyzing may still proceed with defaults.")
        except Exception as e:
            print(f"[PROJECT] Could not load component functions: {e}")

        # Instantiate MTAProcess and run analyzing activity
        try:
            mta = MTAProcess(self.configuration, component_functions_obj)

            # Attach CLI hints so MTA can consume them if implemented
            setattr(mta, "cli_auto_temm", bool(auto_temm))
            if metrics is not None:
                setattr(mta, "cli_metrics", metrics)
            if representativity is not None:
                try:
                    setattr(mta, "cli_representativity",
                            float(representativity))
                except Exception:
                    setattr(mta, "cli_representativity", representativity)

            # Run analyzing activity synchronously
            mta.run_analyzing_activity()
            print("[PROJECT] Analyzing phase completed.")

        except Exception as e:
            print(f"[PROJECT] Error while running analyzing phase: {e}")
            raise

    def run_refining(self, args):
        """Run refinement cycles according to CLI flags.

        Accepts --max (max cycles), --accept-inferred (bool), --interactive (bool).
        Loads project/config if provided, updates configuration, loads label manager
        and component functions, instantiates MTAProcess and runs the refinement
        cycles. Refinement cycles are performed inside the MTAProcess.run() loop
        which already implements the refining logic; here we start MTAProcess in
        the foreground so the user sees progress and can interrupt.
        """

        # Load project if specified
        project_path = getattr(args, "project", None)
        config_file = getattr(args, "config", None)
        if project_path:
            if config_file:
                candidate = config_file if os.path.isabs(
                    config_file) else os.path.join(project_path, config_file)
                if os.path.exists(candidate):
                    self.configuration = Configuration.from_json(candidate)
                else:
                    self.load(project_path)
            else:
                self.load(project_path)

        # Apply CLI refining parameters
        max_cycles = getattr(args, "max", None)
        accept_inferred = getattr(args, "accept_inferred", False)
        interactive = getattr(args, "interactive", True)

        if max_cycles is not None:
            try:
                self.configuration.refining.max_refinement_cycles = int(
                    max_cycles)
            except Exception:
                setattr(self.configuration.refining,
                        "max_refinement_cycles", int(max_cycles))

        setattr(self.configuration, "accept_inferred", bool(accept_inferred))
        setattr(self.configuration, "interactive_infer", bool(interactive))

        print(
            f"[PROJECT] Starting refinement (max={self.configuration.refining.max_refinement_cycles}, accept_inferred={accept_inferred}, interactive={interactive}) for project: {self.configuration.common.project_path}")

        # Load label manager (best-effort)
        try:
            lm_path = os.path.join(
                self.configuration.common.project_path, self.configuration.common.label_manager)
            spec = importlib.util.spec_from_file_location(
                "label_manager", lm_path)
            lm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lm_module)
            if hasattr(lm_module, "label_manager"):
                label_manager_cls = getattr(lm_module, "label_manager")
                lbl_manager = label_manager_cls()
            else:
                lbl_manager = None
                print(
                    "[PROJECT] No label_manager class found; continuing without it.")
        except Exception as e:
            lbl_manager = None
            print(f"[PROJECT] Warning: failed to load label manager: {e}")

        # Load component functions (optional)
        component_functions_obj = None
        try:
            comp_rel = self.configuration.modelling.generated_environment.component_functions_path
            comp_path = os.path.join(
                self.configuration.common.project_path, comp_rel)
            spec = importlib.util.spec_from_file_location(
                "ComponentFunctions", comp_path)
            comp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(comp_module)
            if hasattr(comp_module, "ComponentFunctions"):
                component_functions_obj = getattr(
                    comp_module, "ComponentFunctions")(label_manager=lbl_manager)
            else:
                print(
                    "[PROJECT] No ComponentFunctions class found in component functions file; refinement may still proceed with defaults.")
        except Exception as e:
            print(f"[PROJECT] Could not load component functions: {e}")

        # Instantiate MTAProcess and run it to perform refinement cycles
        try:
            mta = MTAProcess(self.configuration, component_functions_obj)

            # Attach CLI hints
            setattr(mta, "cli_accept_inferred", bool(accept_inferred))
            setattr(mta, "cli_interactive", bool(interactive))
            setattr(mta, "cli_max_refine", int(
                getattr(self.configuration.refining, "max_refinement_cycles", 0)))

            # Run the MTAProcess run loop synchronously so refinement cycles execute here
            mta.run()
            print("[PROJECT] Refinement cycles completed.")

        except Exception as e:
            print(f"[PROJECT] Error while running refinement: {e}")
            raise

    def display(self, args):
        """Display a concise project status summary.

        Shows project path, project name, modelling/training/transferring/refining
        configuration and basic existence checks for key artifacts. Safe to call
        from the CLI 'status' subcommand.
        """

        # Try to load project config if provided (CLI global --project / --config)
        project_path = getattr(args, "project", None)
        config_file = getattr(args, "config", None)
        if project_path:
            try:
                if config_file:
                    candidate = config_file if os.path.isabs(
                        config_file) else os.path.join(project_path, config_file)
                    if os.path.exists(candidate):
                        self.configuration = Configuration.from_json(candidate)
                    else:
                        self.load(project_path)
                else:
                    self.load(project_path)
            except Exception as e:
                print(
                    f"[PROJECT] Warning: could not load project configuration: {e}")

        conf = getattr(self, "configuration", None)
        if conf is None:
            print("[PROJECT] No configuration loaded.")
            return

        print("")
        print("[STATUS] Project status summary")
        print("[STATUS] ------------------------")
        try:
            print(f"Project path: {conf.common.project_path}")
            print(f"Project name: {conf.common.project_name}")
        except Exception:
            pass

        # Modelling
        try:
            print(f"Modelling mode: {conf.modelling.mode}")
            if conf.modelling.mode == "generated" and conf.modelling.generated_environment is not None:
                wm = conf.modelling.generated_environment.world_model
                used_traces = os.path.join(
                    conf.common.project_path, wm.used_traces_path)
                print(
                    f"World-model traces dir: {used_traces} (exists={os.path.exists(used_traces)})")
        except Exception:
            pass

        # Training
        try:
            jp = conf.training.joint_policy
            print(
                f"Training joint_policy path: {jp} (exists={os.path.exists(os.path.join(conf.common.project_path, jp))})")
        except Exception:
            pass

        # Analyzing
        try:
            print(
                f"Analyzing artifacts: {getattr(conf.analyzing, 'figures_path', 'n/a')}")
        except Exception:
            pass

        # Transferring
        try:
            td = conf.transferring
            print(f"Transferring deploy mode: {td.deploy_mode}")
            print(
                f"Transfer trajectory batch size: {td.trajectory_batch_size}")
            env_api_path = os.path.join(
                conf.common.project_path, 'transferring', 'environment_api.py')
            print(
                f"Environment API module: {env_api_path} (exists={os.path.exists(env_api_path)})")
        except Exception:
            pass

        # Refining
        try:
            print(
                f"Refining max cycles: {conf.refining.max_refinement_cycles}")
            print(
                f"Refining auto-continue: {getattr(conf.refining, 'auto_continue_refinement', False)}")
        except Exception:
            pass

        # Label manager and component functions
        try:
            lm = os.path.join(conf.common.project_path,
                              conf.common.label_manager)
            print(f"Label manager: {lm} (exists={os.path.exists(lm)})")
        except Exception:
            pass
        try:
            cf = os.path.join(conf.common.project_path,
                              conf.modelling.generated_environment.component_functions_path)
            print(f"Component functions: {cf} (exists={os.path.exists(cf)})")
        except Exception:
            pass

        # Transferring process status
        try:
            running = self.transferring_process is not None and getattr(
                self.transferring_process, 'is_alive', lambda: False)()
            print(f"Transferring process running: {running}")
        except Exception:
            print("Transferring process running: unknown")

        print("[STATUS] End of summary")
        print("")

    def cleanup(self, args):
        """Clean temporary files in the project.

        Supports CLI flags:
          --traces : remove collected/used traces
          --checkpoints : remove training/checkpoint artifacts
          --all : remove both traces and checkpoints

        The method is conservative: it only removes files/directories located
        under the configured project path and is idempotent.
        """

        # Load project/config if provided (same logic as other CLI entrypoints)
        project_path = getattr(args, "project", None)
        config_file = getattr(args, "config", None)
        if project_path:
            if config_file:
                candidate = config_file if os.path.isabs(
                    config_file) else os.path.join(project_path, config_file)
                if os.path.exists(candidate):
                    self.configuration = Configuration.from_json(candidate)
                else:
                    self.load(project_path)
            else:
                self.load(project_path)

        conf = getattr(self, "configuration", None)
        if conf is None:
            print("[PROJECT] No configuration loaded. Nothing to clean.")
            return

        proj_root = os.path.abspath(conf.common.project_path)
        if not os.path.isdir(proj_root):
            print(
                f"[PROJECT] Project root is not a directory: {proj_root}. Nothing to clean.")
            return

        clean_all = getattr(args, "all", False)
        clean_traces = getattr(args, "traces", False) or clean_all
        clean_checkpoints = getattr(args, "checkpoints", False) or clean_all

        if not (clean_traces or clean_checkpoints):
            print(
                "[PROJECT] No cleanup option specified. Use --traces, --checkpoints or --all.")
            return

        def _is_within(base, target):
            try:
                base = os.path.abspath(base)
                target = os.path.abspath(target)
                return os.path.commonpath([base]) == os.path.commonpath([base, target])
            except Exception:
                return False

        # Clean traces
        if clean_traces:
            try:
                traces_rel = None
                try:
                    traces_rel = conf.modelling.generated_environment.world_model.used_traces_path
                except Exception:
                    traces_rel = None

                if traces_rel:
                    traces_dir = os.path.join(proj_root, traces_rel)
                    if os.path.exists(traces_dir) and _is_within(proj_root, traces_dir):
                        print(
                            f"[PROJECT] Removing traces directory: {traces_dir}")
                        try:
                            shutil.rmtree(traces_dir)
                        except Exception as e:
                            # ignore errors during cleanup but log a warning for visibility
                            pass

                        # Recreate empty traces directory to preserve expected layout
                        os.makedirs(traces_dir, exist_ok=True)
                        print("[PROJECT] Traces cleaned.")
                    else:
                        print(
                            f"[PROJECT] Traces directory not found: {traces_dir}")
                else:
                    # conservative fallback: look for common traces folders
                    candidates = [os.path.join(proj_root, 'traces'), os.path.join(
                        proj_root, 'data', 'traces')]
                    removed_any = False
                    for c in candidates:
                        if os.path.exists(c) and _is_within(proj_root, c):
                            print(f"[PROJECT] Removing traces folder: {c}")
                            try:
                                shutil.rmtree(c)
                            except Exception as e:
                                pass
                            removed_any = True
                    if removed_any:
                        print("[PROJECT] Traces cleaned (fallback locations).")
                    else:
                        print("[PROJECT] No traces found to clean.")
            except Exception as e:
                print(f"[PROJECT] Error while cleaning traces: {e}")

        # Clean checkpoints
        if clean_checkpoints:
            try:
                jp = None
                try:
                    jp = conf.training.joint_policy
                except Exception:
                    jp = None

                removed_any = False
                if jp:
                    # joint_policy may be an absolute path or relative to project
                    candidate = jp if os.path.isabs(
                        jp) else os.path.join(proj_root, jp)
                    if os.path.exists(candidate) and _is_within(proj_root, candidate):
                        print(
                            f"[PROJECT] Removing joint policy/checkpoint at: {candidate}")
                        if os.path.isdir(candidate):
                            try:
                                shutil.rmtree(candidate)
                            except Exception as e:
                                pass
                        else:
                            try:
                                os.remove(candidate)
                            except Exception:
                                pass
                        removed_any = True
                # fallback common checkpoint dirs
                candidates = [os.path.join(proj_root, 'checkpoints'), os.path.join(
                    proj_root, 'models'), os.path.join(proj_root, 'training_checkpoints')]
                for c in candidates:
                    if os.path.exists(c) and _is_within(proj_root, c):
                        print(f"[PROJECT] Removing checkpoint folder: {c}")
                        try:
                            shutil.rmtree(c)
                        except Exception as e:
                            pass
                        removed_any = True

                if removed_any:
                    print("[PROJECT] Checkpoints cleaned.")
                else:
                    print("[PROJECT] No checkpoints found to clean.")
            except Exception as e:
                print(f"[PROJECT] Error while cleaning checkpoints: {e}")

        print("[PROJECT] Cleanup completed.")

    def export(self, args):
        """Export results and metrics.

        CLI: --format {json,csv,yaml}, --output path
        The method collects common artifact directories (analysis_results,
        exp_results, traces, checkpoints/joint_policy) and copies them into
        the output folder. It also attempts to aggregate JSON metrics into a
        summary file in the requested format.
        """

        # Load project/config if provided
        project_path = getattr(args, "project", None)
        config_file = getattr(args, "config", None)
        if project_path:
            if config_file:
                candidate = config_file if os.path.isabs(
                    config_file) else os.path.join(project_path, config_file)
                if os.path.exists(candidate):
                    self.configuration = Configuration.from_json(candidate)
                else:
                    self.load(project_path)
            else:
                self.load(project_path)

        conf = getattr(self, "configuration", None)
        if conf is None:
            print("[PROJECT] No configuration loaded. Nothing to export.")
            return

        fmt = getattr(args, "format", "json")
        out = getattr(args, "output", "export/")

        # Resolve output path relative to project if not absolute
        out_dir = out if os.path.isabs(out) else os.path.join(os.getcwd(), out)
        # If user passed relative path and project path is known, place under project by default
        if not os.path.isabs(out) and getattr(args, "project", None):
            out_dir = os.path.join(os.path.abspath(
                conf.common.project_path), out)

        os.makedirs(out_dir, exist_ok=True)

        proj_root = os.path.abspath(conf.common.project_path)

        def _is_within(base, target):
            try:
                base = os.path.abspath(base)
                target = os.path.abspath(target)
                return os.path.commonpath([base]) == os.path.commonpath([base, target])
            except Exception:
                return False

        artifacts = []
        metrics_aggregate = {}

        # Candidate artifact folders
        candidates = []
        # analysis_results (created by MTA)
        candidates.append(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'analysis_results'))
        # exp_results (created by MTA)
        candidates.append(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'exp_results'))
        # traces from configuration
        try:
            traces_rel = conf.modelling.generated_environment.world_model.used_traces_path
            candidates.append(os.path.join(proj_root, traces_rel))
        except Exception:
            pass
        # training joint policy path
        try:
            jp = conf.training.joint_policy
            jp_path = jp if os.path.isabs(jp) else os.path.join(proj_root, jp)
            candidates.append(jp_path)
        except Exception:
            pass
        # analyzing figures (if any)
        try:
            figs = getattr(conf.analyzing, 'figures_path', None)
            if figs:
                candidates.append(os.path.join(proj_root, figs)
                                  if not os.path.isabs(figs) else figs)
        except Exception:
            pass

        # Copy candidate folders/files into output
        for src in candidates:
            try:
                if not src:
                    continue
                if not os.path.exists(src):
                    continue
                if not _is_within(proj_root, src) and os.path.commonpath([src, proj_root]) != proj_root:
                    # avoid exporting files outside project root unless they are inside package dirs
                    print(
                        f"[PROJECT] Skipping artifact outside project root: {src}")
                    continue

                dest_name = os.path.basename(src.rstrip(os.sep))
                dest = os.path.join(out_dir, dest_name)
                if os.path.isdir(src):
                    # If dest exists, merge by copying contents
                    if os.path.exists(dest):
                        # copy content of src into dest
                        for item in os.listdir(src):
                            s_item = os.path.join(src, item)
                            d_item = os.path.join(dest, item)
                            if os.path.isdir(s_item):
                                shutil.copytree(
                                    s_item, d_item, dirs_exist_ok=True)
                            else:
                                shutil.copy2(s_item, d_item)
                    else:
                        shutil.copytree(src, dest)
                else:
                    # single file
                    shutil.copy2(src, dest)

                artifacts.append({'src': src, 'dest': dest})

                # Try to collect any JSON metrics inside the copied folder
                if os.path.isdir(src):
                    for root, _, files in os.walk(src):
                        for f in files:
                            if f.endswith('.json'):
                                fp = os.path.join(root, f)
                                try:
                                    data = json.load(open(fp, 'r'))
                                    metrics_aggregate[f] = data
                                except Exception:
                                    pass
                else:
                    if src.endswith('.json'):
                        try:
                            data = json.load(open(src, 'r'))
                            metrics_aggregate[os.path.basename(src)] = data
                        except Exception:
                            pass

            except Exception as e:
                print(f"[PROJECT] Warning: failed to export {src}: {e}")

        # Write summary according to requested format
        summary = {
            'project': getattr(conf.common, 'project_name', None),
            'project_path': proj_root,
            'exported_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'artifacts': artifacts,
            'metrics_aggregate_keys': list(metrics_aggregate.keys())
        }

        summary_path = os.path.join(out_dir, f'summary.{fmt}')

        try:
            if fmt == 'json':
                out_obj = {'summary': summary, 'metrics': metrics_aggregate}
                json.dump(out_obj, open(summary_path, 'w+'), indent=2)
                print(f"[PROJECT] Export summary written to {summary_path}")

            elif fmt == 'yaml':
                try:
                    import yaml
                    out_obj = {'summary': summary,
                               'metrics': metrics_aggregate}
                    yaml.safe_dump(out_obj, open(
                        summary_path, 'w+'), sort_keys=False)
                    print(
                        f"[PROJECT] Export summary written to {summary_path}")
                except Exception:
                    # fallback to json if yaml not available
                    out_obj = {'summary': summary,
                               'metrics': metrics_aggregate}
                    json.dump(out_obj, open(os.path.join(
                        out_dir, 'summary.json'), 'w+'), indent=2)
                    print(
                        f"[PROJECT] PyYAML not available; wrote JSON summary to summary.json")

            elif fmt == 'csv':
                # For CSV, if metrics_aggregate contains dicts/lists, write a simple key,value CSV
                csv_path = os.path.join(out_dir, 'summary.csv')
                import csv
                with open(csv_path, 'w+', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['key', 'value'])
                    # write summary metadata
                    writer.writerow(['project', summary.get('project')])
                    writer.writerow(
                        ['project_path', summary.get('project_path')])
                    writer.writerow(
                        ['exported_at', summary.get('exported_at')])
                    writer.writerow(['artifacts_count', len(artifacts)])
                    # write metrics keys
                    for k, v in metrics_aggregate.items():
                        try:
                            writer.writerow([k, json.dumps(v)])
                        except Exception:
                            writer.writerow([k, str(v)])
                print(f"[PROJECT] Export summary written to {csv_path}")

            else:
                print(
                    f"[PROJECT] Unknown export format '{fmt}'; supported: json,csv,yaml")

        except Exception as e:
            print(f"[PROJECT] Error while writing export summary: {e}")

        print(f"[PROJECT] Export completed to {out_dir}")

    def run_transferring_process(self):
        """Run the transferring process of the project."""
        self.transferring_process = TransferringProcess(
            self.configuration, self.environment_api, joint_policy=None)
        self.transferring_process.start()
        print("Process transferring started independently.")

    def run_transferring(self, args):
        """Handle the CLI 'deploy' command.

        Supports mutually exclusive --direct and --remote, optional --checkpoint
        (path to joint policy folder) and --api (target environment API URL).
        This method updates configuration where appropriate, instantiates the
        EnvironmentAPI (from provided --api or from project transferring folder)
        and starts the transferring process which will perform deployment.
        """

        # Load project/config if provided
        project_path = getattr(args, "project", None)
        config_file = getattr(args, "config", None)
        if project_path:
            if config_file:
                candidate = config_file if os.path.isabs(
                    config_file) else os.path.join(project_path, config_file)
                if os.path.exists(candidate):
                    self.configuration = Configuration.from_json(candidate)
                else:
                    self.load(project_path)
            else:
                self.load(project_path)

        # Determine deploy mode
        if getattr(args, "direct", False) and getattr(args, "remote", False):
            raise ValueError(
                "Please specify only one of --direct or --remote for deploy.")
        if getattr(args, "direct", False):
            deploy_mode = "DIRECT"
        elif getattr(args, "remote", False):
            deploy_mode = "REMOTE"
        else:
            raise ValueError(
                "Please specify one of --direct or --remote for deploy.")

        # Check checkpoint arg
        checkpoint = getattr(args, "checkpoint", None)
        if checkpoint:
            # If a relative path given, make it absolute relative to project
            if not os.path.isabs(checkpoint):
                candidate = os.path.join(
                    self.configuration.common.project_path, checkpoint)
            else:
                candidate = checkpoint
            if os.path.exists(candidate):
                # Update configuration to point to this joint_policy folder
                try:
                    self.configuration.training.joint_policy = candidate
                    print(f"[PROJECT] Using checkpoint from {candidate}")
                except Exception:
                    # best-effort: attach dynamic attribute
                    setattr(self.configuration.training,
                            "joint_policy", candidate)
            else:
                print(
                    f"[PROJECT] Warning: checkpoint path {candidate} does not exist; continuing.")

        # Prepare EnvironmentAPI instance
        api_url = getattr(args, "api", None)
        if api_url:
            try:
                # If the project's environment_api module accepts an URL, prefer that
                try:
                    self.environment_api = EnvironmentAPI(api_url)
                except TypeError:
                    # Fallback: try loading the project's environment_api module and instantiate without args
                    env_api_path = os.path.join(
                        self.configuration.common.project_path, "transferring", "environment_api.py")
                    spec = importlib.util.spec_from_file_location(
                        "EnvironmentAPI", env_api_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "EnvironmentAPI"):
                        self.environment_api = getattr(
                            module, "EnvironmentAPI")(api_url)
                    else:
                        raise ImportError(
                            f"No EnvironmentAPI class found in {env_api_path}")
                print(
                    f"[PROJECT] Environment API instantiated with URL: {api_url}")
            except Exception as e:
                print(
                    f"[PROJECT] Warning: failed to instantiate EnvironmentAPI with URL '{api_url}': {e}")
                # Try to load default module
                env_api_path = os.path.join(
                    self.configuration.common.project_path, "transferring", "environment_api.py")
                if os.path.exists(env_api_path):
                    spec = importlib.util.spec_from_file_location(
                        "EnvironmentAPI", env_api_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "EnvironmentAPI"):
                        self.environment_api = getattr(
                            module, "EnvironmentAPI")()
                else:
                    raise FileNotFoundError(
                        f"Environment API not found at {env_api_path}")
        else:
            # Load EnvironmentAPI from project transferring folder
            env_api_path = os.path.join(
                self.configuration.common.project_path, "transferring", "environment_api.py")
            if not os.path.exists(env_api_path):
                raise FileNotFoundError(
                    f"Environment API not found at {env_api_path}")
            spec = importlib.util.spec_from_file_location(
                "EnvironmentAPI", env_api_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "EnvironmentAPI"):
                # Try to instantiate with no args
                try:
                    self.environment_api = getattr(module, "EnvironmentAPI")()
                except TypeError:
                    # If the EnvironmentAPI constructor requires an URL and none provided, instantiate with None
                    self.environment_api = getattr(
                        module, "EnvironmentAPI")(None)
            else:
                raise ImportError(
                    f"No EnvironmentAPI class found in {env_api_path}")

        # Update configuration deploy_mode
        try:
            self.configuration.transferring.deploy_mode = deploy_mode
        except Exception:
            setattr(self.configuration.transferring,
                    "deploy_mode", deploy_mode)

        print(f"[PROJECT] Starting deploying in mode {deploy_mode}...")

        # Start transferring process (it will handle direct vs remote flows)
        self.run_transferring_process()

        # If user invoked direct mode and only wanted a one-shot deploy, we may wait a short while
        if deploy_mode == "DIRECT":
            print("[PROJECT] Direct deploy initiated; transferring process will perform the deployment and continue running.")
        else:
            print("[PROJECT] Remote deploy started; transferring process will collect data and spawn MTA workers as needed.")

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
