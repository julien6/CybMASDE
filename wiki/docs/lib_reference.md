# CybMASDE - Python Library Reference

This page provides a **detailed API reference** for CybMASDE’s Python library.  
It summarizes the main programmatic entry points available under the backend modules, along with examples and best practices for integrating CybMASDE into scripts or research workflows.

---

## 📘 Overview

CybMASDE’s backend is organized under the following main namespaces:

| Module | Description |
|---------|-------------|
| `world_model.project` | Core orchestrator and entrypoint (Project class) |
| `world_model.mta_process` | Modeling–Training–Analyzing process |
| `world_model.transferring_process` | Deployment and transfer management |
| `world_model.environment_api` | Interface for remote or local environments |
| `world_model.jopm` , `vae_utils` , `rdlm_utils` | Models composing the Joint Observation Prediction Model (JOPM) |
| `world_model.component_functions` | Reward, done, and rendering logic |
| `world_model.project_configuration` | Data schema for configuration files |
| `api_server.server` | Web backend utilities for the Angular GUI |

---

> **Note**
>
> - Many methods accept an `argparse.Namespace` -like object (attributes matching CLI flags).  
> You can use `types.SimpleNamespace` for programmatic calls.
> - This reference was compiled from code inspection and docstrings; for full implementation details, consult the source files under:
> - `backend/world_model/world_model/`

> - `backend/api_server/`

---

## 🧩 Table of Contents

01. [Project (Core Orchestrator)](#project---core-orchestrator)  
02. [MTAProcess (Model–Train–Analyze)](#mtaprocess---modeltrainanalyze)  
03. [TransferringProcess (Deployment)](#transferringprocess---deployment)  
04. [EnvironmentAPI (Remote Environment Wrapper)](#environmentapi---remote-environment-wrapper)  
05. [World Models: JOPM / VAE / RDLM](#world-models-jopm--vae--rdlm)  
06. [Joint Policy Implementations](#joint-policy)  
07. [Component Functions](#componentfunctions)  
08. [Project Configuration Schema](#project-configuration-schema)  
09. [Organizational Specification Helpers](#organizational-specification-helper-scripts)  
10. [Server Utilities (Frontend/Backend Glue)](#server--frontend-helpers)  
11. [Usage Recipes](#quick-usage-recipes)

---

<a id="project---core-orchestrator"></a>

## 🧠 Project - Core Orchestrator

📁 `backend/world_model/world_model/project.py`

### Import

```python
from world_model.project import Project
```

### Description

The `Project` class orchestrates all phases of a CybMASDE pipeline:

**Modeling → Training → Analyzing → Transferring → Refining**

It can be used programmatically to automate workflows or reproduce CLI commands.

---

### **Constructor**

```python
Project(configuration)
```

* **configuration**: a parsed `Configuration` object, or `None` to load later.

Example:

```python
proj = Project(None)
```

---

### **save**

```python
save(self, file_path, name)
```

Save the project configuration and artifacts.

Example:

```python
proj.save("/tmp/my_project/project_configuration.json", name="my_project")
```

---

### **create_project**

```python
create_project(self, name, description, output, template)
```

Create a new project folder from a template (mirrors `cybmasde init` ).

Example:

```python
proj.create_project("demo", "Demo project", output="/tmp/demo", template="handcrafted")
```

---

### **load**

```python
load(self, file_path)
```

Load project configuration from a JSON file.

Example:

```python
proj.load("/path/to/project/project_configuration.json")
```

---

### **validate**

```python
validate(self, args)
```

Validate configuration consistency.

Example:

```python
from types import SimpleNamespace
proj.validate(SimpleNamespace(quiet=False, strict=True))
```

---

### **from_dict**

```python
from_dict(cls, data)
```

Construct a `Project` from a configuration dictionary.

Example:

```python
proj = Project.from_dict(my_config_dict)
```

---

### **run / run2**

```python
run(self, args)
run2(self, args)
```

Execute the entire pipeline ( `cybmasde run` ).
Handles CLI arguments for automated or interactive runs.

Example:

```python
args = SimpleNamespace(full_auto=True, max_refine=3)
proj.run(args)
```

---

### **run_modeling**

```python
run_modeling(self, args)
```

Launch the modeling activity (auto or manual mode).

Example:

```python
proj.run_modeling(SimpleNamespace(auto=True, vae_dim=32, lstm_hidden=64))
```

---

### **run_training**

```python
run_training(self, args)
```

Train agent policies under organizational constraints.

Example:

```python
proj.run_training(SimpleNamespace(algo="MAPPO", batch_size=64, lr=1e-4, gamma=0.99, epochs=100))
```

---

### **run_analyzing**

```python
run_analyzing(self, args)
```

Run Auto-TEMM analysis and generate figures/statistics.

Example:

```python
proj.run_analyzing(SimpleNamespace(auto_temm=True, metrics=["reward","org_fit"]))
```

---

### **run_refining**

```python
run_refining(self, args)
```

Execute iterative refinement cycles.

Example:

```python
proj.run_refining(SimpleNamespace(max=3, interactive=True))
```

---

### **display**

```python
display(self, args)
```

Display concise project status (for `cybmasde status` ).

---

### **cleanup**

```python
cleanup(self, args)
```

Clean temporary files and checkpoints.

Example:

```python
proj.cleanup(SimpleNamespace(all=True))
```

---

### **export**

```python
export(self, args)
```

Export results and metrics to a chosen format.

Example:

```python
proj.export(SimpleNamespace(format="json", output="./exports"))
```

---

### **run_transferring**

```python
run_transferring(self, args)
```

Deploy policies and manage trajectory collection.

Example:

```python
proj.run_transferring(SimpleNamespace(remote=True, api="http://localhost:8080/api"))
```

---

### **stop**

```python
stop(self)
```

Stop all running background processes cleanly.

---

### 🧩 Typical Programmatic Workflow

```python
from types import SimpleNamespace
proj = Project(None)
proj.create_project("demo", "desc", output="/tmp/demo", template="handcrafted")
proj.load("/tmp/demo/project_configuration.json")
proj.validate(SimpleNamespace(quiet=False, strict=False))
proj.run_training(SimpleNamespace(algo="MAPPO", epochs=5))
```

---

<a id="mtaprocess---modeltrainanalyze"></a>

## ⚙️ MTAProcess - Model/Train/Analyze

📁 `backend/world_model/world_model/mta_process.py`

### Main Classes

| Class            | Purpose                                                 |
| ---------------- | ------------------------------------------------------- |
| `MTAProcess` | Main process handling modeling, training, and analyzing |
| `MeanStdStopper` | Stop condition helper for iterative refinement          |

---

### Example

```python
from world_model.mta_process import MTAProcess
mta = MTAProcess(configuration, componentFunctions, transferring_pid=None)
mta.run()
```

### Key Methods

* `run()`:  Launch the MTA loop
* `run_modelling_activity()`:  Execute environment/world model generation
* `run_training_activity()`:  Execute agent training
* `run_analyzing_activity()`:  Execute Auto-TEMM analysis
* `run_autoencoder_with_hpo()` / `run_rdlm_with_hpo()`:  Hyperparameter optimization
* `load_traces(path)`:  Load trajectories
* `get_joint_observations(histories)`:  Extract joint observations
* `generate_organizational_model(...)`:  Build inferred org specs

---

<a id="transferringprocess---deployment"></a>

## 🚀 TransferringProcess - Deployment

📁 `backend/world_model/world_model/transferring_process.py`

```python
from world_model.transferring_process import TransferringProcess
tp = TransferringProcess(configuration, environment_api, joint_policy)
tp.run()
```

### Key Methods

| Method                         | Description                       |
| ------------------------------ | --------------------------------- |
| `run()` | Execute deployment loop           |
| `load_component_functions()` | Load reward/rendering logic       |
| `create_random_joint_policy()` | Fallback random policy            |
| `load_latest_joint_policy()` | Restore last checkpoint           |
| `save_joint_histories()` | Store replayed trajectories       |
| `create_mta_process()` | Trigger MTA when new data arrives |

---

<a id="environmentapi---remote-environment-wrapper"></a>

## 🌍 EnvironmentAPI - Remote Environment Wrapper

📁 `backend/world_model/world_model/environment_api.py`

### Example

```python
from world_model.environment_api import EnvironmentAPI
env = EnvironmentAPI("http://localhost:8080/api")
obs = env.retrieve_joint_observation()
```

### Methods

| Method                         | Description                   |
| ------------------------------ | ----------------------------- |
| `agents()` | Return agent list and specs   |
| `retrieve_joint_observation()` | Get current joint observation |
| `retrieve_joint_histories()` | Retrieve trajectories         |
| `apply_joint_action(action)` | Apply joint action            |
| `deploy_joint_policy(policy)` | Deploy a policy remotely      |

---

<a id="world-models-jopm--vae--rdlm"></a>

## 🧮 World Models - JOPM, VAE, RDLM

### `WorldModelEnv`

Run simulated episodes combining the JOPM, component functions, and label manager.

```python
from world_model.world_model_env import WorldModelEnv
env = WorldModelEnv("/path/to/jopm.pkl", "/path/to/component_functions.py", "/path/to/label_manager.py")
obs = env.reset()
next_obs, reward, done, info = env.step({"agent_0": 0})
```

---

### `JOPM`

Handles joint observation prediction.

| Method                             | Description               |
| ---------------------------------- | ------------------------- |
| `save(file_path)` | Serialize JOPM            |
| `load(cls, file_path)` | Load JOPM                 |
| `predict_next_joint_observation()` | Forecast next observation |

---

### `VAE` and `RDLM`

Neural models used in the JOPM world model.

* `VAE` – Encodes observation data to a latent space
* `RDLM` – Models temporal dynamics in latent space

Utilities include:

* `train_vae()` and `vae_loss()`
* `rdlm_objective()` for optimization

---

<a id="joint-policy"></a>

## 🧭 Joint Policy

📁 `backend/world_model/world_model/joint_policy.py`

### Implementations

| Class                  | Description                     |
| ---------------------- | ------------------------------- |
| `joint_policy` | Base policy wrapper             |
| `random_joint_policy` | Produces random joint actions   |
| `marllib_joint_policy` | Loads MARLlib-compatible models |

Example:

```python
from world_model.joint_policy import random_joint_policy
policy = random_joint_policy(action_space, agents)
action = policy.next_action(joint_observation)
```

---

<a id="componentfunctions"></a>

## 🧩 ComponentFunctions

📁 `backend/world_model/world_model/component_functions.py`

### Example

```python
class ComponentFunctions:
    def reward_fn(self, obs, action, next_obs): ...
    def done_fn(self, obs, action, next_obs): ...
    def render_fn(self, obs, action, next_obs): ...
```

This module defines **domain-specific logic** for:

* Reward computation
* Termination conditions
* Visualization

---

<a id="project-configuration-schema"></a>

## 🧰 Project Configuration Schema

📁 `backend/world_model/world_model/project_configuration.py`

### Main Classes

| Class                                                            | Role                |
| ---------------------------------------------------------------- | ------------------- |
| `Configuration` | Root project schema |
| `Modelling` , `Training` , `Analyzing` , `Transferring` , `Refining` | Submodules          |
| `JOPM_conf` , `Autoencoder_conf` , `RDLM_conf` | Model definitions   |

Key methods:

* `from_json(path)`
* `serialize_configuration()`
* `to_dict()` / `to_json()`

Example:

```python
from world_model.project_configuration import Configuration
cfg = Configuration.from_json("project_configuration.json")
```

---

<a id="organizational-specification-helper-scripts"></a>

## 🧠 Organizational Specification Helper Scripts

Located under:

* `backend/world_model/world_model/organizational_specification_function.py`
* `project_example/training/organizational_specifications/`

These define heuristics to infer **roles**, **goals**, and **mission assignments** based on trajectories.

Example:

```python
def function_for_role_0(trajectory, observation, agent_name, label_manager):
    ...
```

---

<a id="server--frontend-helpers"></a>

## 🖥️ Server / Frontend Helpers

📁 `backend/api_server/server.py`

These utilities connect the backend with the Angular GUI for project creation, saving, and launching.

### Key Functions

| Function                                                | Description                           |
| ------------------------------------------------------- | ------------------------------------- |
| `add_a_recent_project(project_configuration)` | Register project in GUI               |
| `replace_json_paths(obj, project_path)` | Replace inlined JSONs with file paths |
| `restore_json_paths(obj, initial_config, project_path)` | Inverse of above                      |
| `load_project()` / `save_project()` | Load or persist GUI projects          |
| `save_and_run()` | Save project then trigger execution   |

---

<a id="quick-usage-recipes"></a>

## 🧩 Quick Usage Recipes

### 1️⃣ Programmatic Training Example

```python
from types import SimpleNamespace
from world_model.project import Project

proj = Project(None)
proj.create_project("demo", "desc", output="/tmp/demo", template="handcrafted")
proj.load("/tmp/demo/project_configuration.json")
proj.validate(SimpleNamespace(quiet=False, strict=False))
proj.run_training(SimpleNamespace(algo="MAPPO", epochs=10))
```

---

### 2️⃣ World Model Simulation Example

```python
from world_model.world_model_env import WorldModelEnv

env = WorldModelEnv("/path/to/jopm.pkl", "/path/to/component_functions.py", "/path/to/label_manager.py")
obs = env.reset()
action = {"agent_0": 0, "agent_1": 1}
next_obs, reward, done, info = env.step(action)
```
