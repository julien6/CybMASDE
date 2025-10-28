### **WARNING : This projet is a work in progress, onging changes are likely to break some of the functionalities**

# CybMASDE

**CybMASDE** (Cyber Multi-Agent System Design Environment) is a modular and extensible platform for the **design, training, analysis, and deployment of multi-agent systems (MAS)**.
It implements the **MAMAD** method (MOISE+MARL Assisted MAS Design), combining:

* **Modelling**: generation of simulated environments (via World Models or MCAS)
* **Training**: multi-agent learning constrained by organizational specifications (MOISE+MARL)
* **Analyzing**: inference and explainability of organizational structures (Auto-TEMM / TEMM)
* **Transferring**: deployment and synchronization with real environments

Its purpose is to produce **joint policies that are efficient, stable, and explainable**, applicable both in **academic research** and **industrial contexts** (with a strong focus on cyber-defense).

For complete documentation (including installation steps, tutorials, CLI/API references, architecture diagrams, example projects, and contribution guidelines) please consult the **CybMASDE Wiki** at https://julien6.github.io/CybMASDE/. This site is the canonical reference for users and contributors.

---

## 🚀 Key Features

* Structured project creation and validation (`init`,         `validate`)
* Automatic environment modelling via **World Models (VAE, LSTM, JOPM)** or manual modelling via **MCAS**
* Multi-agent training with **MARLlib + Ray RLlib** (MAPPO, MADDPG, QMIX, etc.)
* Native integration of **MOISE+MARL organizational constraints** (action masking, reward shaping)
* Organizational analysis with **Auto-TEMM**: trajectory clustering, role/goal extraction, organizational metrics (SOF, FOF, OF)
* Automatic or manual refinement loops combining training and analysis
* Deployment into real environments, with two execution modes:

  + **DIRECT**: policy embedded in the agents
  + **REMOTE**: policy executed by CybMASDE, actions sent to agents via API
* Result tracking and export (logs, metrics, visualizations, organizational specifications in JSON/CSV/YAML)

---

## 🏗 Software Architecture

CybMASDE is organized around a **Python backend** orchestrated by a **REST API**, accessible via:

* a **unified CLI** (`cybmasde ...`) for automation (batch/HPC), 
* an **Angular-based GUI** for project configuration and visual monitoring.

**Core technologies:**

* **Python 3.8+ / Flask**: backend and REST API
* **PyTorch**: training World Models (VAE, LSTM)
* **MARLlib + Ray RLlib**: scalable multi-agent reinforcement learning
* **Optuna**: hyperparameter optimization
* **Angular**: modern and ergonomic GUI

---

## ⚙️ Installation

```bash
git clone https://github.com/julien6/CybMASDE.git
cd CybMASDE
```

To install everything at once

```bash
./install.sh
```

To install the backend (including server API and CLI)

```bash
cd frontend
./install.sh
```

Optional: to install the GUI frontend

```bash
cd frontend
./install.sh
```

---

## 📘 Quickstart

### Create and configure a project

```bash
cybmasde init -n overcooked_test --template worldmodel
cybmasde validate
```

### Run the full pipeline in fully automated mode

```bash
cybmasde run --full-auto \
  --project ./overcooked_test \
  --max-refine 10 \
  --reward-threshold 3.5 \
  --std-threshold 0.05 \
  --accept-inferred
```

### Deploy a policy into a real environment

```bash
cybmasde deploy --remote --api http://localhost:8080/api
```

### Export results

```bash
cybmasde export --format json --output ./results
```

---

## 📂 Project Structure

```
<project_name>/
│── project_configuration.json   # main configuration
│── modelling/                   # simulated environments + MCAS
│── training/                    # hyperparameters, checkpoints
│── analyzing/                   # metrics, visualizations, inferred specs
│── transferring/                # configuration and deployment
└── label_manager.py             # observation/action label mapping
```

---

## 📑 Use Cases

* **Academic research**: rapid prototyping of MARL environments, organizational explainability via Auto-TEMM
* **Cyber-defense**: integration with real infrastructures (via environment APIs) to evaluate resilience and adaptability
* **Industry**: deployment of robust and interpretable policies in distributed environments

---

## 📖 References

* Method **MAMAD**: MOISE+MARL Assisted MAS Design
* Organizational framework **MOISE+MARL API (MMA)**: [github.com/julien6/MOISE-MARL](https://github.com/julien6/MOISE-MARL)
* Libraries: [MARLlib](https://github.com/Replicable-MARL/MARLlib), [Ray RLlib](https://docs.ray.io), [Optuna](https://optuna.org)
* Associated PhD thesis: see `blop/docs`

---

## ✍️ Author

Developed in the context of a PhD in **Distributed AI for Cyber-Defense**, by [Julien Soule](https://julien6.github.io/home/).
