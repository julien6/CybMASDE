# Frequently Asked Questions (FAQ)

This section collects common questions about installing, using, and extending **CybMASDE**.  
If you don’t find your answer here, feel free to open an [issue on GitHub](https://github.com/julien6/CybMASDE/issues).

---

## 🧰 Installation and Setup

### **Q1. What operating systems are supported?**

CybMASDE is primarily developed and tested on **Linux (Ubuntu 22.04+)**, but it can also run on:
* **macOS** (with Python 3.8)
* **Windows 10/11** (through WSL2 or native installation)

The graphical frontend (Angular) is fully cross-platform.

---

### **Q2. How do I install CybMASDE?**

Simply clone the repository and run the provided script:

```bash
git clone https://github.com/julien6/CybMASDE.git
cd CybMASDE
./install.sh
```

It will automatically install all backend (Python) and frontend (Node.js) dependencies.

See the [Installation Guide](installation.md) for details.

---

### **Q3. The install script fails with “Permission denied”. What should I do?**

Make the script executable first:

```bash
chmod +x install.sh
./install.sh
```

If the problem persists, ensure your Python and Node versions meet the requirements:

* **Python ≥ 3.8**
* **Node.js ≥ 18**

---

### **Q4. How can I verify my installation?**

Run:

```bash
cybmasde --version
cybmasde --help
```

If both commands work, your installation is successful.

---

## ⚙️ Project Management

### **Q5. How do I create a new project?**

Use the `init` command:

```bash
cybmasde init -n my_project -d "My first CybMASDE project"
```

You can also choose a template (e.g., `--template worldmodel` ).

This creates a complete project structure with a default configuration file ( `project_configuration.json` ).

You can aslo choose to create a new project using the GUI interface via `File -> New Project`

---

### **Q6. What is the purpose of the `project_configuration.json` file?**

It defines **all parameters** of your project:

* Environment paths
* Model and training hyperparameters
* Analysis and transfer settings

Every command ( `run` , `train` , `analyze` , etc.) reads this configuration to ensure reproducibility.

See [Architecture](architecture.md) for a breakdown of this structure.

---

### **Q7. How do I check if my configuration is valid?**

Run:

```bash
cybmasde validate
```

The validator will check:

* File existence and format consistency
* JSON schema compliance
* Hyperparameter structure validity

If something’s wrong, the CLI will display detailed errors with line numbers.

---

## 🧠 Learning and Analysis

### **Q8. What algorithms are supported for training?**

CybMASDE integrates several Multi-Agent Reinforcement Learning (MARL) algorithms, including:

* **MAPPO**
* **MADDPG**
* **QMIX**
* **ROMA**
* **VDN**
* **IQL**

Each can be selected via the CLI or configuration file.

---

### **Q9. What is the MOISE+MARL approach?**

It combines **organizational modeling (MOISE+)** and **multi-agent reinforcement learning (MARL)**.
Agents learn while being constrained by organizational rules — roles, missions, and norms — allowing structured cooperation instead of pure self-optimization.

This is one of CybMASDE’s key theoretical innovations.
Learn more in the [Introduction](introduction.md).

---

### **Q10. How does the analysis phase work?**

CybMASDE’s **Analyzing** module uses the **Auto-TEMM** (Trajectory-Explained Multi-agent Modeling) process to:

* Cluster agent trajectories
* Extract organizational patterns
* Compute explicability metrics (SOF, FOF, OF)
* Visualize behavioral similarities and divergences

Results are saved under `/analyzing/figures/` and `/analyzing/statistics.json` .

---

## 🔄 Refinement and Deployment

### **Q11. What is the difference between `run` and `refine` ?**

* `cybmasde run` executes the entire MAMAD pipeline (Model-Train-Analyze-Transfer-Refine).
* `cybmasde refine` only executes additional refinement loops (analysis + retraining) after initial runs.

Refinement cycles continue until convergence or user interruption.

---

### **Q12. How can I deploy trained agents in a real environment?**

Use the `deploy` command:

```bash
cybmasde deploy --remote --api http://localhost:8080/api
```

Deployment can happen in two modes:

* **DIRECT:** agents execute policies locally
* **REMOTE:** CybMASDE executes policies and sends actions to agents via an API

See [CLI API Reference](cli_reference.md), [GUI Reference](gui_reference.md) or [Library API Reference](lib_reference.md) for details.

---

## 🧩 Development and Customization

### **Q13. Can I add new MARL algorithms?**

Yes.
All algorithms are implemented as subclasses of `BaseTrainer` .
To add one:

1. Implement a new trainer in `/training/algorithms/`
2. Register it in `training/algorithm_registry.py`
3. Add your configuration schema under `training/hyperparameters.json`

CybMASDE will automatically detect it at runtime.

---

### **Q14. Can I use my own environment?**

Absolutely.
You can provide your own environment either by:

* Implementing it manually (`modelling/handcrafted_environment.py`), or
* Using traces to generate a **World Model** (Autoencoder + RNN).

Your environment must follow the **PettingZoo API** standard ( `reset()` , `step()` , `render()` ).

---

### **Q15. How can I integrate CybMASDE into a Python script or notebook?**

You can import it directly as a library:

```python
from cybmasde import CybMASDEProject

project = CybMASDEProject("my_project/project_configuration.json")
project.validate()
project.run(full_auto=True)
```

This allows integration with external workflows (e.g., Ray Tune, WandB, or custom dashboards).

---

## 🖥️ GUI and Visualization

### **Q16. How do I start the graphical interface?**

You first need your environmental to be running.

Then, from the main directory:

In a first tab, run the backend server

```bash
cd backend
source env/bin/activate
cd api_server
python server.py
```

In a second tab, run the frontend

```bash
cd frontend
npm run start
```

Then a native-like Electron window should appear.
---

### **Q17. What can I do from the GUI?**

* Edit project configurations visually
* Launch and monitor training runs
* View Auto-TEMM figures and statistics
* Validate configuration files interactively

It provides a simplified version of all CLI functionalities.

---

## 🧩 Troubleshooting

### **Q18. My CLI commands are not recognized.**

Ensure that the `cybmasde` executable is installed in your PATH.
If not, reinstall in editable mode:

```bash
pip install -e .
```

---

### **Q19. The GUI doesn’t load after `npm run start` .**

Check that:

* Node.js ≥ 18 is installed (`node -v`)
* Port 4200 is not already in use
* The backend is installed and accessible

---

### **Q20. The analysis phase fails with “missing module sklearn”.**

Install missing dependencies manually:

```bash
pip install scikit-learn matplotlib pandas
```

---

## 🧭 General

### **Q21. What’s the difference between the CLI, GUI, and Python API?**

| Interface      | Best For                                | Description                                                      |
| -------------- | --------------------------------------- | ---------------------------------------------------------------- |
| **CLI**        | Automation, batch runs, reproducibility | Command-line orchestration of the MAMAD pipeline                 |
| **GUI**        | Interactive use, visualization          | Angular web interface for project creation and analysis          |
| **Python API** | Research and scripting                  | Direct use of CybMASDE components within custom Python workflows |

All three share the same backend logic and configuration schema.

---

### **Q22. Where are logs and results stored?**

Each project maintains its own folders:

```
/training/statistics.json
/analyzing/figures/
/transferring/configuration.json
/refining/logs/
```

Logs are stored in `/logs/` inside the project directory.

---

### **Q23. Is CybMASDE open source?**

Yes.
The full source code is available on GitHub:
👉 [https://github.com/julien6/CybMASDE](https://github.com/julien6/CybMASDE)

The project is licensed under **MIT** unless otherwise stated.

---

### **Q24. Who maintains CybMASDE?**

CybMASDE was developed by **Julien Soulé** as part of a PhD project between:

* **Université Grenoble Alpes - LCIS Laboratory**, and
* **Thales LAS - La Ruche (Rennes, France)**

It continues to evolve as an open research and development platform for **multi-agent systems and cyber-defense**.

---

### **Q25. Where can I learn more about the theory behind CybMASDE?**

The theoretical foundations are detailed in:

* The associated **PhD manuscript**, available upon request.
* The **MAMAD methodology** documentation (see [Architecture](architecture.md)).
* The **MOISE+MARL** papers referenced in the bibliography.
