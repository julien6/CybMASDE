# Getting Started

Welcome to **CybMASDE**!  
This section will guide you through the creation, validation, and execution of your first CybMASDE project whether you use it through the **CLI**, the **Python API**, or the **Web GUI**.

CybMASDE projects are fully self-contained and reproducible. Each project includes its own configuration file ( `project_configuration.json` ), environment definitions, and analysis outputs.  

---

## 🧱 1. Create a New Project

To create a new project, use the `init` command:

```bash
cybmasde init -n my_first_project -d "Exploring multi-agent learning in Overcooked-AI"
```

By default, this will:

* Create a new project directory (`./my_first_project/`)
* Generate a complete **project structure**
* Initialize a **default configuration file** (`project_configuration.json`)

You can also specify an environment template using the `--template` option:

```bash
cybmasde init -n overcooked_test --template worldmodel
```

Available templates:

| Template      | Description                                                    |
| ------------- | -------------------------------------------------------------- |
| `handcrafted` | Starts from a manually coded environment (e.g. Overcooked-AI). |
| `worldmodel` | Uses a latent world model (Autoencoder + RNN).                 |
| `minimal` | Creates a lightweight structure for quick testing.             |

---

## 🧪 2. Validate the Project

Before running anything, it’s a good idea to **validate** the configuration file and dependencies:

```bash
cd my_first_project
cybmasde validate
```

This command checks that:

* All required files and paths exist, 
* JSON structures follow the official schema, 
* Model and analysis parameters are consistent.

Example output:

```
[VALIDATION] Checking configuration...
[OK] Common parameters verified.
[OK] Modeling environment accessible.
[OK] Training hyperparameters valid.
[OK] Analysis module initialized.
✅ Project 'my_first_project' is valid and ready.
```

If you prefer a quiet validation mode:

```bash
cybmasde validate -q
```

---

## ⚙️ 3. Run the Pipeline

Once validated, you can execute the full MAMAD workflow:

```bash
cybmasde run --full-auto
```

This command runs, in sequence:

1. **Modeling**: build or load the environment and world model
2. **Training**: learn agent policies under MOISE+ constraints
3. **Analyzing**: evaluate and interpret the trained agents
4. **Transferring**: deploy policies in a real or simulated system
5. **Refining**: iterate based on analysis results

Each step logs its progress in the project directory under `/logs/` .

To control execution manually (e.g., pause between stages):

```bash
cybmasde run --semi-auto
```

Or to execute each stage separately:

```bash
cybmasde model --auto
cybmasde train --algo MAPPO
cybmasde analyze --auto-temm
```

---

## 🔍 4. Inspect and Analyze the Results

After training, results are stored inside your project folder, typically under:

```
/training/statistics.json
/analyzing/figures/
/analyzing/statistics.json
```

To visualize trajectories, metrics, and organizational compliance, you can use the built-in **Auto-TEMM** analyzer:

```bash
cybmasde analyze --auto-temm
```

Or inspect the generated figures manually in the GUI (see below).

---

## 🖥️ 5. Use the Web Interface (Optional)

CybMASDE also provides a **visual interface** based on Angular for easier project management.

To launch the Web Interface

Make sure your environment is running, then from the main project directory open two terminal tabs.

Tab 1 - backend server:

```bash
cd backend
source env/bin/activate      # POSIX; use env\Scripts\activate on Windows
cd api_server
python server.py
```

Tab 2 - frontend:

```bash
cd frontend
npm run start
```

A native-like Electron window should appear. If nothing appears, check the backend/frontend logs or open your browser at: http://localhost:4200

Then, an native-like desktop Electron window should appears.\
If nothing appears, you better check logs at opening your browser at: [http://localhost:4200](http://localhost:4200)

From there, you can:

* Create and edit project configurations visually, 
* Launch and monitor the transferring and mta processes, 
* Inspect Auto-TEMM analysis results interactively.
* ...

---

## 🧠 6. (Optional) Use CybMASDE as a Python Library

If you want to embed CybMASDE into your own scripts or notebooks:

```python
from cybmasde import CybMASDEProject

project = CybMASDEProject("my_first_project/project_configuration.json")
project.validate()
project.run(full_auto=True)
```

You can also directly invoke submodules:

```python
from cybmasde.training import Trainer
trainer = Trainer(config_path="training/hyperparameters.json")
trainer.run()
```

---

## 📂 7. Typical Project Structure

Once created, your project will look like this:

```
my_first_project/
│
├── project_configuration.json
├── common/
├── modelling/
│   ├── handcrafted_environment.py
│   └── generated_environment/
│       ├── world_model/
│       └── component_functions.py
│
├── training/
│   ├── hyperparameters.json
│   ├── statistics.json
│   └── joint_policy/
│
├── analyzing/
│   ├── figures/
│   ├── statistics.json
│   └── inferred_organizational_specifications/
│
├── transferring/
│   └── configuration.json
│
└── refining/
    └── refinement_logs/
```

---

## 🧩 8. Next Steps

Congratulations, you’ve successfully created and executed your first CybMASDE project! 🎉

You can now explore:

* [CLI API Reference](cli_reference.md) - Learn about all CLI commands and options.
* [GUI Reference](gui_reference.md) - Learn how about the graphical usage with the provided GUI.
* [Library API Reference](lib_reference.md) - Integrate CybMASDE library functionalities into your own research workflows.
* [Architecture](architecture.md) - Understand how CybMASDE components interact.
