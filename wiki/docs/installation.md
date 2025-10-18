# Installation Guide

CybMASDE can be installed on Linux (and optionally _Windows 10_ as far as we know) using the provided installation script.  
The process automatically installs all required backend and frontend dependencies, creates the recommended environment structure, and prepares the platform for CLI or GUI use.

---

## 🧰 Prerequisites

Before installing CybMASDE, make sure the following tools are available on your system:

* **Git** (for cloning the repository)
* **Python 3.8**  
* **pip** (Python package manager)
* **node** (runtime environment for running JavaScript on the server side)
* **npm** (package manager for managing dependencies and third-party libraries for Node. js)

You can check if these are already installed:

```bash
git --version
python --version
pip --version
```

---

## 🚀 Quick Installation

Clone the CybMASDE repository and run the installation script:

```bash
# Clone the repository
git clone https://github.com/julien6/CybMASDE.git
cd CybMASDE

# Run the installation script
./install.sh
```

The script will automatically:

* Create or activate a virtual environment (`env` in the `backend` folder), 
* Install all Python dependencies from `requirements.txt`, 
* Install the _npm_ packages required for the graphical frontend, 
* Set up the command-line entry point (`cybmasde`), 
* Verify that the environment variables and paths are configured correctly.

---

## 🧩 Optional: Manual Installation (Advanced Users)

If you prefer to install manually, you can do something like:

```bash
git clone https://github.com/julien6/CybMASDE.git
cd CybMASDE/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To enable the CLI globally (optional):

```bash
pip install -e .
```

---

## 🧠 Post-Installation Check

Verify that CybMASDE is correctly installed by running:

```bash
cybmasde --version
```

You should see an output similar to:

```
CybMASDE_1.0.0
```

You can also check the available commands:

```bash
cybmasde --help
```

---

## 🖥️ Using the Graphical Interface (Optional)

If you want to launch the **Angular-based GUI**, navigate to the GUI folder and start it.\
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

A native-like Electron window should pop up.
Otherwise, you should check logs at opening your browser at:
👉 `http://localhost:4200`

---

## 🧯 Troubleshooting

| Issue                               | Possible Fix                                                                       |
| ----------------------------------- | ---------------------------------------------------------------------------------- |
| `Permission denied: install.sh` | Run `chmod +x install.sh` before executing.                                        |
| `ModuleNotFoundError` after install | Ensure you are using the correct Python environment ( `source .venv/bin/activate` ). |
| GUI doesn’t launch                  | Make sure Node.js ≥ 18 is installed ( `node -v` ).                                   |

---

## ✅ Next Steps

Once installed, you can proceed to:

* [Getting Started](getting-started.md): Learn how to create and run your first CybMASDE project.
* [CLI API Reference](cli_reference.md): Explore the CLI, configuration files, and GUI options.
* [GUI Reference](gui_reference.md) - Learn how about the graphical usage with the provided GUI.
* [Library API Reference](lib_reference.md) - Integrate CybMASDE library functionalities into your own research workflows.
* [Architecture](architecture.md): Understand how the MAMAD workflow is implemented.
