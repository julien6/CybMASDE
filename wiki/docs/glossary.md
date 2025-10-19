# Glossary - CybMASDE

This glossary provides short definitions of terms used across the CybMASDE documentation. For detailed information and examples, consult the linked wiki pages.

## General Terms

### API

Application Programming Interface, a contract that lets components (or applications) exchange data and commands. See also the CLI and the Library API: [cli_reference](cli_reference.md), [lib_reference](lib_reference.md).

### Dependency

A library or package required by a project module. See the installation procedure: [installation](installation.md).

### Virtual Environment

Isolated Python environment (virtualenv/venv) used to manage dependencies without affecting the system Python. See manual installation: [installation](installation.md).

### Workflow

A sequence of automated (CI/CD) or operational steps. Example automation in GitHub Actions: [faq](faq.md) (CI section).

### MkDocs

A static site generator used for the documentation: [mkdocs](https://www.mkdocs.org/).

### GitHub Actions

CI/CD service to automate tests and deployments. See FAQ and contribution guide: [faq](faq.md), [contributing](contributing.md).

## CybMASDE-Specific Concepts

### MAMAD

Core methodology of CybMASDE: Modeling, Analyzing, Monitoring/Adapting, Deploying. Overview and context: [architecture](architecture.md).

### MOISE+

Organizational formalism used to constrain and explain multi‑agent behavior. Referenced in architecture and training docs: [architecture](architecture.md).

### MARL (Multi‑Agent Reinforcement Learning)

Family of reinforcement learning algorithms for multi‑agent systems (examples: MAPPO, MADDPG, QMIX, ROMA). See Training section: [architecture](architecture.md).

### World Model

Learned model (latent dynamics, representations) that approximates the environment to accelerate or stabilize training. See library reference: [lib_reference](lib_reference.md).

### Joint Policy

A shared or coordinated strategy among agents resulting from multi‑agent training. Examples and export options: [cli_reference](cli_reference.md).

### Auto‑TEMM / TEMM

Analysis and explainability methods used to infer and explain organizational structures and behaviors. See Library/API reference: [lib_reference](lib_reference.md).

### Transferring

Deployment and synchronization of trained policies to real or remote environments (via REST API). See usage and FAQ: [getting-started](getting-started.md), [faq](faq.md).

### Refining

Post‑deployment improvement iterations (retraining or adjusting configuration based on analysis). See Next Steps: [getting-started](getting-started.md).

## Interfaces & Tools

### CLI ( `cybmasde` )

Command-line interface to orchestrate the pipeline (init, validate, run, deploy, export...). Full CLI reference: [cli_reference](cli_reference.md). The CLI entrypoint is documented throughout the CLI reference pages.

### GUI (Angular)

Angular-based graphical interface for creating/editing projects and visualizing metrics. GUI guide: [gui_reference](gui_reference.md).

### Python API

Programmatic usage via the Python library (example: [ `CybMASDEProject` ](getting-started.md)). See the Getting Started and Library API pages: [getting-started](getting-started.md), [lib_reference](lib_reference.md).

### REST API / IPC

Communication modes between backend ↔ frontend or backend ↔ external environments:

* REST API for network exchanges (remote deployment).
* IPC for local integration (Electron). Architecture details: [architecture](architecture.md).

## Algorithms & Models Mentioned

* VAE, RNN... Architectures for learned dynamic representations (world models). See: [lib_reference](lib_reference.md).  
* MAPPO, MADDPG, QMIX, ROMA... Example MARL algorithms mentioned in the docs: [architecture](architecture.md).

## Configuration Files & Schemas

### project_configuration.json

Main project configuration file describing project steps, paths, and parameters. Usage reference: [getting-started](getting-started.md).

### label_manager.py

Optional script to manage labels/custom project hooks (referenced by the GUI docs): [gui_reference](gui_reference.md).
