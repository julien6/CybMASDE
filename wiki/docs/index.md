# Welcome to the CybMASDE Documentation

**CybMASDE** (Cyber Multi-Agent System Development Environment) is a modular platform designed to **model, train, analyze, and deploy intelligent multi-agent systems (MAS)** based on the **MAMAD methodology** (MOISE+MARL Assisted MAS Design).  
It combines **performance** from multi-agent reinforcement learning (MARL) and **organizational modeling (MOISE+)** for controlling and explaining emerging agents' behaviour with **practical tools** for experimentation, transfer, and explainability in complex cyber-physical environments.

CybMASDE can be used:

* 🧠 **As a research framework**, to study agent autonomy, coordination, adaptation, and explainability.  
* ⚙️ **As an engineering tool**, to design and deploy agent-based behaviors in real or simulated infrastructures.  
* 💻 **Through three interfaces**:

  + a **CLI (Command Line Interface)** for automation and batch workflows, 

  + a **Python library** for direct integration into experiments, 

  + and a **Graphical User Interface (GUI)** built with Angular for visual project editing and monitoring.

---

## Table of Contents

* [Introduction](introduction.md)
* [Installation](installation.md)
* [Getting Started](getting-started.md)
* [Architecture](architecture.md)
* [CLI API Reference](cli_reference.md)
* [GUI Reference](gui_reference.md)
* [Library API Reference](lib_reference.md)
* [Contributing](contributing.md)
* [Changelog](changelog.md)
* [FAQ](faq.md)
* [Glossary](glossary.md)

---

## Overview

CybMASDE was designed to **bridge the gap between research prototypes and operational systems** by providing:

* A **structured workflow** for the lifecycle of multi-agent systems (from simulation to deployment).  
* An **organizationally-aware reinforcement learning engine** (MOISE+MARL).  
* Integrated support for **world models**, **multi-agent policy training**, and **automatic explainability (Auto-TEMM)**.  
* A **transfer component** to synchronize simulated and real environments (via REST APIs).  

Each phase of the MAMAD method is implemented as a separate module:

1. **Modeling**: world model training or handcrafted environment definition.  
2. **Training**: organizationally guided MARL policy optimization.  
3. **Analyzing**: behavioral explainability and stability assessment.  
4. **Transferring**: policy deployment and iterative refinement in the target environment.

---

## Example Workflow

A typical workflow using the CLI may look like this:

```bash

# Create and validate a new project

cybmasde init -n overcooked_project --template worldmodel
cybmasde validate

# Run the full MAMAD pipeline automatically

cybmasde run --full-auto --reward-threshold 3.5 --max-refine 5

# Deploy the learned joint policy remotely

cybmasde deploy --remote --api http://localhost:8080/api
```
