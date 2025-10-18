# Introduction

**CybMASDE** (Cyber Multi-Agent System Design Environment) is a unified research and development platform for **designing, training, analyzing, and deploying intelligent multi-agent systems (MAS)**.  
It was originally developed as part of a doctoral research project between Université Grenoble Alpes (**[LCIS Laboratory](https://lcis.fr/)**) and **Thales LAS - La Ruche (Rennes, France)**, within the context of **cyber-defense and autonomous system coordination**.

CybMASDE provides both **theoretical grounding** and **practical tooling** to bridge the gap between **simulation-based AI research** and **operational, explainable, and deployable multi-agent systems**.

---

## 🎯 Objectives

The main objective of CybMASDE is to provide a **methodological and software framework** that supports the entire lifecycle of a multi-agent system (from conceptual modeling to real-world deployment).  
It integrates multiple AI paradigms under a unified structure called **MAMAD**:

> **MAMAD** (*MOISE+MARL Assisted MAS Design*)

This framework allows:
* The modeling of environments, world models, and organizations, 
* The training of agent policies using organizationally constrained MARL, 
* The analysis of learned behaviors and emergent dynamics (e.g., via explainability tools like **TEMM**), 
* The transfer of these behaviors into real or hybrid infrastructures, 
* And the iterative refinement of agent-organization configurations through feedback cycles.

---

## 🧩 Theoretical Foundations

CybMASDE is built on two key scientific paradigms:

1. **Multi-Agent Reinforcement Learning (MARL)**  
   Provides the adaptive intelligence layer where agents learn optimal behaviors via trial and error in cooperative, competitive, or mixed environments.

2. **MOISE+ Organizational Modeling**  
   Defines explicit organizational roles, missions, and constraints that guide agents’ decisions, ensuring *structured autonomy* and *coordinated behavior*.

By coupling **MOISE+** and **MARL**, CybMASDE enables a new research direction called **MOISE+MARL**, where *learning is guided by organizational norms* and *organizational structures evolve based on learned behaviors*.

---

## 🧠 Why CybMASDE?

Traditional multi-agent frameworks focus either on:
* **Engineering tools** (simulation engines, orchestration layers), or  
* **AI libraries** (learning algorithms, policy optimization).

CybMASDE combines both worlds, allowing users to:
* **Model** complex, multi-level environments (agents, organizations, infrastructures), 
* **Train** adaptive agents under explicit organizational constraints, 
* **Analyze** their behaviors through automated and explainable methods, 
* **Transfer** policies to real-world cyber-physical systems (via APIs or embedded agents), 
* And **Refine** the full system iteratively.

This makes CybMASDE a **comprehensive MAMAD-compliant pipeline** from conception to deployment.

---

## 🧮 Key Components

CybMASDE is composed of five core modules:

| Module | Purpose | Example Features |
|---------|----------|------------------|
| **Modeling** | Build or generate environments and world models. | Handcrafted environments, latent dynamics models (VAE, RNN, etc.) |
| **Training** | Optimize agent behaviors under MARL and MOISE+ constraints. | MAPPO, MADDPG, QMIX, ROMA, organizational reward shaping |
| **Analyzing** | Interpret, visualize, and explain agent behaviors. | Auto-TEMM, trajectory clustering, explainability metrics |
| **Transferring** | Deploy learned policies in real or hybrid environments. | REST API deployment, trajectory synchronization |
| **Refining** | Iterate based on analysis outcomes. | Feedback loops, re-training triggers, organizational adaptation |

Each component is fully configurable via a **project configuration file** ( `project_configuration.json` ), making the pipeline flexible for both research and applied contexts.

---

## 💡 Typical Use Cases

CybMASDE is designed to support a wide range of application domains:

* **Cyber-Defense**: autonomous intrusion detection and response coordination  
* **Swarm Robotics**: decentralized control and collective adaptation  
* **Industrial IoT**: resource optimization in distributed networks  
* **Microservice Management**: adaptive orchestration under changing workloads  
* **Research & Simulation**: experimentation in MARL and multi-agent organization learning

---

## ⚙️ Interfaces and Integration

CybMASDE can be used through several interfaces:

* **CLI (Command Line Interface)**: for automation, HPC batch runs, and reproducible experiments.  
* **Python Library**: for integration into research workflows and Jupyter notebooks.  
* **Web GUI (Angular)**: for visual project creation, configuration, and monitoring.  

All three interfaces rely on a **shared backend API** and **consistent configuration schema**, ensuring that projects remain interoperable across use modes.

---

## 🔬 Research Impact

CybMASDE provides a *testbed* for studying key research questions in distributed AI:

* How can organizations guide learning in multi-agent systems?  
* How can explainability and interpretability be integrated into MARL?  
* How can policies trained in simulation be safely transferred to real infrastructures?  
* How can agent coordination be maintained under uncertainty or partial observability?  

The platform has been validated in multiple experimental scenarios:
* **Company Infrastructure (Cyber-Defense)**
* **Drone Swarm**
* **Microservices on Kubernetes**
* **Warehouse Management**
* **Overcooked-AI Environments**
* ...

---

## 🧭 Learn More

If you’re new to CybMASDE, start with:

* [Installation](installation.md): Set up the environment and dependencies.  
* [Getting Started](getting-started.md): Create your first project.  
* [Architecture](architecture.md): Understand the internal organization of CybMASDE.  
* [CLI API Reference](cli_reference.md): Explore the full CLI reference.  
