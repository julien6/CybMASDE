# Command-Line Interface (CLI) Reference

CybMASDE provides a unified **Command-Line Interface (CLI)** that allows users to create, validate, execute, analyze, and deploy multi-agent projects without requiring the graphical interface.  
The CLI is designed to be **human-friendly**, **scriptable**, and **HPC-compatible**.

All commands are accessible via:

```bash
cybmasde [command] [options]
```

---

## 🧭 Overview

| Command                 | Purpose                               |
| ----------------------- | ------------------------------------- |
| [ `init` ](#init)         | Create a new project                  |
| [ `validate` ](#validate) | Validate project configuration        |
| [ `run` ](#run)           | Execute the complete pipeline         |
| [ `model` ](#model)       | Run only the Modeling activity        |
| [ `train` ](#train)       | Run only the Training activity        |
| [ `analyze` ](#analyze)   | Run only the Analysis activity        |
| [ `refine` ](#refine)     | Run refinement cycles                 |
| [ `deploy` ](#deploy)     | Deploy policies in real environments  |
| [ `status` ](#status)     | Display project state and metrics     |
| [ `clean` ](#clean)       | Remove temporary data and checkpoints |
| [ `export` ](#export)     | Export results and metrics            |
| [ `help` ](#help)         | Display CLI help                      |

---

## ⚙️ Global Options

| Option                 | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| `-h, --help` | Show command help                                     |
| `-v, --version` | Display CybMASDE version                              |
| `-p, --project <path>` | Specify the project path (default: current directory) |
| `-c, --config <file>` | Use an alternative configuration file                 |

Example:

```bash
cybmasde run -p ./my_project -c ./configs/alternative.json
```

---

## 🧱 `init`

### Description

Create a new CybMASDE project and generate its directory structure, including default configuration files and environment templates.

### Syntax

```bash
cybmasde init -n <project_name> [-d <description>] [-o <output_dir>] [--template <type>]
```

### Options

| Option              | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| `-n, --name` | Name of the project *(required)*                                        |
| `-d, --description` | Short project description                                               |
| `-o, --output` | Output directory (default: `./<project_name>` )                          |
| `--template` | Type of environment template: `handcrafted` , `worldmodel` , or `minimal` |

### Example

```bash
cybmasde init -n swarm_test --template worldmodel -d "Swarm coordination with MOISE+MARL"
```

---

## ✅ `validate`

### Description

Check that the configuration file and associated resources are valid.

### Syntax

```bash
cybmasde validate [--strict] [-q]
```

### Options

| Option        | Description                      |
| ------------- | -------------------------------- |
| `--strict` | Treat all warnings as errors     |
| `-q, --quiet` | Quiet mode: only prints OK/ERROR |

### Example

```bash
cybmasde validate --strict
```

Output:

```
[OK] Configuration is valid.
```

---

## 🚀 `run`

### Description

Execute the **full MAMAD pipeline** (Modeling → Training → Analyzing → Transferring → Refining).

### Syntax

```bash
cybmasde run [--full-auto | --semi-auto | --manual] [options]
```

### Options

| Option                     | Description                                           |
| -------------------------- | ----------------------------------------------------- |
| `--full-auto` | Run the entire pipeline without user interaction      |
| `--semi-auto` | Pause between each step for confirmation              |
| `--manual` | Manually select which phases to execute               |
| `--skip-model` | Skip modeling and use an existing environment         |
| `--skip-analyze` | Skip analysis (useful for testing only)               |
| `--max-refine <N>` | Maximum number of refinement cycles                   |
| `--reward-threshold <val>` | Stop once average reward exceeds this value           |
| `--std-threshold <val>` | Stop when reward variance drops below this value      |
| `--accept-inferred` | Accept inferred organizational specs automatically    |
| `--interactive-infer` | Ask for manual validation of inferred specs (default) |

### Example

```bash
cybmasde run --full-auto --max-refine 3 --reward-threshold 1.5
```

---

## 🧩 `model`

### Description

Run only the **Modeling** activity — generating or loading an environment.

### Syntax

```bash
cybmasde model [--auto | --manual] [options]
```

### Options

| Option                | Description                                         |
| --------------------- | --------------------------------------------------- |
| `--auto` | Build environment from traces using the world model |
| `--manual` | Use handcrafted environment code                    |
| `--traces <dir>` | Specify a directory of traces for model training    |
| `--vae-dim <val>` | Latent dimension for VAE (default: 32)              |
| `--lstm-hidden <val>` | Hidden dimension for RNN layers (default: 64)       |

### Example

```bash
cybmasde model --auto --traces ./data/traces --vae-dim 64
```

---

## 🧠 `train`

### Description

Train multi-agent policies using selected MARL algorithms under MOISE+ constraints.

### Syntax

```bash
cybmasde train [--algo <alg>] [options]
```

### Options

| Option               | Description                                     |
| -------------------- | ----------------------------------------------- |
| `--algo <name>` | Algorithm (MAPPO, MADDPG, QMIX, ROMA, IQL, VDN) |
| `--batch-size <val>` | Batch size (default: 64)                        |
| `--lr <val>` | Learning rate (default: 1e-4)                   |
| `--gamma <val>` | Discount factor (0.9–0.99)                      |
| `--clip <val>` | PPO clipping parameter (0.1–0.3)                |
| `--seed <val>` | Random seed                                     |
| `--epochs <N>` | Number of training epochs                       |

### Example

```bash
cybmasde train --algo MAPPO --epochs 10 --batch-size 128
```

---

## 🔬 `analyze`

### Description

Run the **Auto-TEMM** or standard analysis process on trained policies and trajectories.

### Syntax

```bash
cybmasde analyze [--auto-temm] [--metrics <list>] [--representativity <val>]
```

### Options

| Option                     | Description                                       |
| -------------------------- | ------------------------------------------------- |
| `--auto-temm` | Run Auto-TEMM (automated clustering and analysis) |
| `--metrics <list>` | Metrics to compute: reward, stability, org_fit    |
| `--representativity <val>` | Representativity threshold (0.0–1.0)              |

### Example

```bash
cybmasde analyze --auto-temm --metrics reward,org_fit
```

---

## 🔁 `refine`

### Description

Perform iterative refinement cycles combining analysis and retraining.

### Syntax

```bash
cybmasde refine [--max <N>] [--accept-inferred] [--interactive]
```

### Options

| Option              | Description                                   |
| ------------------- | --------------------------------------------- |
| `--max <N>` | Maximum number of refinement iterations       |
| `--accept-inferred` | Automatically accept inferred specifications  |
| `--interactive` | Ask confirmation before each refinement cycle |

### Example

```bash
cybmasde refine --max 5 --interactive
```

---

## 🌍 `deploy`

### Description

Deploy learned policies into a real or simulated environment.

### Syntax

```bash
cybmasde deploy [--direct | --remote] [options]
```

### Options

| Option                | Description                                    |
| --------------------- | ---------------------------------------------- |
| `--direct` | Deploy policies directly to local agents       |
| `--remote` | Run policies remotely and send actions via API |
| `--checkpoint <file>` | Specify a saved policy checkpoint              |
| `--api <url>` | Define target environment API endpoint         |

### Example

```bash
cybmasde deploy --remote --api http://localhost:8080/api
```

---

## 📊 `status`

### Description

Display the current status of the project, including metrics, policy progress, and recent refinement history.

### Syntax

```bash
cybmasde status
```

### Example Output

```
Project: my_project
Active policy: joint_policy_v3
Average reward: 2.87 ± 0.12
Refinement cycles: 5
Deployment mode: REMOTE
```

---

## 🧹 `clean`

### Description

Remove temporary data, traces, and unused checkpoints to free space.

### Syntax

```bash
cybmasde clean [--traces | --checkpoints | --all]
```

### Example

```bash
cybmasde clean --all
```

---

## 📦 `export`

### Description

Export policies, metrics, and analysis results in standard formats (JSON, CSV, YAML).

### Syntax

```bash
cybmasde export [--format json|csv|yaml] [--output <dir>]
```

### Example

```bash
cybmasde export --format csv --output ./exports
```

---

## 🆘 `help`

### Description

Display the general help menu or the help for a specific command.

### Syntax

```bash
cybmasde help [command]
```

### Example

```bash
cybmasde help train
```

---

## 🧩 Tips for Advanced Users

* Combine CLI options with **project configurations** for reproducibility:

  

```bash
  cybmasde run --config ./configs/experiment.json --max-refine 3
  ```

* Use `--quiet` and redirect logs for **HPC cluster jobs**:

  

```bash
  cybmasde run --full-auto --quiet > logs/run.log 2>&1
  ```

* Integrate CLI commands into Python scripts using `subprocess.run()` for automation.
