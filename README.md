# Resource Allocation for Twin Maintenance (TM) and Task Processing (TP) in Vehicular Edge Computing Networks (VECN)

This repository provides the simulation framework, reinforcement learning algorithms, and pre-computed experimental benchmarks for joint computational resource allocation in Vehicular Edge Computing Networks (VECN). The implementation is calibrated based on the physical and architectural models presented in:

> **Y. Xie, Q. Wu, P. Fan, N. Cheng, W. Chen, J. Wang, and K. B. Letaief**, *"Resource Allocation for Twin Maintenance and Task Processing in Vehicular Edge Computing Network,"* **IEEE Internet of Things Journal**, vol. 12, no. 15, August 2025.

---

## Table of Contents

- [Project Overview & System Working](#project-overview--system-working)
  - [System Architecture](#system-architecture)
  - [Dual Workload Streams](#dual-workload-streams)
  - [Objective & Reward Formulation](#objective--reward-formulation)
  - [Reinforcement Learning Algorithms](#reinforcement-learning-algorithms)
- [Repository Structure](#repository-structure)
- [Using the Simulation Results & Console Log](#using-the-simulation-results--console-log)
  - [Console Output Log (`console_output_v6.log`)](#console-output-log-console_output_v6log)
  - [Benchmark Summaries (CSV)](#benchmark-summaries-csv)
  - [Generated Publication Figures (PNG)](#generated-publication-figures-png)
- [Using the Simulation Notebook (`vecsim_simulation.ipynb`)](#using-the-simulation-notebook-vecsim_simulationipynb)
- [Reproduction Instructions](#reproduction-instructions)
  - [Option A: Google Colab (Recommended)](#option-a-google-colab-recommended)
  - [Option B: Local Environment (Jupyter / CLI)](#option-b-local-environment-jupyter--cli)
- [Key Physical Parameters & Calibration](#key-physical-parameters--calibration)
- [License & Citation](#license--citation)

---

## Project Overview & System Working

### System Architecture
In a Vehicular Edge Computing Network (VECN), mobile vehicles travel across multilane roads covered by a Roadside Base Station (BS) equipped with an edge computing server. The server must dynamically divide its limited CPU computational capacity (*F* = 100 GHz) among multiple vehicles to satisfy two concurrent, competing objectives:

```
                                +-----------------------------+
                                |  Roadside Edge Base Station |
                                |   Total CPU: F = 100 GHz    |
                                +--------------+--------------+
                                               |
                     +-------------------------+-------------------------+
                     | Wireless V2I Uplink / Downlink Scheduling         |
                     +-------------------------+-------------------------+
                                               |
             +---------------------------------+---------------------------------+
             |                                 |                                 |
             v                                 v                                 v
     [ Vehicle 1 ]                     [ Vehicle i ]                     [ Vehicle N ]
     - State Telemetry (TM)            - State Telemetry (TM)            - State Telemetry (TM)
     - Offloaded Task (TP)             - Offloaded Task (TP)             - Offloaded Task (TP)
```

### Dual Workload Streams
For each vehicle *i* (where *i* ∈ {1, ..., *N*}), the edge server handles two distinct service requests:
1. **Twin Maintenance (TM) / Digital Twin (DT) Synchronization:**
   - Real-time vehicle sensory and telemetry data is transmitted to update the digital twin model at the edge server.
   - **Deadline:** Strict latency threshold of 100 ms (*T*<sub>dt</sub> = 0.1 s).
   - **Objective Weight:** *ρ* = 0.8 (high priority given to real-time synchronization accuracy).
2. **Task Processing (TP):**
   - Computation-intensive workloads offloaded by vehicular applications (e.g., path planning, perception).
   - **Deadline:** Latency threshold of 500 ms (*T*<sub>tk</sub> = 0.5 s).
   - **Objective Weight:** 1 − *ρ* = 0.2.

### Objective & Reward Formulation
The system maximizes the composite vehicular satisfaction utility *U* across both services while enforcing latency deadlines:

```
U = ρ · Q_dt + (1 - ρ) · Q_tk
```

Where:
- *Q*<sub>dt</sub> is the Quality-of-Service utility for Digital Twin maintenance.
- *Q*<sub>tk</sub> is the Quality-of-Service utility for Task Processing.
- *ρ* = 0.8 is the weight factor favoring twin maintenance freshness over task completion delay.

When an allocation causes task or digital twin latency to exceed its respective deadline (*T*<sub>dt</sub> = 100 ms or *T*<sub>tk</sub> = 500 ms), delay penalties are applied to guide the policy toward strict deadline adherence.

### Reinforcement Learning Algorithms
The framework evaluates continuous resource-allocation policies across 5 algorithms:
- **TMTPRCS (Proposed Scheme):** Multi-Agent Actor-Critic architecture with individual actor networks and centralized critic evaluation, utilizing Softmax action mapping to allocate feasible CPU shares.
- **MATD3 (Benchmark):** Multi-Agent Twin Delayed Deep Deterministic Policy Gradient with clipped double-Q learning and delayed policy updates.
- **SAC (Benchmark):** Soft Actor-Critic baseline with maximum entropy exploration.
- **PPO (Benchmark):** Proximal Policy Optimization baseline with clipped surrogate objective.
- **Random Allocation (Baseline):** Uniform random feasible CPU allocation for baseline comparison.

---

## Repository Structure

```text
realloc-for-TM-and-TP-in-VECN/
│
├── vecsim_simulation.ipynb          # Main monolithic simulation notebook (Colab & local ready)
│
├── simulation_results_v6/           # Pre-computed simulation outputs & execution logs
│   ├── console_output_v6.log        # Full console execution log from simulation run
│   ├── comparison_summary_N7.csv    # Evaluated metrics for all 5 algorithms at N = 7
│   ├── vehicle_sweep_summary.csv    # Scalability metrics across vehicle densities (N = 4 to 8)
│   ├── fig4_convergence_N7.png      # Convergence curve comparison (500 episodes, N = 7)
│   ├── fig5_resource_utilization.png # Server CPU utilization across vehicle densities
│   ├── fig6_per_vehicle_utility.png # Per-vehicle satisfaction utility across vehicle counts
│   ├── fig7a_twin_delay.png         # Digital Twin maintenance latency vs 100 ms deadline
│   └── fig7b_task_delay.png         # Task processing latency vs 500 ms deadline
│
├── .gitignore                       # Git ignore rules for temporary scripts & drafts
└── README.md                        # Documentation and reproduction guide
```

---

## Using the Simulation Results & Console Log

If you want to inspect or analyze the simulation outcomes without re-running the long training process, all verified outputs are pre-packaged in the [`simulation_results_v6/`](./simulation_results_v6/) folder.

### Console Output Log (`console_output_v6.log`)
The notebook integrates a custom `TeeLogger` that redirects all standard output and error streams directly to [`simulation_results_v6/console_output_v6.log`](./simulation_results_v6/console_output_v6.log).

Key sections recorded in the log file include:
- **System Initialization & Hardware Check:** Device assignment (`cuda` vs `cpu`) and physical parameter verification.
- **Smoke Test Verification:** Verification of observation dimensions, action bounds, and initial reward outputs across 10 test episodes at *N* = 7.
- **Experiment 1 (500 Episodes at *N* = 7):**
  - Per-algorithm training completion checkpoints.
  - Evaluation results: average reward, digital twin deadline satisfaction ratio, task deadline satisfaction ratio, and mean delays.
  - Final tabulated performance summary.
- **Experiment 2 (Density Sweep *N* ∈ {4, 5, 6, 7, 8}):**
  - Execution progression for each vehicle density across 200 episodes.
  - Confirmation of exported Figures 5, 6, 7a, and 7b.
- **Experiment 3 (Multi-Seed Variance Evaluation):**
  - Stability evaluation of the proposed **TMTPRCS** across 5 independent seeds (`42` to `46`).
  - Reports Mean ± Standard Deviation for satisfaction utility and deadline satisfaction ratios.
- **Packaging Confirmation:** Manifest of files archived into `simulation_results_v6.zip` for download.

### Benchmark Summaries (CSV)
- **[`comparison_summary_N7.csv`](./simulation_results_v6/comparison_summary_N7.csv):**
  Contains numerical comparisons of all 5 algorithms under *N* = 7 vehicles, including:
  - `avg_reward`: Mean evaluation reward.
  - `server_utilization_pct`: Percentage of allocated server compute capacity.
  - `dt_cpu_ghz` & `task_cpu_ghz`: Allocated CPU clock frequency shares (in GHz).
  - `dt_delay_mean_ms` / `task_delay_mean_ms`: Mean latencies (along with median and 95th percentiles).
  - `dt_deadline_sat_ratio` / `task_deadline_sat_ratio`: Adherence ratios against the 100 ms and 500 ms deadlines.
- **[`vehicle_sweep_summary.csv`](./simulation_results_v6/vehicle_sweep_summary.csv):**
  Comprehensive matrix recording how utility, compute allocation, and delays scale as vehicle density increases from *N* = 4 to *N* = 8.

### Generated Publication Figures (PNG)
| Artifact | Description |
| :--- | :--- |
| **[`fig4_convergence_N7.png`](./simulation_results_v6/fig4_convergence_N7.png)** | Training reward convergence across 500 episodes (*N* = 7) with a 10-episode moving average. |
| **[`fig5_resource_utilization.png`](./simulation_results_v6/fig5_resource_utilization.png)** | Edge server CPU resource utilization across vehicle densities (*N* = 4, 5, 6, 7, 8). |
| **[`fig6_per_vehicle_utility.png`](./simulation_results_v6/fig6_per_vehicle_utility.png)** | Average per-vehicle satisfaction utility comparison across varying vehicle densities. |
| **[`fig7a_twin_delay.png`](./simulation_results_v6/fig7a_twin_delay.png)** | Twin Maintenance delay across vehicle counts benchmarked against the 100 ms threshold. |
| **[`fig7b_task_delay.png`](./simulation_results_v6/fig7b_task_delay.png)** | Task Processing delay across vehicle counts benchmarked against the 500 ms threshold. |

---

## Using the Simulation Notebook (`vecsim_simulation.ipynb`)

The primary simulation file is [`vecsim_simulation.ipynb`](./vecsim_simulation.ipynb). It is fully self-contained and organized into documented stages:

1. **Setup & Logging (Cells 1–2):** Imports libraries, detects GPU availability (`cuda` vs `cpu`), configures `TeeLogger`, and initializes physical constants.
2. **Environment Simulation (Cells 3–4):** Implements vehicle mobility (3 lanes, Poisson/uniform arrivals, bounded speeds), wireless channel path loss with Bessel-function fading, and delay calculation functions.
3. **Neural Network Models (Cells 5–6):** Defines PyTorch `ActorNetwork`, `CriticNetwork`, and experience `ReplayBuffer`.
4. **Algorithm Implementations (Cells 7–18):** Contains classes for `RandomBaseline`, `TMTPRCS`, `MATD3`, `PPO`, and `SAC`.
5. **Standardized Evaluation Metrics (Cells 19–22):** Deterministic evaluation harness assessing latency, deadline satisfaction, and compute usage.
6. **Smoke Testing (Cells 23–24):** 10-episode verification test checking tensor dimensions and allocation sanity.
7. **Experiment 1 — Algorithm Comparison (Cells 25–30):** Trains all 5 algorithms at *N* = 7 for 500 episodes, plots Figure 4, and exports `comparison_summary_N7.csv`.
8. **Experiment 2 — Vehicle Density Sweep (Cells 31–42):** Iterates across *N* ∈ {4, 5, 6, 7, 8} (200 episodes each), exports Figures 5, 6, 7a, and 7b, and outputs `vehicle_sweep_summary.csv`.
9. **Experiment 3 — Multi-Seed Stability (Cells 43–48):** Evaluates TMTPRCS across 5 distinct random seeds (300 episodes each) to quantify mean and standard deviation.
10. **Packaging & Export (Cells 49–50):** Automatically zips all generated artifacts into `simulation_results_v6.zip` and triggers an in-browser download if run on Google Colab.

---

## Reproduction Instructions

### Option A: Google Colab (Recommended)
Running in Google Colab provides access to free GPU acceleration and direct file download triggers:

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload `vecsim_simulation.ipynb`.
3. Set hardware accelerator: **Runtime** → **Change runtime type** → **T4 GPU** (or CPU).
4. Run all cells sequentially via **Runtime** → **Run all** (or execute cell by cell).
5. At the conclusion of Cell 50, `simulation_results_v6.zip` will automatically download to your local machine.

### Option B: Local Environment (Jupyter / CLI)

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/realloc-for-TM-and-TP-in-VECN.git
cd realloc-for-TM-and-TP-in-VECN
```

#### 2. Create and Activate Virtual Environment
```bash
# Python 3.8+ recommended
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install torch numpy pandas matplotlib scipy jupyter nbformat
```

#### 4. Run the Notebook
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook vecsim_simulation.ipynb
```
Select **Kernel** → **Restart & Run All**.

#### 5. Headless Execution (Optional CLI)
To execute the notebook from the terminal and save an executed copy:
```bash
jupyter nbconvert --to notebook --execute vecsim_simulation.ipynb --output vecsim_simulation_executed.ipynb
```

Execution logs will stream in real time to the terminal and simultaneously be saved to `console_output_v6.log`.

---

## Key Physical Parameters & Calibration

The simulation environment is configured according to the official baseline:

| Parameter | Symbol / Variable | Value | Description |
| :--- | :--- | :--- | :--- |
| **Number of Lanes** | — | 3 | Highway roadway lanes |
| **Lane Width** | — | 3.75 m | Standard highway lane width |
| **BS Distance from Road** | *l*₀ | 8.0 m | Perpendicular offset to road edge |
| **BS Antenna Height** | *h*<sub>bs</sub> | 25.0 m | Height of edge server tower |
| **Time Slot Duration** | *Δt* | 0.1 s (100 ms) | Simulation time-step |
| **Channel Bandwidth** | *W* | 20 MHz | Wireless transmission bandwidth |
| **Carrier Frequency** | *f*<sub>c</sub> | 2.0 GHz | RF carrier frequency |
| **Vehicle Transmission Power** | *P*<sub>tx</sub> | 23 dBm (≈ 0.20 W) | Uplink transmit power |
| **Background Noise Power** | *σ*² | -110 dBm (10⁻¹⁴ W) | AWGN noise floor |
| **Vehicle Velocity Range** | *v* | 10 m/s – 15 m/s | 36 km/h – 54 km/h |
| **DT Data Size** | *D*<sub>dt</sub> | 10 bytes – 200 bytes | Telemetry update packet payload |
| **Task Data Size** | *D*<sub>tk</sub> | 1000 bytes – 1500 bytes | Offloaded application task payload |
| **DT Latency Deadline** | *T*<sub>dt</sub> | 0.1 s (100 ms) | Hard deadline for twin freshness |
| **Task Latency Deadline** | *T*<sub>tk</sub> | 0.5 s (500 ms) | Task completion deadline |
| **Compute Workload Intensity** | — | 0.25 MHz/byte | CPU cycles required per byte |
| **Server Total CPU Capacity** | *F* | 100.0 GHz | Total edge server computation clock |
| **Twin Utility Weight** | *ρ* | 0.8 | Priority weighting for DT freshness |

---

## License & Citation

This project is made available for academic research and educational purposes. If you reference this implementation or data in your research, please cite the original article:

```bibtex
@article{xie2025resource,
  author={Xie, Y. and Wu, Q. and Fan, P. and Cheng, N. and Chen, W. and Wang, J. and Letaief, K. B.},
  journal={IEEE Internet of Things Journal}, 
  title={Resource Allocation for Twin Maintenance and Task Processing in Vehicular Edge Computing Network}, 
  year={2025},
  volume={12},
  number={15},
  pages={},
  doi={10.1109/JIOT.2025.XXXXXXX}
}
```
