# Navigate Without Forgetting: Continual Reinforcement Learning for Mobile Robot Navigation in Webots
_Hamza A. Abushahla, Ariel Justine Navarro Panopio, Layth Al-Khairulla, and Imran Omar Arif_

This repository contains code and resources for the paper: "[Navigate Without Forgetting: Continual Reinforcement Learning for Mobile Robot Navigation in Webots](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4234)".


## Overview

This project implements a **Continual Reinforcement Learning (CRL)** framework for mobile robot navigation using the **e-puck** robot in **Webots**. We design a unified agent trained sequentially across multiple tasks—**maze navigation**, **line following**, and **obstacle avoidance**—using **Soft Actor-Critic (SAC)**. The setup evaluates the agent’s ability to learn new behaviors while retaining previously acquired skills.

The system is built on **DeepBots** with a Gym-compatible interface, and features structured training curricula, automatic performance logging, and CRL evaluation metrics.


## Folder Structure
```
.
├── controllers/
│   ├── robot/                 # Handles low-level robot control
│   │   └── robot.py
│   └── supervisor/           # Supervisor with Gym environment and RL loop
│       ├── supervisor.py
│       └── logs/             # Saved models and replay buffers
├── worlds/
│   ├── MazeNav.wbt
│   ├── LineFollow.wbt
│   └── ObstacleAvoid.wbt
├── registration/
│   └── register_envs.py      # Registers environments with Gym
├── results/
│   └── plots/                # Figures and evaluation metrics
└── README.md
```

### Running the Experiments

1. **Launch the Webots Simulator**

Open one of the provided `.wbt` worlds (e.g., `MazeNav.wbt`) in Webots. Keep the simulation paused for now.

2. **Configure the Supervisor**

Edit the `controllers/supervisor/supervisor.py` file to specify the experiment mode:

```python
# ====== Manual Experiment Configuration ======
MODE = "training"         # "training" or "evaluation"
TASK = "maze"             # "maze", "line", or "obstacle"
LOAD_FROM = None          # None, or a previously trained task: "maze", "line", "obstacle"
CHECKPOINT_STEP = None    # e.g., 1000, 2000, ..., or None for latest
```

3. **Run a Single Task**

Set `MODE = "training"` and `TASK = "<task_name>"` (e.g., `"line"`), with `LOAD_FROM = None` to start learning from scratch:

```bash
# Launch Webots and run the simulation
# Then observe training logs in terminal output
```

4. **Sequential Task Training**

To perform continual learning, train one task at a time, updating `LOAD_FROM` to the last trained task:

- Train `maze` from scratch  
- Then set `TASK = "line"`, `LOAD_FROM = "maze"`  
- Then set `TASK = "obstacle"`, `LOAD_FROM = "line"`

Each task builds on the policy learned from the previous one.

5. **Evaluation**

Switch to evaluation mode to test a trained policy:

```python
MODE = "evaluation"
TASK = "line"             # Task to evaluate
LOAD_FROM = "line"        # Model to load
```

Then run the Webots simulation and observe performance metrics in logs.


## Citation & Reaching out
If you use our work for your own research, please cite us with the below: 

```bibtex
@Article{abushahla2025navigate,
  AUTHOR = {Abushahla, Hamza A. and Panopio, Ariel J. N. and Al-Khairulla, Layth and Arif, Imran Omar},
  TITLE = {Navigate Without Forgetting: Continual Reinforcement Learning for Mobile Robot Navigation in Webots},
  JOURNAL = {},
  YEAR = {2025},
  VOLUME = {},
  NUMBER = {},
  PAGES = {},
  DOI = {},
  NOTE = {To appear}
}
```
