# Continual Learning Manual Workflow

This project uses a manual, stage-wise continual learning workflow for Webots RL tasks. Each training and evaluation stage is run manually, and results are aggregated offline.

## Manual Sequence

1. **Train and evaluate each task in order:**
   - Set environment variables or command-line arguments for each stage:
     - `MODE=train` or `MODE=eval`
     - `LOAD_FROM=<previous_task>` (empty for first task)
   - Example (Webots GUI > Simulation > Optional arguments):
     - `--MODE=train --LOAD_FROM=` (for first task)
     - `--MODE=eval --LOAD_FROM=obstacle` (for zero-shot or after training)

2. **Stages:**
   - 1-A: Train obstacle (`MODE=train`, `LOAD_FROM=`)
   - 1-B: Evaluate obstacle (`MODE=eval`, `LOAD_FROM=obstacle`)
   - 2-A: Zero-shot evaluate line (`MODE=eval`, `LOAD_FROM=obstacle`)
   - 2-B: Train line (`MODE=train`, `LOAD_FROM=obstacle`)
   - 2-C: Re-evaluate obstacle (`MODE=eval`, `LOAD_FROM=line`)
   - 3-A: Zero-shot evaluate maze (`MODE=eval`, `LOAD_FROM=line`)
   - 3-B: Train maze (`MODE=train`, `LOAD_FROM=line`)
   - 3-C: Final obstacle evaluation (`MODE=eval`, `LOAD_FROM=maze`)
   - 3-D: Final line evaluation (`MODE=eval`, `LOAD_FROM=maze`)
   - 3-E: Final maze evaluation (`MODE=eval`, `LOAD_FROM=maze`)

## Output Files

- **Training:**
  - `logs/sac_model_<task>.zip`
  - `logs/replay_buffer_<task>.pkl`
  - `logs/tb/<task>_<timestamp>/events.out.tfevents.*` (TensorBoard)
- **Evaluation:**
  - `logs/result_<evalTask>_<LOAD_FROM>.json` (summary of evaluation)
  - `logs/monitor_<evalTask>.csv` (if Monitor wrapper is used)

## Aggregation

After all runs are complete, run:

```bash
python continual_driver.py
```

This will print:
- Per-task forgetting
- Average CF, BWT, FWT
- Final average Success Rate

## TensorBoard

To view training curves for each run:

```bash
tensorboard --logdir logs/tb
```

Each run is logged in a unique folder by task and timestamp. 