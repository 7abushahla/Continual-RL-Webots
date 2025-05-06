import json
import numpy as np

# List of tasks in continual learning order
tasks = ["obstacle", "line", "maze"]
final_model = "maze"  # last task trained

def j(path):
    with open(path) as f:
        return json.load(f)

best_after = {}
finals = {}
zeroshot = {}

# right-after-training JSONs end in _<task>.json
for t in tasks:
    best_after[t] = j(f"logs/result_{t}_{t}.json")["mean_return"]

# evaluations after final model   (_maze)
for t in tasks:
    finals[t] = j(f"logs/result_{t}_{final_model}.json")["mean_return"]

# zero-shot on line & maze
zeroshot["line"] = j("logs/result_line_obstacle.json")["mean_return"]
zeroshot["maze"] = j("logs/result_maze_line.json")["mean_return"]

# --------- Metrics ---------------
forget = {t: best_after[t] - finals[t] for t in tasks}
avg_CF = np.mean(list(forget.values()))
BWT = -avg_CF
avg_FWT = np.mean(list(zeroshot.values()))
overall_SR = np.mean([
    j(f"logs/result_{t}_{final_model}.json")["success_rate"] for t in tasks
])

print("Per-task forgetting:", forget)
print(f"Avg CF={avg_CF:.2f}   BWT={BWT:.2f}   FWT={avg_FWT:.2f}")
print(f"Final average Success Rate = {overall_SR:.1%}") 