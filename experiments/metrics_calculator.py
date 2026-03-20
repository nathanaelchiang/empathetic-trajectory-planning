import json
import numpy as np

filename = "results/tot_results_classifier_20260320_005642.json"

with open(filename, "r", encoding="utf-8") as f:
    results = json.load(f)

for label, arr in [
    ("Drift", [x["drift"] for x in results]),
    ("Alignment score", [x["alignment_score"] for x in results]),
    ("Reversal rate", [x["reversal_rate"] for x in results]),
    ("Mean entropy", [np.mean(x["per_turn_entropy"]) for x in results]),
    (
        "Mean step distance",
        [
            np.mean(x["per_step_distances"]) if x["per_step_distances"] else 0.0
            for x in results
        ],
    ),
    ("Traj level score", [x["traj_level_score"] for x in results]),
    ("Mean traj score", [x["mean_traj_score"] for x in results]),
]:
    arr = np.array(arr, dtype=float)
    print(
        f"{label:22s}  mean={arr.mean():.4f}  std={arr.std():.4f}"
        f"  min={arr.min():.4f}  max={arr.max():.4f}"
    )
