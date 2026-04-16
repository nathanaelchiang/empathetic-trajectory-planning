import json
import numpy as np

filename = "results/baseline_results_20260413_034038.json"
print(f"Saved to: {filename}")

with open(filename, "r", encoding="utf-8") as f:
    results = json.load(f)


def safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0.0


metrics = [
    ("Drift", [x.get("drift", 0.0) for x in results]),
    ("Alignment score", [x.get("alignment_score", 0.0) for x in results]),
    ("Reversal rate", [x.get("reversal_rate", 0.0) for x in results]),
    (
        "Mean entropy",
        [safe_mean(x.get("per_turn_entropy", [])) for x in results],
    ),
    (
        "Mean step distance",
        [safe_mean(x.get("per_step_distances", [])) for x in results],
    ),
    ("Traj level score", [x.get("traj_level_score", 0.0) for x in results]),
]

print("\n=== Baseline Summary ===\n")

for label, arr in metrics:
    arr = np.array(arr, dtype=float)
    print(
        f"{label:22s}  mean={arr.mean():.4f}  std={arr.std():.4f}"
        f"  min={arr.min():.4f}  max={arr.max():.4f}"
    )
