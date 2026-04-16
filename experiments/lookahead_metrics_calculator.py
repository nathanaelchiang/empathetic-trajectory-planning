import json
import numpy as np

filename = "results/lookahead_results_classifier_20260413_083322.json"
print(f"Saved to: {filename}")

with open(filename, "r", encoding="utf-8") as f:
    results = json.load(f)


def safe_mean(arr):
    return np.mean(arr) if arr else 0.0


def get_metric(x, key, default=0.0):
    return x[key] if key in x else default


metrics = [
    ("Drift", [get_metric(x, "drift") for x in results]),
    ("Alignment score", [get_metric(x, "alignment_score") for x in results]),
    ("Reversal rate", [get_metric(x, "reversal_rate") for x in results]),
    ("Mean entropy", [safe_mean(x.get("per_turn_entropy", [])) for x in results]),
    (
        "Mean step distance",
        [safe_mean(x.get("per_step_distances", [])) for x in results],
    ),
    ("Traj level score", [get_metric(x, "traj_level_score") for x in results]),
    # 🔥 planner-aware (optional, auto-handled)
    (
        "Mean traj score",
        [
            get_metric(x, "mean_traj_score", get_metric(x, "traj_level_score"))
            for x in results
        ],
    ),
    ("Score variance", [get_metric(x, "mean_score_variance") for x in results]),
    ("Mean lookahead score", [get_metric(x, "mean_lookahead_score") for x in results]),
]

print("\n=== Unified Summary ===\n")

for label, arr in metrics:
    arr = np.array(arr, dtype=float)
    print(
        f"{label:22s}  mean={arr.mean():.4f}  std={arr.std():.4f}"
        f"  min={arr.min():.4f}  max={arr.max():.4f}"
    )
