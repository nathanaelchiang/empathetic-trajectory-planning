import json
import numpy as np

filename = "results/topk_results_classifier_20260416_181941.json"
print(f"Saved to: {filename}")

with open(filename, "r", encoding="utf-8") as f:
    results = json.load(f)


def safe_mean(arr):
    """Return the mean of arr as a float, or 0.0 if arr is empty."""
    return float(np.mean(arr)) if arr else 0.0


def get_metric(x, key, default=0.0):
    """Return x[key] if present, otherwise default."""
    return x[key] if key in x else default


print("\n=== Top-k Metrics Summary ===\n")

metrics = [
    # Core trajectory metrics
    ("Drift", [get_metric(x, "drift") for x in results]),
    ("Alignment score", [get_metric(x, "alignment_score") for x in results]),
    ("Reversal rate", [get_metric(x, "reversal_rate") for x in results]),
    (
        "Mean entropy",
        [safe_mean(x.get("per_turn_entropy", [])) for x in results],
    ),
    (
        "Mean step distance",
        [safe_mean(x.get("per_step_distances", [])) for x in results],
    ),
    ("Traj level score", [get_metric(x, "traj_level_score") for x in results]),
    # Top-k specific
    ("Score variance", [get_metric(x, "mean_score_variance") for x in results]),
]

# ---- Print summary ----
for label, arr in metrics:
    arr = np.array(arr, dtype=float)
    print(
        f"{label:22s}  mean={arr.mean():.4f}  std={arr.std():.4f}"
        f"  min={arr.min():.4f}  max={arr.max():.4f}"
    )


# ---- Extra diagnostics (VERY useful for top-k) ----
print("\n=== Top-k Diagnostics ===\n")

# How often top-k actually mattered
selection_spread = []

for x in results:
    log = x.get("candidate_log", [])
    for turn in log:
        scores = turn.get("scores", [])
        if scores:
            spread = max(scores) - min(scores)
            selection_spread.append(spread)

if selection_spread:
    selection_spread = np.array(selection_spread)
    print(
        f"Score spread (per turn): mean={selection_spread.mean():.4f} "
        f"std={selection_spread.std():.4f}"
    )

# How decisive the model is
score_variances = [get_metric(x, "mean_score_variance") for x in results]
score_variances = np.array(score_variances)

print(f"Decision variance: mean={score_variances.mean():.4f} (higher = clearer winner)")

# Optional: correlation between variance and performance
traj_scores = np.array([get_metric(x, "traj_level_score") for x in results])

if len(score_variances) == len(traj_scores):
    corr = np.corrcoef(score_variances, traj_scores)[0, 1]
    print(f"Corr(score_variance, traj_score) = {corr:.4f}")
