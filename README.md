# Empathetic Trajectory Planning

Generating empathetic dialogue responses that follow a target emotional trajectory across multi-turn conversations. Rather than responding greedily turn-by-turn, each planner considers future emotional states when selecting a reply, producing more coherent and emotionally consistent conversations.

## Overview

Each conversation has a user emotion (e.g. `sad`, `proud`, `guilty`) derived from the **EmpatheticDialogues** dataset. The assistant's target emotion is determined by a fixed mapping (e.g. user `sad` → assistant `caring`). Planners must steer the conversation toward that target while remaining contextually appropriate.

Four planning strategies are implemented, ranging from no planning to full tree search:

```
no planning (baseline) → single-turn (top-k) → linear lookahead → tree (ToT)
```

## Project Structure

```
empathetic-trajectory-planning/
├── experiments/
│   ├── run_baseline_generation.py      # Greedy baseline (no planning)
│   ├── topk_planner.py                 # Single-turn top-k reranking
│   ├── lookahead_planner.py            # Linear rollout lookahead
│   ├── tot_planner.py                  # Tree-of-Thought beam search
│   ├── baseline_metrics_calculator.py  # Aggregate metrics for baseline results
│   ├── topk_metrics_calculator.py      # Aggregate metrics for top-k results
│   ├── lookahead_metrics_calculator.py # Aggregate metrics for lookahead results
│   ├── tot_metrics_calculator.py       # Aggregate metrics for ToT results
│   ├── demo.py                         # End-to-end demo (mock or real)
│   └── test_emotion_pipeline.py        # Pipeline smoke tests
├── models/
│   └── emotion_classifier.py           # RoBERTa-based emotion classifier (GoEmotions)
├── evaluation/
│   └── trajectory.py                   # Drift and trajectory extraction utilities
├── emotion/
│   └── assistant_targets.py            # User → assistant emotion target mapping
├── results/                            # Output JSON files from experiment runs
├── requirements.txt
└── setup.sh                            # Conda environment setup
```

## Planners

### Baseline (`run_baseline_generation.py`)
Greedy generation with a system prompt listing the target emotions. No candidate sampling or lookahead. Serves as the lower bound for comparison.

### Top-k Reranking (`topk_planner.py`)
Samples K candidate replies per turn and picks the one with the highest P(target emotion) from the classifier. Single-turn scoring only — no lookahead.

| Hyperparameter | Default | Description |
|---|---|---|
| `K` | 5 | Candidates sampled per turn |
| `MAX_NEW_TOKENS` | 150 | Max tokens per generation |
| `SCORER_MODE` | `classifier` | `classifier`, `llm_judge`, or `both` |

### Linear Lookahead (`lookahead_planner.py`)
For each of the K candidates, simulates a cheap linear rollout (single sample, no branching) over the next `LOOKAHEAD_DEPTH` turns using real future user utterances. Scores the full simulated trajectory and selects the candidate with the best trajectory-level score.

| Hyperparameter | Default | Description |
|---|---|---|
| `K` | 5 | Candidates sampled per turn |
| `LOOKAHEAD_DEPTH` | 2 | Future assistant turns to simulate |
| `MAX_NEW_TOKENS` | 512 | Max tokens per generation |
| `SCORER_MODE` | `classifier` | `classifier`, `llm_judge`, or `both` |

### Tree-of-Thought (`tot_planner.py`)
Beam search over a tree of assistant replies. At each depth, expands every surviving beam node into K branches, scores each path using a trajectory-level score, and prunes to the top `BEAM_WIDTH` nodes. The first reply of the best-scoring leaf path is committed.

| Hyperparameter | Default | Description |
|---|---|---|
| `K_BRANCHES` | 3 | Branches expanded per node |
| `BEAM_WIDTH` | 2 | Nodes kept after each depth level |
| `DEPTH` | 2 | Look-ahead depth in assistant turns |
| `MAX_NEW_TOKENS` | 512 | Max tokens per generation |
| `SCORER_MODE` | `classifier` | `classifier`, `llm_judge`, or `both` |


## Metrics

All planners are evaluated on the following metrics per conversation:

| Metric | Description | Direction |
|---|---|---|
| **Drift** | Mean cosine distance between consecutive emotion distributions | Lower is better |
| **Alignment score** | Fraction of turns where predicted emotion matches target | Higher is better |
| **Reversal rate** | Fraction of turns where the emotion label changes | Lower is better |
| **Mean entropy** | Mean Shannon entropy of per-turn emotion distributions | — |
| **Mean step distance** | Mean L1 distance between consecutive emotion vectors | Lower is better |
| **Trajectory level score** | Composite: `alignment × (1 − 0.5 × reversal) × (1 − 0.5 × drift)` | Higher is better |

## Setup

```bash
# Create conda environment (requires CUDA 12.1.1 and Anaconda)
bash setup.sh cu118   # GPU (CUDA 11.8)
bash setup.sh cpu     # CPU only

# Or install dependencies manually
pip install -r requirements.txt
```

**requirements.txt**
```
torch
transformers
numpy
scikit-learn
matplotlib
datasets
```

## Running Experiments

### Quick demo (no GPU required)

```bash
python -m experiments.demo --mode mock
```

Runs a mock pipeline with hardcoded dialogues — no model loading, completes in ~30 seconds.

### Full experiments

```bash
# Baseline
python -m experiments.run_baseline_generation

# Top-k reranking
python -m experiments.topk_planner

# Linear lookahead
python -m experiments.lookahead_planner \
  --lookahead-depth 2 \
  --k 5 \

# Tree-of-Thought
python -m experiments.tot_planner

Results are saved to `results/` as timestamped JSON files (e.g. `results/lookahead_results_classifier_20260416_120000.json`).

### Computing aggregate metrics
Replace `filename` with the name of the results file. 

```bash
python -m experiments.baseline_metrics_calculator
python -m experiments.topk_metrics_calculator
python -m experiments.lookahead_metrics_calculator
python -m experiments.tot_metrics_calculator
```

## Models

| Component | Model |
|---|---|
| Generation (default) | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Emotion classifier | `SamLowe/roberta-base-go_emotions` (28 labels) |

Generation models are loaded in 4-bit quantization (`bfloat16`) to fit in GPU memory. The emotion classifier runs on GPU if available, otherwise CPU.

The active generation model can be changed by editing the `model_name` variable at the top of each planner file. Commented alternatives include Qwen 2.5 (0.5B–7B), Mistral 7B, and DeepSeek-R1 distills.

## Data

Conversations are loaded from the HuggingFace `empathetic_dialogues` dataset (train split), filtered to conversations with ≥ 4 turns and an even number of turns (user–assistant pairs).
