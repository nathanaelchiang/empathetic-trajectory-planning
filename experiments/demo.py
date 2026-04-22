"""
Combined demo for the empathetic trajectory planning project.

Two modes:

  --mode mock     (default) CPU-only, ~30s. Runs the evaluation pipeline
                  (emotion classifier + trajectory metrics) on pre-generated
                  example outputs from all four planners on two
                  EmpatheticDialogues scenarios (one "sad", one "proud").
                  No LLM is loaded — only the RoBERTa GoEmotions classifier.

  --mode real     End-to-end. Loads a small LLM (default
                  DeepSeek-R1-Distill-Qwen-1.5B in 4-bit) and the classifier,
                  pulls N conversations from EmpatheticDialogues, and runs
                  all four planners (baseline, top-k, lookahead, ToT) on each.
                  Requires a CUDA GPU — the planner modules call `.to('cuda')`
                  directly.

Place this file at experiments/demo.py and run from the project root:

    python -m experiments.demo --mode mock
    python -m experiments.demo --mode real --num-conversations 2
    python -m experiments.demo --mode real --num-conversations 1 --model-name Qwen/Qwen2.5-3B-Instruct
"""

import sys
import os
import argparse
import numpy as np

# This file lives in experiments/, so the project root is one level up.
# Make sure it's importable regardless of where we run from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.emotion_classifier import EmotionClassifier
from evaluation.trajectory import compute_drift
from emotion.assistant_targets import get_assistant_target
from experiments.run_baseline_generation import (
    get_trajectory_labels,
    compute_per_step_distances,
    compute_emotion_entropy,
    compute_reversal_rate,
    compute_peak_drift_turn,
    compute_trajectory_alignment,
    compute_trajectory_level_score,
)


PLANNER_ORDER = ["baseline", "topk", "lookahead", "tot"]
PLANNER_LABELS = {
    "baseline": "Baseline (no planning)",
    "topk": "Top-k Reranking",
    "lookahead": "Linear Lookahead",
    "tot": "Tree-of-Thoughts",
}


def evaluate_dialogue(assistant_turns, target_trajectory, classifier):
    """Run the full metrics pipeline on a single generated dialogue."""
    labels = get_trajectory_labels(assistant_turns, classifier)
    probs = classifier.predict_proba(assistant_turns)
    per_step_dist = compute_per_step_distances(probs)
    entropies = compute_emotion_entropy(probs)
    return {
        "labels": labels,
        "drift": float(compute_drift(assistant_turns, classifier)),
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "reversal_rate": float(compute_reversal_rate(labels)),
        "peak_drift_turn": int(compute_peak_drift_turn(per_step_dist)),
        "alignment_score": float(
            compute_trajectory_alignment(labels, target_trajectory)
        ),
        "traj_level_score": float(
            compute_trajectory_level_score(labels, target_trajectory, per_step_dist)
        ),
    }


def print_per_planner_block(name, user_turns, assistant_turns, metrics):
    print(f"\n--- {name} ---")
    if user_turns is not None:
        for i, (u, a) in enumerate(zip(user_turns, assistant_turns)):
            print(f"  [User {i + 1}]      {u}")
            print(f"  [Assistant {i + 1}] {a}")
    else:
        for i, a in enumerate(assistant_turns):
            print(f"  Turn {i + 1}: {a}")
    print(f"  Predicted emotions: {metrics['labels']}")
    print(
        f"  alignment={metrics['alignment_score']:.2f}  "
        f"drift={metrics['drift']:.3f}  "
        f"reversal={metrics['reversal_rate']:.2f}  "
        f"traj_score={metrics['traj_level_score']:.3f}"
    )


def print_comparison_table(rows):
    """rows = [(display_name, metrics_dict), ...]"""
    print("\n" + "-" * 78)
    print(
        f"{'Planner':<24}{'align':>8}{'drift':>8}{'reversal':>10}"
        f"{'traj_score':>12}{'entropy':>10}"
    )
    print("-" * 78)
    for name, m in rows:
        print(
            f"{name:<24}"
            f"{m['alignment_score']:>8.2f}"
            f"{m['drift']:>8.3f}"
            f"{m['reversal_rate']:>10.2f}"
            f"{m['traj_level_score']:>12.3f}"
            f"{m['mean_entropy']:>10.3f}"
        )
    print("-" * 78)


EXAMPLE_1 = {
    "situation": "My dog passed away last week and I can't stop thinking about him.",
    "user_emotion": "sad",
    "user_turns": [
        "My dog passed away last week and I can't stop thinking about him.",
        "He was with me for 12 years. The house feels so empty.",
        "I keep expecting to hear him at the door when I come home.",
    ],
    "planner_outputs": {
        "baseline": [
            "Dogs are wonderful companions. You should try to get out more and maybe adopt another dog soon, it really helps.",
            "Twelve years is a long time. Have you considered donating his toys to a shelter? It's a productive thing to do.",
            "That's just muscle memory, it will fade in a few weeks. Try to stay busy.",
        ],
        "topk": [
            "I'm so sorry for your loss. Twelve years is a lifetime of love, and grief that deep is a reflection of how much he meant to you.",
            "That emptiness is so real. The absence of a companion who shaped your daily rhythms leaves a space nothing else quite fills.",
            "Those little moments of expectation are some of the hardest parts. It's your heart still reaching for him.",
        ],
        "lookahead": [
            "I'm so sorry. Losing a dog who's been with you that long is losing a daily companion, and that pain deserves to be felt fully.",
            "Twelve years of shared routines and quiet company — of course the silence is loud. That ache is love that has nowhere to land right now.",
            "Those almost-moments can catch you off guard. It shows how deeply he was woven into your life. Be gentle with yourself when they come.",
        ],
        "tot": [
            "I'm deeply sorry for your loss. A bond like that, built over twelve years, doesn't simply end — the grief you're feeling is a testament to how much love was there.",
            "An empty house after so much shared presence can feel almost unbearable. It's okay to let yourself miss him in every quiet corner.",
            "Those expectant moments are part of how love lingers. Let yourself feel them. In time, they'll shift from aching to tender, though never disappear entirely.",
        ],
    },
}

EXAMPLE_2 = {
    "situation": "I just got promoted at work after three years of grinding!",
    "user_emotion": "proud",
    "user_turns": [
        "I just got promoted at work after three years of grinding!",
        "My manager said my work on the Q3 launch was a huge part of the decision.",
        "I'm going to celebrate with my family tonight.",
    ],
    "planner_outputs": {
        "baseline": [
            "Congrats. Promotions can come with a lot more stress though, make sure you're ready for the workload increase.",
            "Q3 launches are usually stressful. Did it go smoothly or were there issues you had to fix?",
            "That sounds fine. Don't forget to stay humble about it at work tomorrow.",
        ],
        "topk": [
            "That's fantastic! Three years of consistent effort paying off like that is such a deserved moment. You should feel really proud.",
            "That recognition means so much — having your specific contribution called out shows the impact you made was visible and valued.",
            "What a perfect way to mark this! Sharing it with family makes the achievement even sweeter.",
        ],
        "lookahead": [
            "Congratulations! Three years of grinding is no small thing, and the fact that it paid off is a beautiful validation of your persistence.",
            "That's incredible — being recognized specifically for the Q3 launch shows your work genuinely moved the needle. You should let that sink in.",
            "That sounds wonderful. Celebrating with the people who supported you through those three years is the perfect way to honor the journey.",
        ],
        "tot": [
            "Congratulations! That's a huge milestone, and three years of steady dedication making it happen is genuinely admirable. You earned every bit of this.",
            "That's the kind of recognition that really lands — specific, tied to real impact. Your work mattered, and it's wonderful that it was seen that way.",
            "Celebrating with family is perfect. They've watched the grind from the inside — sharing the win with them makes it complete.",
        ],
    },
}

MOCK_EXAMPLES = [EXAMPLE_1, EXAMPLE_2]


def run_mock_mode():
    print("Loading emotion classifier (RoBERTa GoEmotions)...")
    classifier = EmotionClassifier()
    print(f"Classifier ready. {len(classifier.label2id)} emotion labels.\n")

    aggregate = {k: [] for k in PLANNER_ORDER}

    for example in MOCK_EXAMPLES:
        situation = example["situation"]
        user_emotion = example["user_emotion"]
        user_turns = example["user_turns"]
        assistant_target = get_assistant_target(user_emotion)
        n_turns = len(user_turns)
        target_trajectory = [assistant_target] * n_turns

        print("\n" + "=" * 78)
        print(f"Situation    : {situation}")
        print(f"User emotion : {user_emotion}")
        print(f"Assistant target (from assistant_targets.py): {assistant_target}")
        print(f"Target trajectory: {target_trajectory}")
        print("=" * 78)

        rows = []
        for planner_key in PLANNER_ORDER:
            assistant_turns = example["planner_outputs"][planner_key]
            metrics = evaluate_dialogue(assistant_turns, target_trajectory, classifier)
            aggregate[planner_key].append(metrics)

            print_per_planner_block(
                PLANNER_LABELS[planner_key], user_turns, assistant_turns, metrics
            )
            rows.append((PLANNER_LABELS[planner_key], metrics))

        print_comparison_table(rows)

    # Aggregate summary across examples
    print("\n" + "=" * 78)
    print("AGGREGATE SUMMARY across all demo examples")
    print("=" * 78)
    print(f"{'Planner':<24}{'align':>8}{'drift':>8}{'reversal':>10}{'traj_score':>12}")
    print("-" * 78)
    for planner_key in PLANNER_ORDER:
        ms = aggregate[planner_key]
        print(
            f"{PLANNER_LABELS[planner_key]:<24}"
            f"{np.mean([m['alignment_score'] for m in ms]):>8.2f}"
            f"{np.mean([m['drift'] for m in ms]):>8.3f}"
            f"{np.mean([m['reversal_rate'] for m in ms]):>10.2f}"
            f"{np.mean([m['traj_level_score'] for m in ms]):>12.3f}"
        )
    print("=" * 78)
    print(
        "\nHigher alignment + higher traj_score + lower drift/reversal = better "
        "trajectory-level emotional coherence."
    )


DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def run_real_llm_mode(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from experiments.run_baseline_generation import (
        load_conversations,
        generate_conversation,  # baseline
    )
    from experiments.topk_planner import generate_topk_conversation
    from experiments.lookahead_planner import generate_lookahead_conversation
    from experiments.tot_planner import generate_tot_conversation

    if not torch.cuda.is_available():
        print(
            "WARNING: no CUDA device detected. The planner modules call "
            "`.to('cuda')` directly, so this mode will fail without a GPU. "
            "Use `--mode mock` for a CPU-friendly demo instead."
        )

    print(f"Loading LLM: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(device_map="auto")
    if not args.no_4bit:
        model_kwargs.update(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()

    print("Loading emotion classifier...")
    classifier = EmotionClassifier()

    print(f"Loading {args.num_conversations} conversation(s) from EmpatheticDialogues")
    conversations = load_conversations(args.num_conversations)

    aggregate = {k: [] for k in PLANNER_ORDER}

    for i, conversation in enumerate(conversations):
        situation = conversation[0]["prompt"]
        user_emotion = conversation[0]["context"]
        n_assistant_turns = sum(1 for idx in range(len(conversation)) if idx % 2 != 0)
        assistant_target = get_assistant_target(user_emotion)
        target_trajectory = [assistant_target] * n_assistant_turns

        print("\n" + "=" * 78)
        print(f"Conversation {i + 1}/{len(conversations)}")
        print(f"Situation    : {situation}")
        print(f"User emotion : {user_emotion}")
        print(f"Assistant target: {assistant_target}")
        print("=" * 78)

        # 1. Baseline
        print("\n[1/4] Generating baseline...")
        baseline_turns = generate_conversation(conversation, tokenizer, model)
        baseline_metrics = evaluate_dialogue(
            baseline_turns, target_trajectory, classifier
        )
        print_per_planner_block("Baseline", None, baseline_turns, baseline_metrics)
        aggregate["baseline"].append(baseline_metrics)

        # 2. Top-k
        print("\n[2/4] Generating top-k (K=3)...")
        topk_turns, _ = generate_topk_conversation(
            conversation,
            tokenizer,
            model,
            classifier,
            target_trajectory,
            scorer_mode="classifier",
            warned_missing=set(),
        )
        topk_metrics = evaluate_dialogue(topk_turns, target_trajectory, classifier)
        print_per_planner_block("Top-k", None, topk_turns, topk_metrics)
        aggregate["topk"].append(topk_metrics)

        # 3. Lookahead
        print("\n[3/4] Generating lookahead (K=3, depth=2)...")
        lookahead_turns, _ = generate_lookahead_conversation(
            conversation,
            tokenizer,
            model,
            classifier,
            target_trajectory,
            lookahead_depth=2,
            k=3,
            scorer_mode="classifier",
            debug=False,
        )
        lookahead_metrics = evaluate_dialogue(
            lookahead_turns, target_trajectory, classifier
        )
        print_per_planner_block("Lookahead", None, lookahead_turns, lookahead_metrics)
        aggregate["lookahead"].append(lookahead_metrics)

        # 4. ToT
        print("\n[4/4] Generating ToT (K=3, beam=2, depth=2)...")
        tot_turns, _, _ = generate_tot_conversation(
            conversation,
            tokenizer,
            model,
            classifier,
            target_trajectory,
            debug=False,
        )
        tot_metrics = evaluate_dialogue(tot_turns, target_trajectory, classifier)
        print_per_planner_block("Tree-of-Thoughts", None, tot_turns, tot_metrics)
        aggregate["tot"].append(tot_metrics)

        print_comparison_table(
            [
                ("Baseline", baseline_metrics),
                ("Top-k", topk_metrics),
                ("Lookahead", lookahead_metrics),
                ("Tree-of-Thoughts", tot_metrics),
            ]
        )

    if len(conversations) > 1:
        print("\n" + "=" * 78)
        print(f"AGGREGATE over {len(conversations)} conversations")
        print("=" * 78)
        print(
            f"{'Planner':<24}{'align':>8}{'drift':>8}{'reversal':>10}{'traj_score':>12}"
        )
        print("-" * 78)
        for key in PLANNER_ORDER:
            ms = aggregate[key]
            print(
                f"{PLANNER_LABELS[key]:<24}"
                f"{np.mean([m['alignment_score'] for m in ms]):>8.2f}"
                f"{np.mean([m['drift'] for m in ms]):>8.3f}"
                f"{np.mean([m['reversal_rate'] for m in ms]):>10.2f}"
                f"{np.mean([m['traj_level_score'] for m in ms]):>12.3f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Demo for the empathetic trajectory planning project. "
            "Use --mode mock for a CPU-only demo, --mode real for end-to-end "
            "generation with an LLM."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
        help="mock = no LLM (CPU-friendly), real = end-to-end with an LLM.",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=1,
        help="(real mode) number of EmpatheticDialogues conversations to run.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="(real mode) HF model id for the generation LLM.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="(real mode) disable 4-bit quantization; use fp16 instead.",
    )
    args = parser.parse_args()

    if args.mode == "mock":
        run_mock_mode()
    else:
        run_real_llm_mode(args)


if __name__ == "__main__":
    main()
