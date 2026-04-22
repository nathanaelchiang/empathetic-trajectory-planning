"""
Top-k local reranking baseline.
Generates K candidate replies per turn and picks the best-scoring one.
This is *local* planning (single-turn lookahead), not trajectory planning —
label it as such when comparing against ToT / MCTS in the paper.
"""

import json
import random
import os
import re
import numpy as np
import torch
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM

from models.emotion_classifier import EmotionClassifier
from evaluation.trajectory import compute_drift
from experiments.run_baseline_generation import (
    load_conversations,
    get_trajectory_labels,
    compute_per_step_distances,
    compute_emotion_entropy,
    compute_reversal_rate,
    compute_peak_drift_turn,
    compute_trajectory_alignment,
    compute_trajectory_level_score,
)
from emotion.assistant_targets import get_assistant_target

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# model_name = "Qwen/Qwen2.5-3B-Instruct"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name = "Qwen/Qwen2.5-7B-Instruct"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
STRIP_THINK = "R1" in model_name

JUDGE_MODEL = "claude-sonnet-4-20250514"
K = 5
MAX_NEW_TOKENS = 150

# choose: "classifier", "llm_judge", or "both"
SCORER_MODE = "classifier"


def strip_think_tags(text):
    """Remove <think>...</think> blocks produced by reasoning models (e.g. DeepSeek-R1)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Candidate generation
def generate_candidates(messages, tokenizer, model, k=K):
    """Sample k candidate replies from the model, one at a time."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {kk: v.to("cuda") for kk, v in inputs.items()}

    candidates = []
    for _ in range(k):
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        candidates.append(strip_think_tags(decoded) if STRIP_THINK else decoded.strip())
    return candidates


# Scorers
def score_with_classifier(candidates, target_emotion, classifier, warned_missing=None):
    """Return P(target_emotion) for each candidate; returns zeros if label is unknown."""
    label_index = classifier.label2id.get(target_emotion, None)

    if label_index is None:
        if warned_missing is not None and target_emotion not in warned_missing:
            print(f"  [WARN] '{target_emotion}' not in classifier labels.")
            warned_missing.add(target_emotion)
        return [0.0] * len(candidates)

    probs = classifier.predict_proba(candidates)
    return [float(p[label_index]) for p in probs]


def score_with_llm_judge(candidates, target_emotion, dialogue_history):
    """Score candidates using an LLM judge (Anthropic API); returns 0.0 on parse failure."""
    import anthropic

    client = anthropic.Anthropic()

    history_str = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in dialogue_history
        if m["role"] != "system"
    )

    scores = []
    for candidate in candidates:
        prompt = f"""You are evaluating a candidate response in an empathetic dialogue.

Dialogue so far:
{history_str}

Candidate assistant response:
\"{candidate}\"

Target emotional tone for this turn: {target_emotion}

Rate this response on a scale from 0.0 to 1.0 based on:
1. Emotional coherence with the prior conversation
2. Alignment with the target emotional tone: {target_emotion}

Respond with ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}"""

        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            scores.append(float(json.loads(response.content[0].text)["score"]))
        except Exception:
            scores.append(0.0)

    return scores


def normalize_scores(x):
    """Min-max normalize an array of scores to [0, 1]; returns uniform weights if all values are equal."""
    x = np.array(x, dtype=float)
    r = x.max() - x.min()
    if r > 0:
        return (x - x.min()) / r
    return np.ones_like(x) / len(x)


def score_candidates(
    candidates,
    target_emotion,
    classifier,
    dialogue_history,
    scorer_mode=SCORER_MODE,
    warned_missing=None,
):
    """Score and normalize candidates using the selected scorer(s).

    Args:
        candidates: List of candidate reply strings.
        target_emotion: Target emotion label for this turn.
        classifier: EmotionClassifier instance.
        dialogue_history: Message history list used by the LLM judge.
        scorer_mode: One of "classifier", "llm_judge", or "both".
        warned_missing: Optional set to accumulate unknown emotion label warnings.

    Returns:
        List of normalized float scores in [0, 1], one per candidate.
    """
    if scorer_mode == "classifier":
        clf = score_with_classifier(
            candidates, target_emotion, classifier, warned_missing
        )
        return list(normalize_scores(clf))

    if scorer_mode == "llm_judge":
        llm = score_with_llm_judge(candidates, target_emotion, dialogue_history)
        return list(normalize_scores(llm))

    if scorer_mode == "both":
        clf = normalize_scores(
            score_with_classifier(
                candidates, target_emotion, classifier, warned_missing
            )
        )
        llm = normalize_scores(
            score_with_llm_judge(candidates, target_emotion, dialogue_history)
        )
        return list((clf + llm) / 2)

    raise ValueError(f"Invalid scorer_mode: {scorer_mode}")


# Conversation generation


def generate_topk_conversation(
    conversation,
    tokenizer,
    model,
    classifier,
    target_trajectory,
    scorer_mode=SCORER_MODE,
    warned_missing=None,
):
    """Generate assistant turns using single-turn top-k reranking.

    For each assistant turn, samples K candidates and selects the one with the
    highest score against the current target emotion. No lookahead is performed.

    Args:
        conversation: List of turn dicts with an 'utterance' key.
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace causal LM.
        classifier: EmotionClassifier for scoring.
        target_trajectory: List of target emotion labels, one per assistant turn.
        scorer_mode: One of "classifier", "llm_judge", or "both".
        warned_missing: Optional set to accumulate unknown emotion label warnings.

    Returns:
        generated_turns: List of selected assistant reply strings.
        candidate_log: Per-turn dicts recording candidates, scores, and selection metadata.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a supportive and empathetic conversational partner. "
                f"Guide the conversation through these emotional tones in order: "
                f"{', '.join(target_trajectory)}."
            ),
        }
    ]

    generated_turns = []
    candidate_log = []
    assistant_idx = 0

    for idx, turn in enumerate(conversation):
        utterance = turn["utterance"].strip()

        if idx % 2 == 0:
            messages.append({"role": "user", "content": utterance})
        else:
            target_emotion = (
                target_trajectory[assistant_idx]
                if assistant_idx < len(target_trajectory)
                else target_trajectory[-1]
            )

            candidates = generate_candidates(messages, tokenizer, model, k=K)
            scores = score_candidates(
                candidates,
                target_emotion,
                classifier,
                messages,
                scorer_mode=scorer_mode,
                warned_missing=warned_missing,
            )

            best_idx = int(np.argmax(scores))
            best_reply = candidates[best_idx]

            candidate_log.append(
                {
                    "turn": assistant_idx,
                    "target_emotion": target_emotion,
                    "candidates": candidates,
                    "scores": scores,
                    "selected_idx": best_idx,
                    "score_variance": float(np.var(scores)),
                    "scorer_mode": scorer_mode,
                }
            )

            generated_turns.append(best_reply)
            messages.append({"role": "assistant", "content": best_reply})
            assistant_idx += 1

    return generated_turns, candidate_log


def main():
    """Load data and model, run top-k reranking over all conversations, and save results."""
    warned_missing = set()
    classifier = EmotionClassifier()
    print("Classifier labels:")
    print(sorted(classifier.label2id.keys()))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model.eval()

    conversations = load_conversations(100)
    results = []

    for i, conversation in enumerate(conversations):
        print(f"\nRunning {i + 1}/100")

        situation = conversation[0]["prompt"]
        gold_emotion_label = conversation[0]["context"]

        n_assistant_turns = sum(1 for idx in range(len(conversation)) if idx % 2 != 0)
        assistant_target = get_assistant_target(gold_emotion_label)
        target_trajectory = [assistant_target] * n_assistant_turns

        generated_turns, candidate_log = generate_topk_conversation(
            conversation,
            tokenizer,
            model,
            classifier,
            target_trajectory,
            scorer_mode=SCORER_MODE,
            warned_missing=warned_missing,
        )

        # --------------------------------------------------
        # DEBUG: print first 5 conversations in detail
        # --------------------------------------------------
        if i < 5:
            print("\n------------------------------")
            print(f"Situation: {situation}")
            print(f"User emotion: {gold_emotion_label}")
            print(f"Assistant target: {assistant_target}")
            print("\nGenerated dialogue:")
            for t_idx, reply in enumerate(generated_turns):
                print(f"  Turn {t_idx + 1}: {reply}")

            print("\nPredicted trajectory:")
            print(get_trajectory_labels(generated_turns, classifier))
            print("------------------------------\n")

        labels = get_trajectory_labels(generated_turns, classifier)
        drift = compute_drift(generated_turns, classifier)
        probs = classifier.predict_proba(generated_turns)
        per_step_dist = compute_per_step_distances(probs)
        entropies = compute_emotion_entropy(probs)
        reversal_rate = compute_reversal_rate(labels)
        peak_drift_turn = compute_peak_drift_turn(per_step_dist)
        alignment_score = compute_trajectory_alignment(labels, target_trajectory)
        traj_level_score = compute_trajectory_level_score(
            labels, target_trajectory, per_step_dist
        )
        mean_score_var = (
            float(np.mean([c["score_variance"] for c in candidate_log]))
            if candidate_log
            else 0.0
        )

        results.append(
            {
                "situation": situation,
                "target_trajectory": target_trajectory,
                "generated_dialogue": generated_turns,
                "trajectory": labels,
                "drift": drift,
                "per_step_distances": per_step_dist,
                "per_turn_entropy": entropies,
                "reversal_rate": reversal_rate,
                "peak_drift_turn": peak_drift_turn,
                "alignment_score": alignment_score,
                "traj_level_score": traj_level_score,
                "mean_score_variance": mean_score_var,
                "candidate_log": candidate_log,
                "metadata": {
                    "model": model_name,
                    "k": K,
                    "scorer": SCORER_MODE,
                    "planner": "topk_reranking",
                    "temperature": 0.8,
                    "seed": 42,
                },
            }
        )

        print(
            f"  Drift: {drift:.4f} | Alignment: {alignment_score:.2f} | "
            f"Reversal: {reversal_rate:.2f} | Traj: {traj_level_score:.4f}"
        )

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/topk_results_{SCORER_MODE}_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to: {filename}")

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
        ("Score variance", [x["mean_score_variance"] for x in results]),
    ]:
        arr = np.array(arr)
        print(
            f"{label:22s}  mean={arr.mean():.4f}  std={arr.std():.4f}"
            f"  min={arr.min():.4f}  max={arr.max():.4f}"
        )

    if warned_missing:
        print("\n[MISSING FROM TARGET_MAP]")
        for label in sorted(warned_missing):
            print(f'  "{label}": "",')


if __name__ == "__main__":
    main()
