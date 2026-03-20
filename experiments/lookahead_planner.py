"""
Lookahead reranking planner.
Generates K candidate replies per turn, then for each candidate simulates
a cheap linear rollout (1-2 future turns, single sample, no branching)
and scores the resulting mini-trajectory. Picks the candidate whose
lookahead trajectory scores best.

This sits between top-k (single-turn) and ToT (tree-structured):
  no planning (baseline) → single-turn (top-k) → linear lookahead → tree (ToT)
"""

import json
import random
import os
import argparse
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

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"

K = 5  # number of candidates per turn
LOOKAHEAD_DEPTH = 2  # how many future assistant turns to simulate per candidate
# MAX_NEW_TOKENS = 64  # for both candidates and rollout replies
MAX_NEW_TOKENS = 150

# choose: "classifier", "llm_judge", or "both"
SCORER_MODE = "classifier"
JUDGE_MODEL = "claude-sonnet-4-20250514"


def get_model_device(model):
    return next(model.parameters()).device


# Generation helpers
def generate_candidates(messages, tokenizer, model, k=K):
    """Generate K candidate replies via batch sampling."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    device = get_model_device(model)
    inputs = {kk: v.to(device) for kk, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=k,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    candidates = []
    for seq in output_ids:
        new_tokens = seq[prompt_len:]
        candidates.append(
            tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        )
    return candidates


def sample_single_reply(messages, tokenizer, model):
    """Sample a single reply cheaply (used for rollout turns)."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    device = get_model_device(model)
    inputs = {kk: v.to(device) for kk, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Scoring helpers


def score_classifier_batch(replies, target_emotion, classifier):
    """Return P(target_emotion) for each reply."""
    if target_emotion not in classifier.label2id:
        raise ValueError(
            f"Target '{target_emotion}' not in classifier label space.\n"
            f"Available: {sorted(classifier.label2id.keys())}"
        )
    probs = classifier.predict_proba(replies)
    label_index = classifier.label2id[target_emotion]
    return [float(p[label_index]) for p in probs]


def score_llm_judge_batch(replies, target_emotion, dialogue_history):
    """Score replies using an LLM judge (Anthropic API)."""
    import anthropic

    client = anthropic.Anthropic()

    history_str = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in dialogue_history
        if m["role"] != "system"
    )

    scores = []
    for reply in replies:
        prompt = (
            f"You are evaluating a candidate response in an empathetic dialogue.\n\n"
            f"Dialogue so far:\n{history_str}\n\n"
            f'Candidate assistant response:\n"{reply}"\n\n'
            f"Target emotional tone: {target_emotion}\n\n"
            f"Rate 0.0-1.0 based on emotional coherence and alignment with the target.\n"
            f'Respond with ONLY a JSON object: {{"score": <float>, "reason": "<one sentence>"}}'
        )
        try:
            response = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=64,
                messages=[{"role": "user", "content": prompt}],
            )
            scores.append(float(json.loads(response.content[0].text)["score"]))
        except Exception:
            scores.append(0.0)
    return scores


def normalize_scores(x):
    x = np.array(x, dtype=float)
    r = x.max() - x.min()
    if r > 0:
        return (x - x.min()) / r
    return np.ones_like(x) / len(x)


def trajectory_score_from_replies_targets(replies, targets, classifier):
    """
    Compute trajectory-level score from a list of assistant replies and
    their corresponding target emotion labels.
    """
    if not replies:
        return 0.0
    if len(replies) == 1:
        if not targets:
            return 0.0
        probs = classifier.predict_proba(replies)
        label_idx = classifier.label2id.get(targets[0])
        return float(probs[0][label_idx]) if label_idx is not None else 0.0

    labels = get_trajectory_labels(replies, classifier)
    probs = classifier.predict_proba(replies)
    per_step = compute_per_step_distances(probs)
    return compute_trajectory_level_score(labels, targets, per_step)


# Lookahead rollout


def rollout_candidate(
    candidate_reply,
    base_messages,
    future_user_turns,
    future_targets,
    tokenizer,
    model,
):
    """
    Given a candidate reply, simulate a cheap linear rollout by injecting
    real future user turns and sampling single assistant replies.

    Returns the list of all assistant replies in the rollout
    (starting with candidate_reply) and the corresponding target labels.

    future_user_turns: list of real user utterances for upcoming turns
    future_targets:    target emotions for rollout assistant turns
                       (index 0 = target for the candidate itself)
    """
    rollout_replies = [candidate_reply]
    rollout_targets = [future_targets[0]] if future_targets else []

    # Build message history including the candidate
    msgs = list(base_messages) + [{"role": "assistant", "content": candidate_reply}]

    # Simulate future turns
    for step in range(len(future_user_turns)):
        # Inject the real user turn
        user_utt = future_user_turns[step]
        msgs = msgs + [{"role": "user", "content": user_utt}]

        # Sample a single assistant reply (cheap, no branching)
        rollout_reply = sample_single_reply(msgs, tokenizer, model)
        rollout_replies.append(rollout_reply)

        # Track the target for this rollout step
        target_idx = step + 1  # +1 because index 0 is the candidate itself
        if target_idx < len(future_targets):
            rollout_targets.append(future_targets[target_idx])
        else:
            rollout_targets.append(future_targets[-1] if future_targets else "neutral")

        msgs = msgs + [{"role": "assistant", "content": rollout_reply}]

    return rollout_replies, rollout_targets


def score_with_lookahead(
    candidates,
    base_messages,
    future_user_turns,
    future_targets,
    prior_replies,
    prior_targets,
    tokenizer,
    model,
    classifier,
    scorer_mode=SCORER_MODE,
    debug=False,
):
    """
    For each candidate, do a linear rollout and compute the trajectory score
    over [prior_replies + rollout_replies] vs [prior_targets + rollout_targets].

    Returns:
        scores: list of floats (one per candidate)
        debug_info: list of dicts with rollout details (if debug=True)
    """
    scores = []
    debug_info = [] if debug else None

    for cand_idx, candidate in enumerate(candidates):
        rollout_replies, rollout_targets = rollout_candidate(
            candidate_reply=candidate,
            base_messages=base_messages,
            future_user_turns=future_user_turns,
            future_targets=future_targets,
            tokenizer=tokenizer,
            model=model,
        )

        # Full trajectory = prior conversation turns + this rollout
        full_replies = prior_replies + rollout_replies
        full_targets = prior_targets + rollout_targets

        traj_score = trajectory_score_from_replies_targets(
            full_replies, full_targets, classifier
        )
        scores.append(traj_score)

        if debug:
            debug_info.append(
                {
                    "candidate_idx": cand_idx,
                    "candidate": candidate,
                    "rollout_replies": rollout_replies,
                    "rollout_targets": rollout_targets,
                    "full_trajectory_length": len(full_replies),
                    "trajectory_score": float(traj_score),
                }
            )

    return scores, debug_info


# Conversation generation with lookahead
def generate_lookahead_conversation(
    conversation,
    tokenizer,
    model,
    classifier,
    target_trajectory,
    lookahead_depth=LOOKAHEAD_DEPTH,
    k=K,
    scorer_mode=SCORER_MODE,
    debug=False,
):
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
    lookahead_log = []
    assistant_idx = 0

    # Pre-extract all user turns for lookahead access
    user_turns = [
        t["utterance"].strip() for i, t in enumerate(conversation) if i % 2 == 0
    ]
    user_turn_idx = 0

    for idx, turn in enumerate(conversation):
        utterance = turn["utterance"].strip()

        if idx % 2 == 0:
            messages.append({"role": "user", "content": utterance})
            user_turn_idx += 1
        else:
            # --- Generate K candidates for this turn ---
            candidates = generate_candidates(messages, tokenizer, model, k=k)

            # --- Determine lookahead window ---
            # Future user turns available after the current assistant reply
            remaining_user_turns = user_turns[user_turn_idx:]
            effective_lookahead = min(lookahead_depth, len(remaining_user_turns))

            future_user_turns = remaining_user_turns[:effective_lookahead]

            # Target emotions for [current turn, +lookahead turns]
            future_targets = target_trajectory[
                assistant_idx : assistant_idx + 1 + effective_lookahead
            ]
            # Pad if we're near the end
            while len(future_targets) < 1 + effective_lookahead:
                future_targets.append(
                    target_trajectory[-1] if target_trajectory else "neutral"
                )

            # Prior replies/targets for full trajectory scoring
            prior_replies = list(generated_turns)
            prior_targets = [
                target_trajectory[j]
                if j < len(target_trajectory)
                else target_trajectory[-1]
                for j in range(assistant_idx)
            ]

            # --- Score each candidate via lookahead rollout ---
            scores, rollout_debug = score_with_lookahead(
                candidates=candidates,
                base_messages=messages,
                future_user_turns=future_user_turns,
                future_targets=future_targets,
                prior_replies=prior_replies,
                prior_targets=prior_targets,
                tokenizer=tokenizer,
                model=model,
                classifier=classifier,
                scorer_mode=scorer_mode,
                debug=debug,
            )

            best_idx = int(np.argmax(scores))
            best_reply = candidates[best_idx]

            lookahead_log.append(
                {
                    "turn": assistant_idx,
                    "target_emotion": future_targets[0],
                    "effective_lookahead": effective_lookahead,
                    "candidates": candidates,
                    "scores": scores,
                    "selected_idx": best_idx,
                    "score_variance": float(np.var(scores)),
                    "scorer_mode": scorer_mode,
                    **({"rollout_debug": rollout_debug} if debug else {}),
                }
            )

            generated_turns.append(best_reply)
            messages.append({"role": "assistant", "content": best_reply})
            assistant_idx += 1

    return generated_turns, lookahead_log


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-conversations", type=int, default=100)
    parser.add_argument("--lookahead-depth", type=int, default=LOOKAHEAD_DEPTH)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument(
        "--scorer",
        type=str,
        default=SCORER_MODE,
        choices=["classifier", "llm_judge", "both"],
    )
    args = parser.parse_args()

    classifier = EmotionClassifier()
    print("Classifier labels:")
    print(sorted(classifier.label2id.keys()))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model.eval()

    conversations = load_conversations(args.num_conversations)
    results = []

    for i, conversation in enumerate(conversations):
        print(f"\nRunning {i + 1}/{len(conversations)}")

        situation = conversation[0]["prompt"]
        gold_emotion_label = conversation[0]["context"]
        n_assistant_turns = sum(1 for idx in range(len(conversation)) if idx % 2 != 0)
        assistant_target = get_assistant_target(gold_emotion_label)
        target_trajectory = [assistant_target] * n_assistant_turns

        generated_turns, lookahead_log = generate_lookahead_conversation(
            conversation=conversation,
            tokenizer=tokenizer,
            model=model,
            classifier=classifier,
            target_trajectory=target_trajectory,
            lookahead_depth=args.lookahead_depth,
            k=args.k,
            scorer_mode=args.scorer,
            debug=(i < 5),
        )

        # DEBUG: print first 5 conversations
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

        # --- Compute all metrics (same as other planners) ---
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
            float(np.mean([c["score_variance"] for c in lookahead_log]))
            if lookahead_log
            else 0.0
        )
        mean_lookahead_score = (
            float(np.mean([max(c["scores"]) for c in lookahead_log]))
            if lookahead_log
            else 0.0
        )

        results.append(
            {
                "situation": situation,
                "user_emotion": gold_emotion_label,
                "assistant_target": assistant_target,
                "target_trajectory": target_trajectory,
                "generated_dialogue": generated_turns,
                "trajectory": labels,
                "drift": float(drift),
                "per_step_distances": per_step_dist,
                "per_turn_entropy": entropies,
                "reversal_rate": float(reversal_rate),
                "peak_drift_turn": int(peak_drift_turn),
                "alignment_score": float(alignment_score),
                "traj_level_score": float(traj_level_score),
                "mean_score_variance": float(mean_score_var),
                "mean_lookahead_score": float(mean_lookahead_score),
                "lookahead_log": lookahead_log,
                "metadata": {
                    "model": model_name,
                    "k": args.k,
                    "lookahead_depth": args.lookahead_depth,
                    "scorer": args.scorer,
                    "planner": "lookahead_reranking",
                    "temperature": 0.8,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "seed": 42,
                },
            }
        )

        print(
            f"  Drift: {drift:.4f} | Alignment: {alignment_score:.2f} | "
            f"Reversal: {reversal_rate:.2f} | Traj: {traj_level_score:.4f}"
        )

    # --- Save results ---
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/lookahead_results_{args.scorer}_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {filename}")
    print("\n=== Aggregate Summary ===")

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
        ("Mean lookahead score", [x["mean_lookahead_score"] for x in results]),
    ]:
        arr = np.array(arr)
        print(
            f"{label:22s}  mean={arr.mean():.4f}  std={arr.std():.4f}"
            f"  min={arr.min():.4f}  max={arr.max():.4f}"
        )


if __name__ == "__main__":
    main()
