import json
import math
import random
import os
import argparse
import numpy as np
import torch
from datetime import datetime
from copy import deepcopy

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

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
JUDGE_MODEL = "claude-sonnet-4-20250514"

N_SIMULATIONS = 20
N_ROLLOUT = 2
EXPLORATION_C = 1.4
K_EXPAND = 3

# choose default: "classifier", "llm_judge", or "both"
SCORER_MODE = "classifier"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_device(model):
    return next(model.parameters()).device


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def sample_reply(messages, tokenizer, model):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    device = get_model_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_classifier(reply, target_emotion, classifier):
    if target_emotion not in classifier.label2id:
        raise ValueError(
            f"Target emotion '{target_emotion}' "
            f"is not in classifier label space.\n"
            f"Available labels: {sorted(classifier.label2id.keys())}"
        )

    probs = classifier.predict_proba([reply])
    label_index = classifier.label2id[target_emotion]
    return float(probs[0][label_index])


def score_llm_judge(reply, target_emotion, dialogue_history):
    import anthropic

    history_str = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in dialogue_history if m["role"] != "system"
    )

    prompt = f"""You are evaluating a candidate response in an empathetic dialogue.

Dialogue so far:
{history_str}

Candidate assistant response:
"{reply}"

Target emotional tone: {target_emotion}

Rate 0.0–1.0 on emotional coherence and alignment with the target tone.
Respond ONLY with JSON: {{"score": <float>, "reason": "<one sentence>"}}"""

    response = anthropic.Anthropic().messages.create(
        model=JUDGE_MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        return float(json.loads(response.content[0].text)["score"])
    except Exception:
        return 0.0


def score_node(reply, target_emotion, classifier, dialogue_history, scorer_mode=SCORER_MODE):
    if scorer_mode == "classifier":
        return score_classifier(reply, target_emotion, classifier)

    if scorer_mode == "llm_judge":
        return score_llm_judge(reply, target_emotion, dialogue_history)

    if scorer_mode == "both":
        clf = score_classifier(reply, target_emotion, classifier)
        llm = score_llm_judge(reply, target_emotion, dialogue_history)
        return (clf + llm) / 2.0

    raise ValueError(f"Invalid scorer_mode: {scorer_mode}")


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class MCTSNode:
    def __init__(self, messages, reply=None, parent=None, target_emotion=None):
        self.messages = messages
        self.reply = reply
        self.parent = parent
        self.target_emotion = target_emotion
        self.children = []
        self.visits = 0
        self.total_value = 0.0

    @property
    def value(self):
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb1(self, parent_visits, c=EXPLORATION_C):
        if self.visits == 0:
            return float("inf")
        return self.value + c * math.sqrt(math.log(max(parent_visits, 1)) / self.visits)

    def is_leaf(self):
        return len(self.children) == 0

    def best_child(self, c=EXPLORATION_C):
        return max(self.children, key=lambda n: n.ucb1(self.visits, c))

    def best_child_greedy(self):
        return max(self.children, key=lambda n: n.value)


# ---------------------------------------------------------------------------
# MCTS phases
# ---------------------------------------------------------------------------

def selection(root):
    node = root
    while not node.is_leaf():
        node = node.best_child()
    return node


def expansion(node, tokenizer, model, future_user_turns=None, k=K_EXPAND):
    future_user_turns = future_user_turns or []

    # Avoid duplicate expansion
    if node.children:
        return

    for _ in range(k):
        base_messages = deepcopy(node.messages)
        last_role = base_messages[-1]["role"] if base_messages else None

        if last_role == "assistant":
            user_utt = future_user_turns[0] if future_user_turns else "Please continue."
            base_messages.append({"role": "user", "content": user_utt})

        reply = sample_reply(base_messages, tokenizer, model)
        new_msgs = base_messages + [{"role": "assistant", "content": reply}]

        child = MCTSNode(
            new_msgs,
            reply=reply,
            parent=node,
            target_emotion=node.target_emotion,
        )
        node.children.append(child)


def rollout(
    node,
    tokenizer,
    model,
    classifier,
    future_targets,
    future_user_turns,
    depth=N_ROLLOUT,
    scorer_mode=SCORER_MODE
):
    messages = deepcopy(node.messages)
    total_score = 0.0
    future_user_idx = 0

    for d in range(depth):
        last_role = messages[-1]["role"] if messages else None

        if last_role == "assistant":
            if future_user_idx < len(future_user_turns):
                user_utt = future_user_turns[future_user_idx]
            else:
                user_utt = "Please continue."
            messages.append({"role": "user", "content": user_utt})
            future_user_idx += 1

        target = (
            future_targets[d]
            if d < len(future_targets)
            else (future_targets[-1] if future_targets else "neutral")
        )

        reply = sample_reply(messages, tokenizer, model)

        total_score += score_node(
            reply,
            target,
            classifier,
            messages,
            scorer_mode=scorer_mode,
        )

        messages.append({"role": "assistant", "content": reply})

    return total_score / max(depth, 1)


def backpropagation(node, value):
    while node is not None:
        node.visits += 1
        node.total_value += value
        node = node.parent


# ---------------------------------------------------------------------------
# One MCTS turn
# ---------------------------------------------------------------------------

def mcts_select_reply(
    messages,
    target_emotion,
    future_targets,
    future_user_turns,
    tokenizer,
    model,
    classifier,
    n_sims=N_SIMULATIONS,
    k_expand=K_EXPAND,
    n_rollout=N_ROLLOUT,
    scorer_mode=SCORER_MODE
):
    root = MCTSNode(messages, target_emotion=target_emotion)

    for _ in range(n_sims):
        # 1. Selection
        leaf = selection(root)

        # 2. Expansion: expand leaf if needed
        if leaf.is_leaf():
            expansion(
                leaf,
                tokenizer,
                model,
                future_user_turns=future_user_turns,
                k=k_expand,
            )

        # 3. Choose a child to rollout from
        if leaf.children:
            unvisited = [c for c in leaf.children if c.visits == 0]
            rollout_node = random.choice(unvisited) if unvisited else leaf.best_child()
        else:
            rollout_node = leaf

        # 4. Immediate score
        immediate = (
            score_node(
                rollout_node.reply,
                target_emotion,
                classifier,
                messages,
                scorer_mode=scorer_mode,
            )
            if rollout_node.reply else 0.0
        )

        # 5. Future rollout score
        future = rollout(
            rollout_node,
            tokenizer,
            model,
            classifier,
            future_targets,
            future_user_turns,
            depth=n_rollout,
            scorer_mode=scorer_mode,
        )

        value = 0.6 * immediate + 0.4 * future

        # 6. Backprop
        backpropagation(rollout_node, value)

    if not root.children:
        expansion(
            root,
            tokenizer,
            model,
            future_user_turns=future_user_turns,
            k=k_expand,
        )

    best = root.best_child_greedy()
    child_values = [c.value for c in root.children] if root.children else [0.0]

    tree_stats = {
        "n_children": len(root.children),
        "best_value": best.value,
        "best_visits": best.visits,
        "value_variance": float(np.var(child_values)),
        "scorer_mode": scorer_mode,
    }
    return best.reply, tree_stats


# ---------------------------------------------------------------------------
# Full conversation generation
# ---------------------------------------------------------------------------

def generate_mcts_conversation(
    conversation,
    tokenizer,
    model,
    classifier,
    target_trajectory,
    scorer_mode=SCORER_MODE,
    n_simulations=N_SIMULATIONS,
    n_rollout=N_ROLLOUT,
    k_expand=K_EXPAND
):
    messages = [
        {"role": "system", "content": (
            "You are a supportive and empathetic conversational partner. "
            f"Guide the conversation through these emotional tones in order: "
            f"{', '.join(target_trajectory)}."
        )}
    ]

    generated_turns = []
    mcts_log = []
    assistant_idx = 0

    user_turns = [t["utterance"].strip() for i, t in enumerate(conversation) if i % 2 == 0]
    user_turn_idx = 0

    for idx, turn in enumerate(conversation):
        utterance = turn["utterance"].strip()

        if idx % 2 == 0:
            messages.append({"role": "user", "content": utterance})
            user_turn_idx += 1
        else:
            target_emotion = (
                target_trajectory[assistant_idx]
                if assistant_idx < len(target_trajectory)
                else target_trajectory[-1]
            )
            future_targets = target_trajectory[assistant_idx + 1: assistant_idx + 1 + n_rollout]
            future_user_turns = user_turns[user_turn_idx: user_turn_idx + n_rollout]

            best_reply, stats = mcts_select_reply(
                messages,
                target_emotion,
                future_targets,
                future_user_turns,
                tokenizer,
                model,
                classifier,
                n_sims=n_simulations,
                k_expand=k_expand,
                n_rollout=n_rollout,
                scorer_mode=scorer_mode,
            )

            mcts_log.append({
                "turn": assistant_idx,
                "target_emotion": target_emotion,
                "reply": best_reply,
                **stats,
            })

            generated_turns.append(best_reply)
            messages.append({"role": "assistant", "content": best_reply})
            assistant_idx += 1

    return generated_turns, mcts_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scorer",
        choices=["classifier", "llm_judge", "both"],
        default=SCORER_MODE,
    )
    parser.add_argument("--num-conversations", type=int, default=3)
    parser.add_argument("--n-simulations", type=int, default=N_SIMULATIONS)
    parser.add_argument("--n-rollout", type=int, default=N_ROLLOUT)
    parser.add_argument("--k-expand", type=int, default=K_EXPAND)
    args = parser.parse_args()

    scorer_mode = args.scorer
    n_simulations = args.n_simulations
    n_rollout = args.n_rollout
    k_expand = args.k_expand

    classifier = EmotionClassifier()
    print("Classifier labels:")
    print(sorted(classifier.label2id.keys()))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    conversations = load_conversations(args.num_conversations)
    results = []

    for i, conversation in enumerate(conversations):
        print(f"\nRunning {i+1}/{len(conversations)}")

        situation = conversation[0]["prompt"]
        gold_emotion_label = conversation[0]["context"]
        n_assistant_turns = sum(1 for idx in range(len(conversation)) if idx % 2 != 0)
        assistant_target = get_assistant_target(gold_emotion_label)
        target_trajectory = [assistant_target] * n_assistant_turns

        generated_turns, mcts_log = generate_mcts_conversation(
            conversation,
            tokenizer,
            model,
            classifier,
            target_trajectory,
            scorer_mode=scorer_mode,
            n_simulations=n_simulations,
            n_rollout=n_rollout,
            k_expand=k_expand,
        )

        labels = get_trajectory_labels(generated_turns, classifier)
        drift = compute_drift(generated_turns, classifier)
        probs = classifier.predict_proba(generated_turns)
        per_step_dist = compute_per_step_distances(probs)
        entropies = compute_emotion_entropy(probs)
        reversal_rate = compute_reversal_rate(labels)
        peak_drift_turn = compute_peak_drift_turn(per_step_dist)
        alignment_score = compute_trajectory_alignment(labels, target_trajectory)
        traj_level_score = compute_trajectory_level_score(labels, target_trajectory, per_step_dist)
        mean_val_var = float(np.mean([m["value_variance"] for m in mcts_log])) if mcts_log else 0.0

        results.append({
            "situation": situation,
            "target_trajectory": target_trajectory,
            "user_emotion": gold_emotion_label,
            "assistant_target": assistant_target,
            "generated_dialogue": generated_turns,
            "trajectory": labels,
            "drift": drift,
            "per_step_distances": per_step_dist,
            "per_turn_entropy": entropies,
            "reversal_rate": reversal_rate,
            "peak_drift_turn": peak_drift_turn,
            "alignment_score": alignment_score,
            "traj_level_score": traj_level_score,
            "mean_value_variance": mean_val_var,
            "mcts_log": mcts_log,
            "metadata": {
                "model": model_name,
                "n_simulations": n_simulations,
                "n_rollout": n_rollout,
                "k_expand": k_expand,
                "exploration_c": EXPLORATION_C,
                "planner": "mcts",
                "scorer": scorer_mode,
                "temperature": 0.8,
                "seed": 42,
            }
        })

        print(
            f"  User emotion: {gold_emotion_label} | "
            f"Assistant target: {assistant_target}"
        )
        print(
            f"  Drift: {drift:.4f} | Alignment: {alignment_score:.2f} | "
            f"Reversal: {reversal_rate:.2f} | Traj: {traj_level_score:.4f}"
        )

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/mcts_results_{scorer_mode}_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to: {filename}")

    for label, arr in [
        ("Drift", [x["drift"] for x in results]),
        ("Alignment score", [x["alignment_score"] for x in results]),
        ("Reversal rate", [x["reversal_rate"] for x in results]),
        ("Mean entropy", [np.mean(x["per_turn_entropy"]) for x in results]),
        ("Mean step distance", [
            np.mean(x["per_step_distances"]) if x["per_step_distances"] else 0.0
            for x in results
        ]),
        ("Traj level score", [x["traj_level_score"] for x in results]),
        ("Value variance", [x["mean_value_variance"] for x in results]),
    ]:
        arr = np.array(arr)
        print(
            f"{label:22s}  mean={arr.mean():.4f}  std={arr.std():.4f}"
            f"  min={arr.min():.4f}  max={arr.max():.4f}"
        )


if __name__ == "__main__":
    main()