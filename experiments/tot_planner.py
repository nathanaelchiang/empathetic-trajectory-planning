import json
import random
import os
import argparse
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

K_BRANCHES = 3
BEAM_WIDTH = 2
DEPTH = 2
# DEPTH = 3
# MAX_NEW_TOKENS = 64
MAX_NEW_TOKENS = 150
# MAX_NEW_TOKENS = 512
# MAX_NEW_TOKENS = 200

# BEAM_WIDTH = 3
# K_BRANCHES = 4
# DEPTH = 2

# BEAM_WIDTH = 4
# K_BRANCHES = 4

# choose default: "classifier", "llm_judge", or "both"
SCORER_MODE = "classifier"


def strip_think_tags(text):
    """Remove <think>...</think> blocks produced by reasoning models (e.g. DeepSeek-R1)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_model_device(model):
    """Return the device of the first parameter of a model."""
    return next(model.parameters()).device


# Generation helper
def sample_replies(messages, tokenizer, model, n=1, max_new_tokens=MAX_NEW_TOKENS):
    """
    Batch-sample n replies from the same prompt in one generate() call.
    Much faster than calling generate n separate times.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    device = get_model_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=n,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    replies = []
    for seq in output_ids:
        new_tokens = seq[prompt_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        reply = strip_think_tags(decoded) if STRIP_THINK else decoded.strip()
        replies.append(reply)

    return replies


# Scoring
def score_classifier(reply, target_emotion, classifier):
    """Return P(target_emotion) for a single reply."""
    if target_emotion not in classifier.label2id:
        raise ValueError(
            f"Target '{target_emotion}' "
            f"not found in classifier label space.\n"
            f"Available labels: {sorted(classifier.label2id.keys())}"
        )
    probs = classifier.predict_proba([reply])
    label_index = classifier.label2id[target_emotion]
    return float(probs[0][label_index])


def score_classifier_batch(replies, target_emotion, classifier):
    """Return P(target_emotion) for each reply in a batch."""
    if target_emotion not in classifier.label2id:
        raise ValueError(
            f"Target '{target_emotion}' "
            f"not found in classifier label space.\n"
            f"Available labels: {sorted(classifier.label2id.keys())}"
        )
    probs = classifier.predict_proba(replies)
    label_index = classifier.label2id[target_emotion]
    return [float(p[label_index]) for p in probs]


# Trajectory-level score
def trajectory_score_from_replies_targets(replies, mapped_targets, classifier):
    """
    Compute trajectory-level score once from explicit path replies/targets.
    """
    if not replies:
        return 0.0
    if len(replies) == 1:
        if not mapped_targets:
            return 0.0
        probs = classifier.predict_proba(replies)
        label_idx = classifier.label2id.get(mapped_targets[0])
        return float(probs[0][label_idx]) if label_idx is not None else 0.0

    labels = get_trajectory_labels(replies, classifier)
    probs = classifier.predict_proba(replies)
    per_step = compute_per_step_distances(probs)
    return compute_trajectory_level_score(labels, mapped_targets, per_step)


# Tree node
class ToTNode:
    """A node in the Tree-of-Thought search tree.

    Stores the message history up to this point, the reply that produced this
    node, single-turn and trajectory-level scores, and the accumulated path of
    replies and target labels from the root.
    """

    def __init__(
        self,
        messages,
        reply=None,
        score=0.0,
        target_emotion=None,
        mapped_target_emotion=None,
        depth=0,
        parent=None,
        path_replies=None,
        path_targets=None,
        trajectory_score=0.0,
    ):
        self.messages = messages
        self.reply = reply
        self.score = score
        self.target_emotion = target_emotion
        self.mapped_target_emotion = mapped_target_emotion
        self.depth = depth
        self.parent = parent
        self.children = []
        self.path_replies = path_replies or []
        self.path_targets = path_targets or []
        self.trajectory_score = trajectory_score


# ToT: build tree, beam-prune by cached trajectory-level score
def build_tot_tree(
    messages,
    target_trajectory,
    future_user_turns,
    tokenizer,
    model,
    classifier,
    k=K_BRANCHES,
    beam=BEAM_WIDTH,
    depth=DEPTH,
    scorer_mode=SCORER_MODE,
    debug=False,
):
    """Build a ToT search tree and return the leaf node with the highest trajectory score.

    At each depth level, expands every beam-surviving node into k branches,
    scores each branch by P(target_emotion), computes trajectory-level scores
    over the full path, and prunes to the top `beam` nodes.

    Args:
        messages: Current message history (list of role/content dicts).
        target_trajectory: Target emotion labels for upcoming assistant turns.
        future_user_turns: Real user utterances to inject between assistant turns.
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace causal LM.
        classifier: EmotionClassifier for scoring.
        k: Number of branches to expand at each node.
        beam: Number of top nodes to keep after each level.
        depth: Number of assistant turns to plan ahead.
        scorer_mode: One of "classifier", "llm_judge", or "both".
        debug: If True, returns (best_leaf, tree_debug_list); otherwise just best_leaf.

    Returns:
        The best ToTNode leaf (highest trajectory score), optionally with debug info.
    """
    root = ToTNode(messages, depth=0)
    beam_nodes = [root]

    tree_debug = [] if debug else None

    for d in range(depth):
        target_emotion = (
            target_trajectory[d]
            if d < len(target_trajectory)
            else target_trajectory[-1]
        )
        mapped_target = target_emotion
        next_level = []

        for beam_pos, node in enumerate(beam_nodes):
            base_messages = list(node.messages)

            # Preserve user/assistant alternation:
            # if node ends with assistant, inject next user turn before generating assistant
            last_role = base_messages[-1]["role"] if base_messages else None
            injected_user_turn = None
            if last_role == "assistant":
                user_idx_for_this_depth = d
                if user_idx_for_this_depth >= len(future_user_turns):
                    raise ValueError(
                        "ToT tried to expand beyond available real future user turns. "
                        "Check effective_depth logic."
                    )
                user_utt = future_user_turns[user_idx_for_this_depth]
                injected_user_turn = user_utt
                base_messages = base_messages + [{"role": "user", "content": user_utt}]

            roles_before_generation = [m["role"] for m in base_messages]
            messages_before_generation = [
                {"role": m["role"], "content": m["content"]} for m in base_messages
            ]

            replies = sample_replies(
                base_messages,
                tokenizer,
                model,
                n=k,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            turn_scores = score_classifier_batch(
                replies,
                target_emotion,
                classifier,
            )

            for branch_idx, (reply, turn_score) in enumerate(zip(replies, turn_scores)):
                new_msgs = base_messages + [{"role": "assistant", "content": reply}]

                child_path_replies = node.path_replies + [reply]
                child_path_targets = node.path_targets + [mapped_target]

                injected_future_user_turn = None
                # Inject real next user turn for deeper lookahead
                if d < len(future_user_turns):
                    injected_future_user_turn = future_user_turns[d]
                    new_msgs = new_msgs + [
                        {"role": "user", "content": future_user_turns[d]}
                    ]

                child_traj_score = trajectory_score_from_replies_targets(
                    child_path_replies,
                    child_path_targets,
                    classifier,
                )

                child = ToTNode(
                    messages=new_msgs,
                    reply=reply,
                    score=turn_score,
                    target_emotion=target_emotion,
                    mapped_target_emotion=mapped_target,
                    depth=d + 1,
                    parent=node,
                    path_replies=child_path_replies,
                    path_targets=child_path_targets,
                    trajectory_score=child_traj_score,
                )

                node.children.append(child)
                next_level.append(child)

                if debug:
                    tree_debug.append(
                        {
                            "depth": d,
                            "beam_parent_index": beam_pos,
                            "branch_index": branch_idx,
                            "target_emotion": target_emotion,
                            "mapped_target_emotion": mapped_target,
                            "roles_before_generation": roles_before_generation,
                            "messages_before_generation": messages_before_generation,
                            "injected_user_turn_before_generation": injected_user_turn,
                            "reply": reply,
                            "turn_score": float(turn_score),
                            "trajectory_score": float(child_traj_score),
                            "path": child_path_replies,
                            "path_targets": child_path_targets,
                            "injected_future_user_turn_after_reply": injected_future_user_turn,
                        }
                    )

        next_level.sort(key=lambda n: n.trajectory_score, reverse=True)
        beam_nodes = next_level[:beam]

        if debug:
            survivors = []
            for survivor_rank, survivor in enumerate(beam_nodes):
                survivors.append(
                    {
                        "survivor_rank": survivor_rank,
                        "depth": survivor.depth,
                        "reply": survivor.reply,
                        "turn_score": float(survivor.score),
                        "trajectory_score": float(survivor.trajectory_score),
                        "path": survivor.path_replies,
                        "path_targets": survivor.path_targets,
                    }
                )
            tree_debug.append(
                {
                    "depth": d,
                    "beam_survivors": survivors,
                }
            )

        if not beam_nodes:
            break

    if not beam_nodes:
        raise RuntimeError("ToT beam became empty unexpectedly.")

    best = max(beam_nodes, key=lambda n: n.trajectory_score)

    if debug:
        return best, tree_debug
    return best


# Full conversation generation
def generate_tot_conversation(
    conversation,
    tokenizer,
    model,
    classifier,
    target_trajectory,
    scorer_mode=SCORER_MODE,
    k_branches=K_BRANCHES,
    beam_width=BEAM_WIDTH,
    depth=DEPTH,
    debug=False,
):
    """Generate assistant turns for a full conversation using Tree-of-Thought planning.

    For each assistant turn, runs `build_tot_tree` to search ahead `depth` turns,
    then commits the first reply of the best-scoring path.

    Args:
        conversation: List of turn dicts with an 'utterance' key.
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace causal LM.
        classifier: EmotionClassifier for trajectory scoring.
        target_trajectory: List of target emotion labels, one per assistant turn.
        scorer_mode: One of "classifier", "llm_judge", or "both".
        k_branches: Branching factor at each tree node.
        beam_width: Number of nodes kept after each depth level.
        depth: Look-ahead depth in assistant turns.
        debug: If True, captures per-turn tree debug info.

    Returns:
        generated_turns: List of selected assistant reply strings.
        tot_log: Per-turn summary dicts (target, score, depth reached).
        tree_debug_log: Per-turn tree debug dicts (populated only when debug=True).
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
    tot_log = []
    tree_debug_log = []
    assistant_idx = 0

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
            remaining_user_turns = user_turns[user_turn_idx:]
            effective_depth = min(depth, 1 + len(remaining_user_turns))

            future_targets = target_trajectory[
                assistant_idx : assistant_idx + effective_depth
            ]
            future_user_turns = remaining_user_turns[: max(effective_depth - 1, 0)]

            if debug:
                best_leaf, tree_debug = build_tot_tree(
                    messages=messages,
                    target_trajectory=future_targets,
                    future_user_turns=future_user_turns,
                    tokenizer=tokenizer,
                    model=model,
                    classifier=classifier,
                    k=k_branches,
                    beam=beam_width,
                    depth=effective_depth,
                    scorer_mode=scorer_mode,
                    debug=True,
                )
            else:
                best_leaf = build_tot_tree(
                    messages=messages,
                    target_trajectory=future_targets,
                    future_user_turns=future_user_turns,
                    tokenizer=tokenizer,
                    model=model,
                    classifier=classifier,
                    k=k_branches,
                    beam=beam_width,
                    depth=effective_depth,
                    scorer_mode=scorer_mode,
                )
                tree_debug = None

            best_reply = best_leaf.path_replies[0]
            traj_score = best_leaf.trajectory_score

            tot_log.append(
                {
                    "turn": assistant_idx,
                    "target_emotion": future_targets[0]
                    if future_targets
                    else "neutral",
                    "reply": best_reply,
                    "trajectory_score": float(traj_score),
                    "depth_reached": best_leaf.depth,
                    "scorer_mode": scorer_mode,
                }
            )

            if debug:
                tree_debug_log.append(
                    {
                        "assistant_turn": assistant_idx,
                        "target_trajectory_window": future_targets,
                        "future_user_turns_window": future_user_turns,
                        "selected_reply": best_reply,
                        "selected_path": best_leaf.path_replies,
                        "selected_path_targets": best_leaf.path_targets,
                        "selected_trajectory_score": float(traj_score),
                        "tree_debug": tree_debug,
                    }
                )

            generated_turns.append(best_reply)
            messages.append({"role": "assistant", "content": best_reply})
            assistant_idx += 1

    return generated_turns, tot_log, tree_debug_log


# Main
def main():
    """Load data and model, run ToT planning over all conversations, and save results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-conversations", type=int, default=100)
    args = parser.parse_args()

    classifier = EmotionClassifier()
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

    all_tree_debug = []
    results = []

    for i, conversation in enumerate(conversations):
        print(f"\nRunning {i + 1}/{len(conversations)}")

        situation = conversation[0]["prompt"]
        gold_emotion_label = conversation[0]["context"]
        n_assistant_turns = sum(1 for idx in range(len(conversation)) if idx % 2 != 0)
        assistant_target = get_assistant_target(gold_emotion_label)
        target_trajectory = [assistant_target] * n_assistant_turns

        generated_turns, tot_log, tree_debug_log = generate_tot_conversation(
            conversation=conversation,
            tokenizer=tokenizer,
            model=model,
            classifier=classifier,
            target_trajectory=target_trajectory,
            debug=(i < 5),
        )

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
        mean_traj_score = (
            float(np.mean([t["trajectory_score"] for t in tot_log])) if tot_log else 0.0
        )

        results.append(
            {
                "situation": situation,
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
                "mean_traj_score": float(mean_traj_score),
                "tot_log": tot_log,
                "metadata": {
                    "model": model_name,
                    "k_branches": K_BRANCHES,
                    "beam_width": BEAM_WIDTH,
                    "depth": DEPTH,
                    "planner": "tot",
                    "scorer": SCORER_MODE,
                    "temperature": 0.8,
                    "seed": 42,
                },
            }
        )

        if i < 5:
            all_tree_debug.append(
                {
                    "conversation_index": i,
                    "situation": situation,
                    "user_emotion": gold_emotion_label,
                    "assistant_target": assistant_target,
                    "conversation": [
                        {
                            "utterance_idx": turn.get("utterance_idx"),
                            "speaker_idx": turn.get("speaker_idx"),
                            "utterance": turn["utterance"],
                        }
                        for turn in conversation
                    ],
                    "tree_debug_log": tree_debug_log,
                }
            )

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_filename = f"results/tot_results_{SCORER_MODE}_{timestamp}.json"
    debug_filename = f"results/tot_tree_debug_{timestamp}.json"

    with open(results_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(debug_filename, "w", encoding="utf-8") as f:
        json.dump(all_tree_debug, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {results_filename}")
    print(f"Saved tree debug to: {debug_filename}")


if __name__ == "__main__":
    main()
