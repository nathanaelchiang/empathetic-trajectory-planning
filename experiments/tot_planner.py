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

# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

JUDGE_MODEL = "claude-sonnet-4-20250514"

K_BRANCHES = 3
BEAM_WIDTH = 2
DEPTH = 2
MAX_NEW_TOKENS = 64

# choose default: "classifier", "llm_judge", or "both"
SCORER_MODE = "classifier"


def get_model_device(model):
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
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        replies.append(reply)

    return replies


# Scoring
def score_classifier(reply, target_emotion, classifier):
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
    if target_emotion not in classifier.label2id:
        raise ValueError(
            f"Target '{target_emotion}' "
            f"not found in classifier label space.\n"
            f"Available labels: {sorted(classifier.label2id.keys())}"
        )

    probs = classifier.predict_proba(replies)
    label_index = classifier.label2id[target_emotion]
    return [float(p[label_index]) for p in probs]


def score_llm_judge(reply, target_emotion, dialogue_history):
    import anthropic

    history_str = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in dialogue_history
        if m["role"] != "system"
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


def score_reply(
    reply, target_emotion, classifier, dialogue_history, scorer_mode=SCORER_MODE
):
    if scorer_mode == "classifier":
        return score_classifier(reply, target_emotion, classifier)

    if scorer_mode == "llm_judge":
        return score_llm_judge(reply, target_emotion, dialogue_history)

    if scorer_mode == "both":
        clf = score_classifier(reply, target_emotion, classifier)
        llm = score_llm_judge(reply, target_emotion, dialogue_history)
        return (clf + llm) / 2.0

    raise ValueError(f"Invalid scorer_mode: {scorer_mode}")


def score_replies_batch(
    replies, target_emotion, classifier, dialogue_history, scorer_mode=SCORER_MODE
):
    """
    Fast path for classifier mode; fallback to per-reply scoring for other modes.
    """
    if scorer_mode == "classifier":
        return score_classifier_batch(replies, target_emotion, classifier)

    if scorer_mode == "llm_judge":
        return [
            score_llm_judge(reply, target_emotion, dialogue_history)
            for reply in replies
        ]

    if scorer_mode == "both":
        clf_scores = score_classifier_batch(replies, target_emotion, classifier)
        llm_scores = [
            score_llm_judge(reply, target_emotion, dialogue_history)
            for reply in replies
        ]
        return [(c + l) / 2.0 for c, l in zip(clf_scores, llm_scores)]

    raise ValueError(f"Invalid scorer_mode: {scorer_mode}")


# Trajectory-level score
def trajectory_score_from_replies_targets(replies, mapped_targets, classifier):
    """
    Compute trajectory-level score once from explicit path replies/targets.
    """
    if not replies:
        return 0.0

    labels = get_trajectory_labels(replies, classifier)
    probs = classifier.predict_proba(replies)
    per_step = compute_per_step_distances(probs)
    return compute_trajectory_level_score(labels, mapped_targets, per_step)


# Tree node


class ToTNode:
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


# ---------------------------------------------------------------------------
# ToT: build tree, beam-prune by cached trajectory-level score
# ---------------------------------------------------------------------------


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
):
    root = ToTNode(messages, depth=0)
    beam_nodes = [root]

    for d in range(depth):
        target_emotion = (
            target_trajectory[d]
            if d < len(target_trajectory)
            else target_trajectory[-1]
        )
        mapped_target = target_emotion
        next_level = []

        for node in beam_nodes:
            base_messages = list(node.messages)

            # Preserve user/assistant alternation:
            # if node ends with assistant, inject next user turn before generating assistant
            last_role = base_messages[-1]["role"] if base_messages else None
            if last_role == "assistant":
                user_idx_for_this_depth = d
                if user_idx_for_this_depth < len(future_user_turns):
                    user_utt = future_user_turns[user_idx_for_this_depth]
                else:
                    user_utt = "Please continue."
                base_messages = base_messages + [{"role": "user", "content": user_utt}]

            replies = sample_replies(
                base_messages,
                tokenizer,
                model,
                n=k,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            turn_scores = score_replies_batch(
                replies,
                target_emotion,
                classifier,
                base_messages,
                scorer_mode=scorer_mode,
            )

            for reply, turn_score in zip(replies, turn_scores):
                new_msgs = base_messages + [{"role": "assistant", "content": reply}]

                child_path_replies = node.path_replies + [reply]
                child_path_targets = node.path_targets + [mapped_target]

                # Inject real next user turn for deeper lookahead
                if d < len(future_user_turns):
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

        next_level.sort(key=lambda n: n.trajectory_score, reverse=True)
        beam_nodes = next_level[:beam]

        if not beam_nodes:
            break

    if not beam_nodes:
        raise RuntimeError("ToT beam became empty unexpectedly.")

    return max(beam_nodes, key=lambda n: n.trajectory_score)


# ---------------------------------------------------------------------------
# Full conversation generation
# ---------------------------------------------------------------------------


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
    tot_log = []
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
            future_targets = target_trajectory[assistant_idx : assistant_idx + depth]
            future_user_turns = user_turns[user_turn_idx : user_turn_idx + depth]

            best_leaf = build_tot_tree(
                messages=messages,
                target_trajectory=future_targets,
                future_user_turns=future_user_turns,
                tokenizer=tokenizer,
                model=model,
                classifier=classifier,
                k=k_branches,
                beam=beam_width,
                depth=depth,
                scorer_mode=scorer_mode,
            )

            best_reply = best_leaf.path_replies[0]
            traj_score = best_leaf.trajectory_score

            tot_log.append(
                {
                    "turn": assistant_idx,
                    "target_emotion": future_targets[0]
                    if future_targets
                    else "neutral",
                    "reply": best_reply,
                    "trajectory_score": traj_score,
                    "depth_reached": best_leaf.depth,
                    "scorer_mode": scorer_mode,
                }
            )

            generated_turns.append(best_reply)
            messages.append({"role": "assistant", "content": best_reply})
            assistant_idx += 1

    return generated_turns, tot_log


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
    parser.add_argument("--k-branches", type=int, default=K_BRANCHES)
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    global MAX_NEW_TOKENS
    MAX_NEW_TOKENS = args.max_new_tokens

    scorer_mode = args.scorer
    k_branches = args.k_branches
    beam_width = args.beam_width
    depth = args.depth

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

        generated_turns, tot_log = generate_tot_conversation(
            conversation=conversation,
            tokenizer=tokenizer,
            model=model,
            classifier=classifier,
            target_trajectory=target_trajectory,
            scorer_mode=scorer_mode,
            k_branches=k_branches,
            beam_width=beam_width,
            depth=depth,
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
                "mean_traj_score": mean_traj_score,
                "tot_log": tot_log,
                "metadata": {
                    "model": model_name,
                    "k_branches": k_branches,
                    "beam_width": beam_width,
                    "depth": depth,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "planner": "tot",
                    "scorer": scorer_mode,
                    "temperature": 0.8,
                    "seed": 42,
                },
            }
        )

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
    filename = f"results/tot_results_{scorer_mode}_{timestamp}.json"

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
        ("Mean traj score", [x["mean_traj_score"] for x in results]),
    ]:
        arr = np.array(arr)
        print(
            f"{label:22s}  mean={arr.mean():.4f}  std={arr.std():.4f}"
            f"  min={arr.min():.4f}  max={arr.max():.4f}"
        )


if __name__ == "__main__":
    main()
