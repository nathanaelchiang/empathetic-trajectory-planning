import json
import random
import os
import re
import numpy as np
import torch
from datetime import datetime

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.emotion_classifier import EmotionClassifier
from evaluation.trajectory import extract_trajectory, compute_drift
from emotion.assistant_targets import get_assistant_target

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# model_name = "Qwen/Qwen2.5-3B-Instruct"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name = "Qwen/Qwen2.5-7B-Instruct"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
STRIP_THINK = "R1" in model_name

MAX_NEW_TOKENS = 512


def strip_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Data Loading
def load_conversations(n=100):
    dataset = load_dataset(
        "empathetic_dialogues", split="train", trust_remote_code=True
    )

    conversations = {}
    for example in dataset:
        conv_id = example["conv_id"]
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(example)

    full_conversations = []
    for conv in conversations.values():
        conv = sorted(conv, key=lambda x: x["utterance_idx"])
        if len(conv) >= 4 and len(conv) % 2 == 0:
            full_conversations.append(conv)

    random.shuffle(full_conversations)
    return full_conversations[:n]


# Generation
def generate_conversation(conversation, tokenizer, model):
    messages = [
        {
            "role": "system",
            "content": "You are a supportive and empathetic conversational partner.",
        }
    ]
    generated_assistant_turns = []

    for idx, turn in enumerate(conversation):
        utterance = turn["utterance"].strip()
        if idx % 2 == 0:
            messages.append({"role": "user", "content": utterance})
        else:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

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
            reply = strip_think_tags(decoded) if STRIP_THINK else decoded.strip()
            generated_assistant_turns.append(reply)
            messages.append({"role": "assistant", "content": reply})

    return generated_assistant_turns


# Shared metric helpers
def get_trajectory_labels(generated_turns: list[str], classifier) -> list[str]:
    raw = extract_trajectory(generated_turns, classifier)
    if raw and isinstance(raw[0], dict):
        return [t.get("emotion", t.get("label", "unknown")) for t in raw]
    return [str(t) for t in raw]


def compute_per_step_distances(probs: list[np.ndarray]) -> list[float]:
    return [
        float(np.sum(np.abs(probs[i + 1] - probs[i]))) for i in range(len(probs) - 1)
    ]


def compute_emotion_entropy(probs: list[np.ndarray]) -> list[float]:
    entropies = []
    for p in probs:
        p = np.clip(p, 1e-9, 1.0)
        entropies.append(float(-np.sum(p * np.log(p))))
    return entropies


def compute_reversal_rate(labels: list[str]) -> float:
    if len(labels) < 2:
        return 0.0

    reversals = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            reversals += 1

    return reversals / (len(labels) - 1)


def compute_peak_drift_turn(distances: list[float]) -> int:
    return int(np.argmax(distances)) if distances else -1


def compute_trajectory_alignment(
    pred_labels: list[str], target_labels: list[str]
) -> float:
    if not pred_labels:
        return 0.0
    return float(
        np.mean([1.0 if p == t else 0.0 for p, t in zip(pred_labels, target_labels)])
    )


def compute_trajectory_level_score(
    labels: list[str],
    target_labels: list[str],
    per_step_distances: list[float],
) -> float:

    alignment = compute_trajectory_alignment(labels, target_labels)
    reversal = compute_reversal_rate(labels)
    mean_drift = float(np.mean(per_step_distances)) if per_step_distances else 0.0

    norm_drift = min(mean_drift / 2.0, 1.0)

    return alignment * (1.0 - 0.5 * reversal) * (1.0 - 0.5 * norm_drift)


def main():
    classifier = EmotionClassifier()

    print("Classifier label space:")
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

        generated_turns = generate_conversation(conversation, tokenizer, model)

        labels = get_trajectory_labels(generated_turns, classifier)
        drift = compute_drift(generated_turns, classifier)
        probs = classifier.predict_proba(generated_turns)
        per_step_dist = compute_per_step_distances(probs)
        entropies = compute_emotion_entropy(probs)
        reversal_rate = compute_reversal_rate(labels)
        peak_drift_turn = compute_peak_drift_turn(per_step_dist)

        n_turns = len(generated_turns)

        # Map targets into classifier label space
        # target_trajectory_raw = [gold_emotion_label] * n_turns
        # target_trajectory_mapped = [map_target_emotion(t) for t in target_trajectory_raw]

        # Determine assistant target emotion based on user emotion
        assistant_target_label = get_assistant_target(gold_emotion_label)

        target_trajectory = [assistant_target_label] * n_turns

        alignment_score = compute_trajectory_alignment(labels, target_trajectory)
        traj_level_score = compute_trajectory_level_score(
            labels, target_trajectory, per_step_dist
        )

        # 🔎 DEBUG: print first 5 conversations
        if i < 5:
            print("\n------------------------------")
            print(f"Situation: {situation}")
            print(f"User emotion: {gold_emotion_label}")
            print(f"Assistant target: {assistant_target_label}")
            print(f"Predicted trajectory: {labels}")
            print(f"Alignment score: {alignment_score:.2f}")
            print("------------------------------\n")

        results.append(
            {
                "situation": situation,
                "user_emotion": gold_emotion_label,
                "assistant_target": assistant_target_label,
                "generated_dialogue": generated_turns,
                "trajectory": labels,
                "drift": drift,
                "per_step_distances": per_step_dist,
                "per_turn_entropy": entropies,
                "reversal_rate": reversal_rate,
                "peak_drift_turn": peak_drift_turn,
                "alignment_score": alignment_score,
                "traj_level_score": traj_level_score,
            }
        )

        print(
            f"  Drift: {drift:.4f} | Alignment: {alignment_score:.2f} | "
            f"Reversal: {reversal_rate:.2f} | Traj score: {traj_level_score:.4f}"
        )

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/baseline_results_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

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
    ]:
        arr = np.array(arr)
        print(
            f"{label:22s}  mean={arr.mean():.4f}  std={arr.std():.4f}"
            f"  min={arr.min():.4f}  max={arr.max():.4f}"
        )


if __name__ == "__main__":
    main()
