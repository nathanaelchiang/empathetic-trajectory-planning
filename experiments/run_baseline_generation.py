import json
import random
import os
import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.emotion_classifier import EmotionClassifier
from evaluation.trajectory import extract_trajectory, compute_drift

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

model_name = "mistralai/Mistral-7B-Instruct-v0.3"


def load_situations(n=30):
    dataset = load_dataset(
        "empathetic_dialogues",
        split="train",
        trust_remote_code=True
    )
    situations = list(set(example["prompt"] for example in dataset))
    random.shuffle(situations)
    return situations[:n]


def generate_dialogue(prompt, tokenizer, model, turns=6):
    messages = [
        {"role": "system", "content": "You are a supportive and empathetic conversational partner."},
        {"role": "user", "content": prompt}
    ]

    dialogue = []

    for _ in range(turns):

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

        dialogue.append(reply.strip())

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": "Continue the conversation naturally."})

    return dialogue


def main():

    classifier = EmotionClassifier()

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    situations = load_situations(30)

    results = []

    for i, situation in enumerate(situations):
        print(f"Running {i+1}/30")

        dialogue = generate_dialogue(situation, tokenizer, model)

        trajectory = extract_trajectory(dialogue, classifier)
        drift = compute_drift(dialogue, classifier)

        results.append({
            "situation": situation,
            "dialogue": dialogue,
            "trajectory": trajectory,
            "drift": drift,
            "metadata": {
                "model": model_name,
                "temperature": 0.8,
                "turns": 6,
                "seed": 42
            }
        })

    os.makedirs("results", exist_ok=True)

    with open("results/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()