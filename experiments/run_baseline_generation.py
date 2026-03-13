import json
import random
from datasets import load_dataset

from models.emotion_classifier import EmotionClassifier
from evaluation.trajectory import extract_trajectory, compute_drift

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def load_situations(n=30):
    dataset = load_dataset("empathetic_dialogues", split="train")
    
    situations = list(set(example["situation"] for example in dataset))
    random.shuffle(situations)
    
    return situations[:n]

# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_dialogue(prompt, turns=6):
    messages = [
        {"role": "system", "content": "You are a supportive and empathetic conversational partner."},
        {"role": "user", "content": prompt}
    ]

    dialogue = []

    for _ in range(turns):
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        ).to(model.device)

        output_ids = model.generate(
            input_ids,
            max_new_tokens=200,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Extract only newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

        dialogue.append(reply.strip())

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": "Continue the conversation naturally."})
        torch.cuda.empty_cache()
        
    return dialogue
def main():
    classifier = EmotionClassifier()
    situations = load_situations(30)

    results = []

    for i, situation in enumerate(situations):
        print(f"Running {i+1}/30")

        dialogue = generate_dialogue(situation)

        trajectory = extract_trajectory(dialogue, classifier)
        drift = compute_drift(dialogue, classifier)

        results.append({
            "situation": situation,
            "dialogue": dialogue,
            "trajectory": trajectory,
            "drift": drift,
            "metadata": {
                "model": MODEL_NAME,
                "temperature": 0.8,
                "turns": 6,
                "seed": 42
            }
        })

    with open("results/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()