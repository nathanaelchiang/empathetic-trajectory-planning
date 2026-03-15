from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from experiments.run_baseline_generation import (
    load_situations,
    generate_dialogue,
    model_name
)
from models.emotion_classifier import EmotionClassifier
from evaluation.trajectory import extract_trajectory, compute_drift


def main():

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("CUDA available:", torch.cuda.is_available())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    print("Model device:", next(model.parameters()).device)

    classifier = EmotionClassifier()

    situation = load_situations(1)[0]

    print("SITUATION:")
    print(situation)

    dialogue = generate_dialogue(situation, tokenizer, model)

    print("\nDIALOGUE:")
    for turn in dialogue:
        print("-", turn)

    trajectory = extract_trajectory(dialogue, classifier)

    print("\nTRAJECTORY:")
    for step in trajectory:
        print(step)

    print("\nDRIFT:", compute_drift(dialogue, classifier))


if __name__ == "__main__":
    main()