from models.emotion_classifier import EmotionClassifier
from evaluation.trajectory import extract_trajectory, compute_drift


if __name__ == "__main__":
    classifier = EmotionClassifier()

    dialogue = [
        "I just lost my job today.",
        "That must feel overwhelming. I'm really sorry you're going through that.",
        "Yeah, I don't know what to do next."
    ]

    trajectory = extract_trajectory(dialogue, classifier)
    drift = compute_drift(dialogue, classifier)

    print("Trajectory:")
    for step in trajectory:
        print(step)

    print("\nDrift score:", drift)