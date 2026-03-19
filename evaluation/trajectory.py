import torch
import numpy as np


def extract_trajectory(dialogue, classifier):
    trajectory = []

    with torch.no_grad():
        for turn in dialogue:
            emotion, score = classifier.predict_top_emotion(turn)
            trajectory.append({"text": turn, "emotion": emotion, "confidence": score})

    return trajectory


def compute_drift(dialogue, classifier):
    vectors = []

    for turn in dialogue:
        vec = classifier.predict_proba(turn)[0]  # shape (num_emotions,)
        vectors.append(vec)

    if len(vectors) < 2:
        return 0.0

    drift_scores = []

    for i in range(len(vectors) - 1):
        v1 = vectors[i]
        v2 = vectors[i + 1]

        # Cosine similarity using NumPy
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

        drift_scores.append(1 - sim)

    return float(np.mean(drift_scores))
