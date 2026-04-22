import torch
import numpy as np


def extract_trajectory(dialogue, classifier):
    """Return a list of per-turn dicts with 'text', 'emotion', and 'confidence' keys.

    Args:
        dialogue: Iterable of utterance strings representing a conversation.
        classifier: Emotion classifier with a `predict_top_emotion(text)` method
            that returns (emotion_label, confidence_score).

    Returns:
        List of dicts, one per turn, each containing the original text, the
        predicted top emotion label, and the classifier's confidence score.
    """
    trajectory = []

    with torch.no_grad():
        for turn in dialogue:
            emotion, score = classifier.predict_top_emotion(turn)
            trajectory.append({"text": turn, "emotion": emotion, "confidence": score})

    return trajectory


def compute_drift(dialogue, classifier):
    """Compute the mean cosine drift across consecutive turns in a dialogue.

    Drift measures how much the emotion probability distribution shifts between
    adjacent turns. A score of 0 means no change; 1 means complete reversal.

    Args:
        dialogue: Iterable of utterance strings representing a conversation.
        classifier: Emotion classifier with a `predict_proba(text)` method that
            returns an array of shape (1, num_emotions).

    Returns:
        Mean cosine drift as a float in [0, 1]. Returns 0.0 if the dialogue has
        fewer than two turns.
    """
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
