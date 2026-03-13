import torch
import torch.nn.functional as F


def extract_trajectory(dialogue, classifier):
    trajectory = []
    
    for turn in dialogue:
        emotion, score = classifier.predict_top_emotion(turn)
        trajectory.append({
            "text": turn,
            "emotion": emotion,
            "confidence": score
        })
    
    return trajectory


def compute_drift(dialogue, classifier):
    vectors = []
    
    for turn in dialogue:
        vec = classifier.predict_proba(turn)
        vectors.append(vec)
    
    drift_scores = []
    
    for i in range(len(vectors) - 1):
        sim = F.cosine_similarity(
            vectors[i].unsqueeze(0),
            vectors[i + 1].unsqueeze(0)
        )
        drift_scores.append(1 - sim.item())
    
    return sum(drift_scores) / len(drift_scores)