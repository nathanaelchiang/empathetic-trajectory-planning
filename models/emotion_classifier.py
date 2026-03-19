import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EmotionClassifier:
    def __init__(self, model_name="SamLowe/roberta-base-go_emotions", threshold=0.3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

    def predict_proba(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.sigmoid(outputs.logits)

        return probs.detach().cpu().numpy()

    def predict_top_emotion(self, text):
        probs = self.predict_proba(text)[0]  # shape (num_emotions,)
        top_idx = probs.argmax()
        return self.id2label[int(top_idx)], float(probs[top_idx])

    def predict_multi_label(self, text):
        probs = self.predict_proba(text)
        
        results = []
        for i, prob in enumerate(probs):
            if prob > self.threshold:
                results.append((self.id2label[i], float(prob)))
        
        return sorted(results, key=lambda x: x[1], reverse=True)