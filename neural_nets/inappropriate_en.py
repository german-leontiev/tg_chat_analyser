import torch
from transformers import pipeline


model_folder_path = "michellejieli/inappropriate_text_classifier"


classifier = pipeline("sentiment-analysis", model="michellejieli/inappropriate_text_classifier")
 

# Probabilistic prediction of emotion in a text
@torch.no_grad()
def predict_appropriateness(text: str) -> list:
    pred = classifier(text)[0]
    result = pred['score'] if pred["label"] != "NSFW" else 1 - pred["score"]
    return result
