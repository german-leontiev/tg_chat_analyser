import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


model_folder_path = "cointegrated/rubert-tiny-toxicity"

tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
model = AutoModelForSequenceClassification.from_pretrained(model_folder_path)
if torch.cuda.is_available():
    model.cuda()


def predict_toxicity(text, aggregate=True):
    """Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(
            model.device
        )
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba
