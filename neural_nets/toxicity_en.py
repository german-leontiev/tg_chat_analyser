import torch
from transformers import (
    TextClassificationPipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


model_folder_path = "cointegrated/rubert-tiny-toxicity"

tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
model = AutoModelForSequenceClassification.from_pretrained(model_folder_path)


def predict_toxicity(text):
    pipeline =  TextClassificationPipeline(model=model, tokenizer=tokenizer)
    result = pipeline(text)[0]
    if result["label"] == "non-toxic":
        return 1 - result["score"]
    return result["score"]
