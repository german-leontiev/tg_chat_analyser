import torch
from transformers import BertForSequenceClassification, AutoTokenizer

LABELS = ["surprise", "disgust", "neutral", "anger", "sadness", "fear", "joy"]
model_folder_path = "j-hartmann/emotion-english-distilroberta-base"


tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
model = BertForSequenceClassification.from_pretrained(model_folder_path)
if torch.cuda.is_available():
    model.cuda()


# Probabilistic prediction of emotion in a text
@torch.no_grad()
def predict_emotions(text: str) -> list:
    inputs = tokenizer(text, truncation=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    outputs = model(**inputs)

    pred = torch.nn.functional.softmax(outputs.logits, dim=1)

    emotions_list = {}
    for i in range(len(pred[0].tolist())):
        emotions_list[LABELS[i]] = round(pred[0].tolist()[i], 4)
    return emotions_list
