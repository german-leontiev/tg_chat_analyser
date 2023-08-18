import torch
from transformers import BertForSequenceClassification, AutoTokenizer

LABELS = {
    0: "NEUTRAL",
    1: "POSITIVE",
    2: "NEGATIVE",
}
model_folder_path = "MonoHime/rubert-base-cased-sentiment-new"


tokenizer = AutoTokenizer.from_pretrained(model_folder_path, model_max_length=512)
model = BertForSequenceClassification.from_pretrained(model_folder_path)
if torch.cuda.is_available():
    model.cuda()


# Probabilistic prediction of emotion in a text
@torch.no_grad()
def predict_sentiment(text: str) -> list:
    inputs = tokenizer(text, truncation=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    outputs = model(**inputs)

    pred = torch.nn.functional.softmax(outputs.logits, dim=1)

    sentiment_list = {}
    for i in range(len(pred[0].tolist())):
        sentiment_list[LABELS[i]] = round(pred[0].tolist()[i], 4)
    return sentiment_list
