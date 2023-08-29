import torch
from transformers import BertForSequenceClassification, AutoTokenizer

LABELS = {
    0: "Appropriate",
    1: "Inappropriate",
}
model_folder_path = "apanc/russian-inappropriate-messages"


tokenizer = AutoTokenizer.from_pretrained(model_folder_path, model_max_length=512)
model = BertForSequenceClassification.from_pretrained(model_folder_path)
if torch.cuda.is_available():
    model.cuda()


# Probabilistic prediction of emotion in a text
@torch.no_grad()
def predict_appropriateness(text: str) -> list:
    inputs = tokenizer(text, truncation=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    outputs = model(**inputs)

    pred = torch.nn.functional.softmax(outputs.logits, dim=1)

    appropriateness_list = {}
    for i in range(len(pred[0].tolist())):
        appropriateness_list[LABELS[i]] = round(pred[0].tolist()[i], 4)
    return appropriateness_list
