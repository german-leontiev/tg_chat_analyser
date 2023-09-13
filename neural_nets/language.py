import numpy as np
import torch
from transformers import BertForSequenceClassification, AutoTokenizer

model_folder_path = "papluca/xlm-roberta-base-language-detection"

amazon_languages = ['en', 'de', 'fr', 'es', 'ja', 'zh']
xnli_languages = ['ar', 'el', 'hi', 'ru', 'th', 'tr', 'vi', 'bg', 'sw', 'ur']
stsb_languages = ['it', 'nl', 'pl', 'pt']

LABELS = sorted(list(set(amazon_languages + xnli_languages + stsb_languages)))

tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
model = BertForSequenceClassification.from_pretrained(model_folder_path)
if torch.cuda.is_available():
    model.cuda()




# Probabilistic prediction of emotion in a text
@torch.no_grad()
def detect_lang(text: str) -> str:
    inputs = tokenizer(text, truncation=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    outputs = model(**inputs)

    pred = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    return LABELS[np.argmax(pred[0].tolist())]
