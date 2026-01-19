import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizerFast.from_pretrained("./fake_news_model")
model = DistilBertForSequenceClassification.from_pretrained("./fake_news_model")
model.eval()

df = pd.read_csv("Data/Fake.csv").head(50)
texts = df["text"].tolist()

inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)

preds = torch.argmax(outputs.logits, dim=1)
print("Predictions:", preds.tolist())
