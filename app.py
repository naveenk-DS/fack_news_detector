from fastapi import FastAPI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# VERY SMALL MODEL (fits Render free memory)
MODEL_NAME = "prajjwal1/bert-tiny"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.eval()

@app.get("/")
def home():
    return {"message": "Fake News Detection API is running"}

@app.post("/predict")
def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()

    return {"prediction": "REAL" if pred == 1 else "FAKE"}

