from fastapi import FastAPI
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = FastAPI()

# Load trained model from local folder
MODEL_PATH = "./fake_news_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
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
