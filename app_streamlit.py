import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Paste a news article below and click **Check News**")

MODEL_NAME = "hamzab/roberta-fake-news-classification"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

text = st.text_area("Enter News Text Here", height=200)

if st.button("Check News"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()
        label = model.config.id2label[pred]

        if "real" in label.lower():
            st.success("✅ This looks like REAL News")
        else:
            st.error("❌ This looks like FAKE News")



