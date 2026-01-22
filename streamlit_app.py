import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(page_title="Fake News Detection System", layout="centered", page_icon="📰")

st.title("📰 Fake News Detection System")
st.write("Paste a news article below and click **Check News** to detect whether it is Fake or Real.")

@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", 
                   model="hamzab/roberta-fake-news-classification",
                   device=device)

with st.spinner("🔄 Loading AI model..."):
    classifier = load_model()

# INPUT
title_input = st.text_input("📰 News Title:", placeholder="Enter title (helps accuracy)")
text = st.text_area("📝 News Content:", height=200, placeholder="Paste article body...")

# PREDICT - NEW FIXED VERSION
if st.button("🔍 Check News", type="primary"):
    if not text.strip():
        st.warning("⚠️ Enter news content.")
    elif not classifier:
        st.error("❌ Model load failed.")
    else:
        # Format: <title>..</title><content>..</content>
        if title_input.strip():
            formatted = f"<title>{title_input}<content>{text}<end>"
        else:
            formatted = f"<title>News Article</title><content>{text}<end>"
        
        result = classifier(formatted)[0]
        label = result["label"]
        confidence = result["score"] * 100

        st.markdown("---")
        st.subheader("🧠 Result")
        
        if label == "Real":
            st.success("✅ **REAL News**")
        else:
            st.error("❌ **FAKE News**")
        
        st.info(f"**Confidence**: {confidence:.1f}%")
        st.caption(f"Label: `{label}`")

# SIDEBAR (keep your original)
st.sidebar.title("ℹ️ About")
st.sidebar.info("RoBERTa model + special formatting for accurate detection.")
st.sidebar.markdown("👨‍💻 By Naveen")
