import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Fake News Detection System",
    layout="centered",
    page_icon="📰"
)

# ------------------ TITLE ------------------
st.title("📰 Fake News Detection System")
st.write("Paste a news article below and click **Check News** to detect whether it is Fake or Real.")

# ------------------ MODEL SETUP ------------------
MODEL_NAME = "mrm8488/bert-tiny-finetuned-fake-news-detection"

@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

with st.spinner("Loading AI model, please wait..."):
    tokenizer, model = load_model()

# ------------------ INPUT AREA ------------------
text = st.text_area(
    "📝 Enter News Text Here",
    height=220,
    placeholder="Paste any news article here..."
)

# ------------------ PREDICTION BUTTON ------------------
if st.button("🔍 Check News"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some news text before checking.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        # Prediction
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()

        # Correct label mapping from model
        label = model.config.id2label[pred]

        confidence = float(probs[pred]) * 100

        st.markdown("---")
        st.subheader("🧠 Prediction Result")

        # Display result
        if label == "LABEL_1":
            st.success("✅ This looks like REAL News")
        else:
            st.error("❌ This looks like FAKE News")


        # Confidence score
        st.info(f"📊 Confidence: **{confidence:.2f}%**")

# ------------------ SIDEBAR INFO ------------------
st.sidebar.title("ℹ️ About Project")
st.sidebar.write("""
This is a Fake News Detection system built using:

- 🧠 Transformer Model (BERT Tiny)  
- 🤗 HuggingFace  
- 🌐 Streamlit Web App  
- ☁️ Deployed on Render Cloud  

The system analyzes news text and predicts whether it is **Fake** or **Real**.
""")

st.sidebar.markdown("---")
st.sidebar.write("👨‍🎓 Project by Naveen")



