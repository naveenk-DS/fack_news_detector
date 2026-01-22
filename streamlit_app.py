import streamlit as st
from transformers import pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detection System",
    layout="centered",
    page_icon="📰"
)

# ---------------- TITLE ----------------
st.title("📰 Fake News Detection System")
st.write("Paste a news article below and click **Check News** to detect whether it is Fake or Real.")

# ---------------- LOAD MODEL ----------------
MODEL_NAME = "mrm8488/bert-tiny-finetuned-fake-news-detection"

@st.cache_resource
def load_model():
    classifier = pipeline("text-classification", model=MODEL_NAME)
    return classifier

with st.spinner("🔄 Loading AI model... Please wait"):
    classifier = load_model()

# ---------------- INPUT ----------------
text = st.text_area(
    "📝 Enter News Text Here",
    height=220,
    placeholder="Paste any news article here..."
)

# ---------------- PREDICT ----------------
if st.button("🔍 Check News"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some news text before checking.")
    else:
        result = classifier(text)[0]

        label = result["label"]
        confidence = result["score"] * 100

        st.markdown("---")
        st.subheader("🧠 Prediction Result")

        # 🔥 FINAL CORRECT MAPPING
        if "REAL" in label.upper():
            st.success("✅ This looks like REAL News")
        else:
            st.error("❌ This looks like FAKE News")

        st.info(f"📊 Confidence: **{confidence:.2f}%**")

# ---------------- SIDEBAR ----------------
st.sidebar.title("ℹ️ About Project")
st.sidebar.write("""
This is a Fake News Detection system built using:
- 🧠 BERT Tiny (Fine-tuned for Fake News)  
- 🤗 HuggingFace Transformers  
- 🌐 Streamlit Web App  
- ☁️ Hosted on Hugging Face Spaces  
The system predicts whether a news article is **Fake** or **Real**.
""")

st.sidebar.markdown("---")
st.sidebar.write("👨‍🎓 Project by Naveen")
