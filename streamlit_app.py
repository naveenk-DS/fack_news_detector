import streamlit as st
from transformers import pipeline
import torch

# PAGE CONFIG (unchanged)
st.set_page_config(page_title="Fake News Detection System", layout="centered", page_icon="📰")

st.title("📰 Fake News Detection System")
st.write("Paste a news article below and click **Check News** to detect whether it is Fake or Real.")

# Use GPU if available on HF Spaces
device = 0 if torch.cuda.is_available() else -1

@st.cache_resource
def load_model():
    try:
        classifier = pipeline(
            "text-classification", 
            model="hamzab/roberta-fake-news-classification",
            device=device,  # Use GPU if available
            torch_dtype=torch.float16 if device == 0 else None  # Half precision for speed
        )
        return classifier
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

with st.spinner("🔄 Loading AI model..."):
    classifier = load_model()

# INPUT with truncation warning
text = st.text_area(
    "📝 Enter News Text Here",
    height=220,
    placeholder="Paste any news article here...",
    max_chars=2000  # Limit for better performance
)

if st.button("🔍 Check News", type="primary"):
    if not classifier:
        st.error("❌ Model failed to load. Please refresh.")
    elif not text.strip():
        st.warning("⚠️ Please enter some news text.")
    else:
        # Truncate long text
        if len(text) > 512:
            text = text[:500] + "..."
            st.warning("📝 Text truncated for analysis (max 512 tokens)")
        
        result = classifier(text)[0]
        label = result["label"]
        confidence = result["score"] * 100

        st.markdown("---")
        st.subheader("🧠 Prediction Result")

        # Improved label mapping (check exact model labels)
        if "REAL" in label.upper() or "LEGIT" in label.upper():
            st.success("✅ This looks like **REAL** News")
        else:
            st.error("❌ This looks like **FAKE** News")

        col1, col2 = st.columns([3,1])
        with col1:
            st.info(f"**Confidence**: {confidence:.1f}%")
        with col2:
            st.metric("Score", f"{confidence:.0f}%")
