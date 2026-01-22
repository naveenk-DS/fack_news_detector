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

# ---------------- PREDICT ----------------
if st.button("🔍 Check News"):
    if not text.strip():
        st.warning("⚠️ Please enter some news text.")
    elif not classifier:
        st.error("❌ Model failed to load.")
    else:
        # ADD TITLE AND SPECIAL FORMAT (REQUIRED by this model)
        title = st.text_input("📰 News Title (optional):", placeholder="Enter title here...")
        
        # Format input as model expects: <title>TITLE<content>TEXT<end>
        if title.strip():
            formatted_text = f"<title>{title.strip()}<content>{text.strip()}<end>"
        else:
            # Use first sentence as title if none provided
            sentences = text.strip().split('.')
            auto_title = sentences[0].strip() + '.' if sentences else "News Article"
            formatted_text = f"<title>{auto_title}<content>{text[len(auto_title)+1:].strip()}<end>"
        
        st.info("📝 Using formatted input: title + content")  # Debug info
        
        result = classifier(formatted_text)[0]
        label = result["label"]  # Will be "Fake" or "Real" (lowercase first letter)
        confidence = result["score"] * 100

        st.markdown("---")
        st.subheader("🧠 Prediction Result")

        # Model outputs: "Fake" or "Real" exactly
        if label == "Real":
            st.success("✅ **REAL News**")
        else:
            st.error("❌ **FAKE News**")

        st.info(f"📊 **Confidence**: {confidence:.1f}%")
        st.caption(f"Model label: `{label}` | Formatted input length: {len(formatted_text)} chars")
