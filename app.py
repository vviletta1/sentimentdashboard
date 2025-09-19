import streamlit as st
import pandas as pd
from transformers import pipeline
import collections

# --- Load Hugging Face pipelines (force PyTorch!)
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"
)
emotion_pipe = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    framework="pt",
    return_all_scores=True
)

st.title("Sentiment & Emotion Dashboard")

# --- Sidebar input
st.sidebar.header("Upload or Paste Text Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (column: 'text')", type=["csv"])
input_text = st.sidebar.text_area("Or paste multiple lines of text:", height=150)

# --- Get texts
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    texts = df['text'].astype(str).tolist()
elif input_text.strip():
    texts = [line.strip() for line in input_text.split('\n') if line.strip()]
else:
    st.info("Upload a CSV file or paste some text to begin.")
    st.stop()

# --- Sentiment & Emotion Analysis
with st.spinner("Analyzing..."):
    sentiments = sentiment_pipe(texts)
    emotions_list = [emotion_pipe(t)[0] for t in texts]

# --- Sentiment S
