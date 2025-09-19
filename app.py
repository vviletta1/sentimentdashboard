import streamlit as st
import pandas as pd
from transformers import pipeline

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
