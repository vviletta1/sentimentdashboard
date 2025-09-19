import streamlit as st
import pandas as pd
from transformers import pipeline
import collections
import plotly.express as px

# --- Sidebar Logo & Title ---
st.sidebar.image(
    "https://i.imgur.com/9b4GdBR.png",  # Change this to your own logo if you want
    width=120,
    caption="VeeBot AI"
)
st.sidebar.title("💬 VeeBot AI Dashboard")
st.sidebar.markdown("Gain instant insights into customer mood & emotion with state-of-the-art AI.")

# --- Custom Styles ---
st.markdown("""
    <style>
    .main {background-color: #f9fafb;}
    h1, h2, h3 {color: #1d3557 !important; font-family: 'Segoe UI', sans-serif; font-weight: 700;}
    .stButton>button {color: #fff !important; background-color: #457b9d !important; border-radius: 8px !important; font-weight: 600;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {background-color: #e9ecef !important; border-radius: 8px 8px 0 0 !important; font-size: 16px; font-weight: 600; padding: 6px 20px !important; color: #1d3557 !important;}
    .kpi-card {background: #fff; border-radius: 12px; box-shadow: 0 2px 12px #e6e6e6; padding: 22px 10px; text-align: center; margin-bottom: 14px;}
    </style>
""", unsafe_allow_html=True)

st.title("🌈 Sentiment & Emotion Dashboard")
st.sidebar.header("📂 Upload or Paste Text Data")

# --- Load Hugging Face pipelines (force PyTorch)
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

# --- Sidebar Upload or Paste
uploaded_file = st.sidebar.file_uploader("Upload CSV (column: 'text')", type=["csv"])
input_text = st.sidebar.text_area("Or paste multiple lines of text:", height=120)

# --- Get texts
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    texts = df['text'].astype(str).tolist()
elif input_text.strip():
    texts = [line.strip() for line in input_text.split('\n') if line.strip()]
else:
    st.info("📝 Upload a CSV file or paste some text to begin.")
    st.stop()

# --- Sentiment & Emotion Analysis
with st.spinner("Analyzing..."):
    sentiments = sentiment_pipe(texts)
    emotions_list = [emotion_pipe(t)[0] for t in texts]

sentiment_df = pd.DataFrame(sentiments)

# --- Emotion Summary Aggregation
emotion_totals = collections.defaultdict(float)
emotion_counts = collections.defaultdict(int)
emotion_rows = []
for i, emotion_scores in enumerate(emotions_list):
    row = {"text": texts[i]}
    for entry in emotion_scores:
        emotion_totals[entry['label']] += entry['score']
        emotion_counts[entry['label']] += 1
        row[entry['label']] = entry['score']
    emotion_rows.append(row)
avg_emotion = {k: (emotion_totals[k] / emotion_counts[k]) for k in emotion_totals}
top_emotion = max(avg_emotion, key=avg_emotion.get).capitalize() if avg_emotion else ""

# --- KPI CARDS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""<div class='kpi-card'><span style='font-size:32px;'>{len(texts)}</span><br/><span style='color:#1d3557;'>Total Messages</span></div>""", unsafe_allow_html=True)
with col2:
    pct_positive = (sentiment_df['label']=='POSITIVE').mean()*100
    st.markdown(f"""<div class='kpi-card'><span style='font-size:32px;'>{pct_positive:.1f}%</span><br/><span style='color:#1d3557;'>Positive</span></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class='kpi-card'><span style='font-size:32px;'>{top_emotion}</span><br/><span style='color:#1d3557;'>Top Emotion</span></div>""", unsafe_allow_html=True)

# --- Divider after KPIs ---
st.markdown("<hr style='margin:25px 0; border:1px solid #e0e0e0;'/>", unsafe_allow_html=True)

# --- Download Button (robust for "top_emotion" KeyError) ---
top_emotion_list = []
for row in emotion_rows:
    emotion_only = {k: v for k, v in row.items() if k != "text"}
    if emotion_only:
        top = max(emotion_only, key=emotion_only.get)
    else:
        top = ""
    top_emotion_list.append(top)

result_df = pd.DataFrame({
    "text": texts,
    "sentiment": sentiment_df["label"],
    "sentiment_score": sentiment_df["score"],
    "top_emotion": top_emotion_list
})
st.download_button("⬇️ Download Results as CSV", result_df.to_csv(index=False), file_name="dashboard_results.csv")

# --- Divider after Download Button ---
st.markdown("<hr style='margin:25px 0; border:1px solid #e0e0e0;'/>", unsafe_allow_html=True)

# --- Pie Chart (Plotly) ---
st.subheader("🎯 Emotion Distribution")
emotion_sums = {k:0 for k in avg_emotion.keys()}
for row in emotion_rows:
    emotion_only = {k: v for k, v in row.items() if k != "text"}
    if emotion_only:
        main_emotion = max(emotion_only, key=emotion_only.get)
        emotion_sums[main_emotion] += 1
if any(emotion_sums.values()):
    fig = px.pie(names=list(emotion_sums.keys()), values=list(emotion_sums.values()), title="Top Detected Emotions")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No emotion data to display in pie chart yet.")

# --- Divider after Pie Chart ---
st.markdown("<hr style='margin:25px 0; border:1px solid #e0e0e0;'/>", unsafe_allow_html=True)

# --- Automated Insights ---
st.subheader("🤖 Automated Insights")
insight = f"""
- **Most common emotion:** {top_emotion}
- **Positive messages:** {pct_positive:.1f}%
- **Negative messages:** {(sentiment_df['label']=='NEGATIVE').mean()*100:.1f}%
"""
if len(texts) > 20:
    insight += f"- **Peak negative sentiment**: Currently available only if you upload time series data.\n"
else:
    insight += f"- **Sample size:** {len(texts)} messages"
st.markdown(insight)

# --- Divider after Automated Insights ---
st.markdown("<hr style='margin:25px 0; border:1px solid #e0e0e0;'/>", unsafe_allow_html=True)

# --- TABS Layout ---
tab1, tab2 = st.tabs(["📊 Sentiment", "📝 Recent Messages"])
with tab1:
    st.bar_chart(sentiment_df['label'].value_counts())
    st.bar_chart(pd.Series(avg_emotion, name="Avg Score"))

with tab2:
    for i, text in enumerate(texts[:10]):
        st.markdown(f"<div style='background-color:#e9ecef; border-radius:12px; padding:10px; margin-bottom:6px;'>"
                    f"<strong>Text:</strong> {text[:150]}{'...' if len(text) > 150 else ''}<br>"
                    f"<strong>Sentiment:</strong> <span style='color:#457b9d'>{sentiments[i]['label']}</span> "
                    f"({sentiments[i]['score']:.2f})<br>"
                    f"<strong>Top Emotion:</strong> <span style='color:#e76f51'>{max(emotions_list[i], key=lambda x: x['score'])['label']}</span> "
                    f"({max(emotions_list[i], key=lambda x: x['score'])['score']:.2f})"
                    f"</div>",
                    unsafe_allow_html=True)
        st.markdown("<hr style='margin:8px 0; border:0.5px solid #dadada;'/>", unsafe_allow_html=True)
