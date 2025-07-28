import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
import re
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Data cleaning & preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    cleaned_text = ' '.join(tokens)

    return cleaned_text

# Analyze sentiment
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)

    if scores['compound'] >= 0.05:
        return 'Positive', scores
    elif scores['compound'] <= -0.05:
        return 'Negative', scores
    else:
        return 'Neutral', scores



# App
def main():
    st.set_page_config(page_title="Sentiment Analysis", layout="centered")

    st.markdown("""
        <style>
            .main {background-color: #f5f5f5;}
            .title {color: #2c3e50; font-size: 36px; font-weight: bold; text-align: center;}
            .subheader {color: #2c3e50; font-size: 20px; text-align: center;}
            .input-box {background-color: #ffffff; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
            .result-box {padding: 15px; border-radius: 5px; margin-top: 20px;}
            .positive {background-color: #d4edda; color: #155724;}
            .negative {background-color: #f8d7da; color: #721c24;}
            .neutral {background-color: #e2e3e5; color: #383d41;}
            h2 {
            color: #2c3e50 !important;
            }
            h3 {
                color: #2c3e50 !important;
            }
            .stMarkdown p {
            color: #2c3e50 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Analyze the sentiment of text or CSV data</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Single Text Analysis", "CSV File Analysis"])

    with tab1:
        st.subheader("Enter text for Sentiment Analysis")
        user_input = st.text_area("Type or paste your text here:", height=150, key="text_input", help="Enter a sentence or paragraph to analyze its sentiment.")

        if st.button("Analyze Sentiment", key="analyze_button"):
            if user_input:
                cleaned_text = clean_text(user_input)
                sentiment, scores = analyze_sentiment(cleaned_text)

                st.markdown(f'<div class="result-box {sentiment.lower()}">'
                           f'<strong>Sentiment:</strong> {sentiment}<br>'
                           f'<strong>Compound Score:</strong> {scores["compound"]:.2f}<br>'
                           f'<strong>Positive:</strong> {scores["pos"]:.2f}<br>'
                           f'<strong>Negative:</strong> {scores["neg"]:.2f}<br>'
                           f'<strong>Neutral:</strong> {scores["neu"]:.2f}</div>', unsafe_allow_html=True)


            else:
                st.warning("Please enter some text to analyze.")

    with tab2:
        st.subheader("Upload a CSV File for Batch Analysis")
        uploaded_file = st.file_uploader("Choose a CSV file with a 'text' column", type=["csv"], help="CSV should have a 'text' column with text data to analyze.")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'text' not in df.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                df['cleaned_text'] = df['text'].apply(clean_text)
                df['sentiment'], df['scores'] = zip(*df['cleaned_text'].apply(analyze_sentiment))
                df['compound'] = df['scores'].apply(lambda x: x['compound'])

                st.write("Sentiment Analysis Results:")
                st.dataframe(df[['text', 'sentiment', 'compound']])

                # visualization
                st.subheader("Sentiment Distribution")
                fig, ax = plt.subplots()
                sns.countplot(x='sentiment', data=df, palette='viridis', order=['Positive', 'Neutral', 'Negative'])
                plt.xlabel("Sentiments")
                plt.ylabel("Count")
                st.pyplot(fig)

if __name__ == "__main__":
    main()