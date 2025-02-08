import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm
from stqdm import stqdm
import nltk
from collections import Counter
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import emoji
import string
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Setup
nltk.download("stopwords")
indonesian_stopwords = stopwords.words("indonesian")
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Preprocessing Functions
def lower_case(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def remove_emoji(text):
    return emoji.replace_emoji(text, replace="")

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in indonesian_stopwords]
    return " ".join(filtered_words)

def apply_stemming(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)

# Load Model and Tokenizer
# model_dir = "models"
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = BertForSequenceClassification.from_pretrained("diiadi/sentiment-gplay")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to preprocess and predict
def preprocess_data(df):
    # Lowercasing
    df["content_lower"] = [lower_case(text) for text in stqdm(df["content"], desc="Lowercasing Text")]
    st.write("After Lowercasing:")
    st.dataframe(df[["content", "content_lower"]])

    # Removing Punctuation
    df["content_no_punctuation"] = [remove_punctuation(text) for text in stqdm(df["content_lower"], desc="Removing Punctuation")]
    st.write("After Removing Punctuation:")
    st.dataframe(df[["content", "content_no_punctuation"]])

    # Removing Emojis
    df["content_no_emoji"] = [remove_emoji(text) for text in stqdm(df["content_no_punctuation"], desc="Removing Emojis")]
    st.write("After Removing Emojis:")
    st.dataframe(df[["content", "content_no_emoji"]])

    # Removing Stopwords
    df["content_no_stopwords"] = [remove_stopwords(text) for text in stqdm(df["content_no_emoji"], desc="Removing Stopwords")]
    st.write("After Removing Stopwords:")
    st.dataframe(df[["content", "content_no_stopwords"]])

    # Stemming
    df["content_stemmed"] = [apply_stemming(text) for text in stqdm(df["content_no_stopwords"], desc="Applying Stemming")]
    st.write("After Stemming:")
    st.dataframe(df[["content", "content_stemmed"]])

    return df

def predict_sentiment(texts):
    model.eval()
    predictions = []
    for text in stqdm(texts, desc="Predicting Sentiments"):
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        predictions.append(pred)
    return predictions

# Fungsi untuk Generate WordCloud dengan Frekuensi, Tabel, dan Grafik
def generate_wordcloud_with_frequency(data, sentiment, column):
    words = " ".join(data[data["predicted"] == sentiment][column])
    word_counts = Counter(words.split())

    # Generate WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_counts)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.set_title(f"WordCloud untuk Sentimen {sentiment} (dengan Frekuensi)")
    st.pyplot(fig)

    # Tampilkan Tabel Frekuensi Kata
    st.subheader(f"Frekuensi Kata untuk Sentimen {sentiment}")
    most_common = word_counts.most_common(20)  # Ambil 20 kata paling umum
    freq_df = pd.DataFrame(most_common, columns=["Kata", "Frekuensi"])
    st.dataframe(freq_df)  # Tampilkan dalam tabel Streamlit

    # Tampilkan Grafik Frekuensi Kata (Seaborn)
    st.subheader(f"Grafik Frekuensi Kata untuk Sentimen {sentiment}")
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    sns.barplot(x='Kata', y='Frekuensi', data=freq_df)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
    plt.title(f"Top 20 Kata - Sentimen {sentiment}")
    st.pyplot(plt.gcf())  # Use st.pyplot to display the Matplotlib chart

# Streamlit UI
st.title("Sentiment Analysis Playstore")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Original Data:")
    st.dataframe(df)

    if "content" not in df.columns:
        st.error("The uploaded file must contain a 'content' column.")
    else:
        # Preprocessing
        st.subheader("Preprocessing Data")
        df = preprocess_data(df)

        # Predict Sentiments
        st.subheader("Predicting Sentiments")
        df["predicted"] = predict_sentiment(df["content_stemmed"])

        # Assume labels are included in dataset for evaluation
        if "label" in df.columns:
            # Map string labels to integers
            label_mapping = {"negative": 0, "positive": 1}
            y_true = df["label"].map(label_mapping)
            
            # Calculate Metrics
            y_pred = df["predicted"]
            accuracy = accuracy_score(y_true, y_pred)
            class_report = classification_report(y_true, y_pred, target_names=["negative", "positive"], output_dict=True)
            confusion = confusion_matrix(y_true, y_pred)

            # Display Accuracy
            st.subheader("Accuracy Score")
            st.write(f"Accuracy: **{accuracy:.2f}**")

            # Display Classification Report
            st.subheader("Classification Report")
            st.table(pd.DataFrame(class_report).transpose())

            # Display Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)

        # Display Pie Chart
        st.subheader("Sentiment Distribution")
        sentiment_counts = df["predicted"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        sentiment_counts.plot.pie(
            autopct="%1.1f%%",
            labels=["negative", "positive"],
            ax=ax,
            ylabel=""  # Remove default ylabel
        )
        st.pyplot(fig)

        # Tampilkan WordCloud dengan Frekuensi, Tabel, dan Grafik
        st.subheader("WordCloud dengan Frekuensi, Tabel, dan Grafik")
        st.write("Sentimen Positif:")
        generate_wordcloud_with_frequency(df, 1, "content_stemmed")

        st.write("Sentimen Negatif:")
        generate_wordcloud_with_frequency(df, 0, "content_stemmed")

        # Downloadable CSV
        st.subheader("Download Results")
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="predicted_results.csv", mime="text/csv")
