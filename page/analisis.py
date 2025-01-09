import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm
from stqdm import stqdm
import nltk
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
#model_dir = "models"
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = BertForSequenceClassification.from_pretrained("diiadi/sentiment-gplay")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to preprocess and predict
def preprocess_data(df):
    # Lowercasing
    df["textDisplay_lower"] = [lower_case(text) for text in stqdm(df["textDisplay"], desc="Lowercasing Text")]
    st.write("After Lowercasing:")
    st.dataframe(df[["textDisplay", "textDisplay_lower"]])

    # Removing Punctuation
    df["textDisplay_no_punctuation"] = [remove_punctuation(text) for text in stqdm(df["textDisplay_lower"], desc="Removing Punctuation")]
    st.write("After Removing Punctuation:")
    st.dataframe(df[["textDisplay", "textDisplay_no_punctuation"]])

    # Removing Emojis
    df["textDisplay_no_emoji"] = [remove_emoji(text) for text in stqdm(df["textDisplay_no_punctuation"], desc="Removing Emojis")]
    st.write("After Removing Emojis:")
    st.dataframe(df[["textDisplay", "textDisplay_no_emoji"]])

    # Removing Stopwords
    df["textDisplay_no_stopwords"] = [remove_stopwords(text) for text in stqdm(df["textDisplay_no_emoji"], desc="Removing Stopwords")]
    st.write("After Removing Stopwords:")
    st.dataframe(df[["textDisplay", "textDisplay_no_stopwords"]])

    # Stemming
    df["textDisplay_stemmed"] = [apply_stemming(text) for text in stqdm(df["textDisplay_no_stopwords"], desc="Applying Stemming")]
    st.write("After Stemming:")
    st.dataframe(df[["textDisplay", "textDisplay_stemmed"]])

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

# Generate WordCloud
def generate_wordcloud(data, sentiment, column):
    words = " ".join(data[data["predicted"] == sentiment][column])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(words)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"WordCloud for {sentiment} Sentiment")
    st.pyplot(fig)

# Streamlit UI
st.title("Sentiment Analysis Youtube")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Original Data:")
    st.dataframe(df)

    if "textDisplay" not in df.columns:
        st.error("The uploaded file must contain a 'textDisplay' column.")
    else:
        # Preprocessing
        st.subheader("Preprocessing Data")
        df = preprocess_data(df)

        # Predict Sentiments
        st.subheader("Predicting Sentiments")
        df["predicted"] = predict_sentiment(df["textDisplay_stemmed"])

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

        # Display WordClouds
        st.subheader("WordClouds")
        st.write("Positive Sentiment:")
        generate_wordcloud(df, 1, "textDisplay_stemmed")

        st.write("Negative Sentiment:")
        generate_wordcloud(df, 0, "textDisplay_stemmed")

        # Downloadable CSV
        st.subheader("Download Results")
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="predicted_results.csv", mime="text/csv")
