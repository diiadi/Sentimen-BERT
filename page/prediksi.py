# app.py

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the saved model and tokenizer
MODEL_DIR = "models"  # Ganti sesuai lokasi folder model
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Function to perform prediction
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        predicted_label = torch.argmax(logits, dim=1).item()
        return predicted_label, probs[0]

# Streamlit UI
st.header("Prediksi Sentiment")

# Input field
user_input = st.text_input("Masukan kalimat:", "")

# Predict button
if st.button("Prediksi", type="primary"):
    if user_input.strip():
        label, probabilities = predict_sentiment(user_input)
        sentiment = "Positive" if label == 1 else "Negative"
        st.write(f"### Predicted Sentiment: **{sentiment}**")
        st.write(f"Confidence: {probabilities[label]:.2%}")
    else:
        st.write("Please enter a valid sentence!")
