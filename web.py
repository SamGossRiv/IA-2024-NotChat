from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
import string
import pickle
import webbrowser
from threading import Timer


app = Flask(__name__)

# Load model and tokenizer
with open('sentiment_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("sentiment_model_weights.weights.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)


def predict_sentiment_loaded_model(text):
    if not text:
        return "Invalid input text"

    sequence = tokenizer.texts_to_sequences([text])

    if len(sequence[0]) == 0:
        return "Unable to process text"

    padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    prediction = loaded_model.predict(padded)
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    return sentiment


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    comment = data.get('comment', '')
    processed_text = preprocess_text(comment)
    prediction = predict_sentiment_loaded_model(processed_text)
    return jsonify({'prediction': prediction})

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True)

