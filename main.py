# Import libraries
import re
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
import nltk

# Downloading nltk corpus (first time only)
nltk.download('all')

# 1. DATASET LOADING AND RECOLLECTION
dataset = pd.read_csv(
    hf_hub_download(repo_id='jhan21/amazon-food-reviews-dataset', filename='Reviews.csv', repo_type="dataset")
)
print(dataset.head())

# *** para prueba (BORRAR) 
dataset = dataset.head(10000) # Select the first 200 rows
print(dataset.shape)
# 2. DATA PREPROCESSING

# Add a new column 'Sentiment' which is 'positive' if Score > 3, otherwise 'negative'
dataset['Sentiment'] = dataset['Score'].apply(lambda x: 'positive' if x > 3 else 'negative')

# Select relevant columns and drop missing values
dataset = dataset[['Text', 'Sentiment']].dropna()

# Convert labels to binary
dataset['Sentiment'] = dataset['Sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Split dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    dataset['Text'], dataset['Sentiment'], test_size=0.2, random_state=42
)

# Tokenize the text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)
word_index = tokenizer.word_index

# Convert texts to sequences
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences to ensure uniform input size
max_length = 100
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

print(train_padded.shape)
print(test_padded.shape)
'''
# create preprocess_text function
def preprocess_text(text):

    # Tokenize the text
    tokens = word_tokenize(text.lower())

   # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

   # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# apply the function df
dataset['Text'] = dataset['Text'].apply(preprocess_text)

'''
# 3. RNN MODEL (EMBEDDING, LSTM AND DENSE LAYERS)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=max_length), # poner input_dim = 10000
    # trabajar con input_shape
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
print('Model Summary', model.summary())

# 3. MODEL TRAINING
model.fit(
    train_padded,
    np.array(train_labels),
    epochs=10,
    validation_data=(test_padded, np.array(test_labels)),
    verbose=2
)

# 4. EVALUATION
loss, accuracy = model.evaluate(test_padded, np.array(test_labels))
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

# 5. IMPLEMENTATION
# Function to predict the sentiment of a single text

def predict_sentiment(text):
    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

    # Make a prediction
    prediction = model.predict(padded)

    # Convert the prediction to a binary output
    sentiment = 'positive' if prediction > 0.5 else 'negative'

    return sentiment

# Example usage
example_text = "The product was bad and I really hated it!"
predicted_sentiment = predict_sentiment(example_text)
print(f"The sentiment of the review 1 is: {predicted_sentiment}")

example_text2 = "The product was just ok and I liked it"
predicted_sentiment2 = predict_sentiment(example_text2)
print(f"The sentiment of the review 2 is: {predicted_sentiment2}")

example_text3 = "The product was fine and I really loved it!"
predicted_sentiment3 = predict_sentiment(example_text3)
print(f"The sentiment of the review 3 is: {predicted_sentiment3}")