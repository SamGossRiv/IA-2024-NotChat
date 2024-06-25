import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import model_from_json
# Downloading nltk corpus (first time only)
#nltk.download('all')

# 1. DATASET LOADING AND RECOLLECTION
dataset = pd.read_csv(
    hf_hub_download(repo_id='jhan21/amazon-food-reviews-dataset', filename='Reviews.csv', repo_type="dataset")
)
print(dataset.head())

# *** para prueba (BORRAR)
dataset = dataset.head(20000) # Select the first 200 rows
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
    tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=max_length),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
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
    epochs=30,
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
# Ejemplo 1 +
example_text1 = "The product was nice and I really love it!"
predicted_sentiment1 = predict_sentiment(example_text1)
print(f"Example 1: The sentiment of the review is: {predicted_sentiment1}")

# Ejemplo 2 -
example_text2 = "This is the worst product I have ever bought."
predicted_sentiment2 = predict_sentiment(example_text2)
print(f"Example 2: The sentiment of the review is: {predicted_sentiment2}")

# Ejemplo 3 +
example_text3 = "I'm satisfied with the quality of the product."
predicted_sentiment3 = predict_sentiment(example_text3)
print(f"Example 3: The sentiment of the review is: {predicted_sentiment3}")

# Ejemplo 4 -
example_text4 = "The item arrived broken and late. Very disappointing."
predicted_sentiment4 = predict_sentiment(example_text4)
print(f"Example 4: The sentiment of the review is: {predicted_sentiment4}")

# Ejemplo 5 +
example_text5 = "Excellent service and great quality!"
predicted_sentiment5 = predict_sentiment(example_text5)
print(f"Example 5: The sentiment of the review is: {predicted_sentiment5}")

# Ejemplo 6 -
example_text6 = "I will never buy this product again. It's awful."
predicted_sentiment6 = predict_sentiment(example_text6)
print(f"Example 6: The sentiment of the review is: {predicted_sentiment6}")

# Ejemplo 7 +
example_text7 = "The taste was amazing, I'm really impressed."
predicted_sentiment7 = predict_sentiment(example_text7)
print(f"Example 7: The sentiment of the review is: {predicted_sentiment7}")

# Ejemplo 8 -
example_text8 = "Not what I expected. The product is of low quality."
predicted_sentiment8 = predict_sentiment(example_text8)
print(f"Example 8: The sentiment of the review is: {predicted_sentiment8}")

# Ejemplo 9 +
example_text9 = "Great value for the money. Highly recommend it."
predicted_sentiment9 = predict_sentiment(example_text9)
print(f"Example 9: The sentiment of the review is: {predicted_sentiment9}")

# Ejemplo 10 -
example_text10 = "Terrible experience. The product does not work as advertised."
predicted_sentiment10 = predict_sentiment(example_text10)
print(f"Example 10: The sentiment of the review is: {predicted_sentiment10}")

# Guardar la arquitectura del modelo en formato JSON
model_json = model.to_json()
with open("sentiment_model.json", "w") as json_file:
    json_file.write(model_json)

# Guardar los pesos del modelo en formato HDF5
model.save_weights("sentiment_model_weights.weights.h5")

# Cargar la arquitectura del modelo desde el archivo JSON
with open('sentiment_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Cargar los pesos del modelo al modelo cargado
loaded_model.load_weights("sentiment_model_weights.weights.h5")

# Compilar el modelo (asegúrate de compilarlo con los mismos parámetros que utilizaste antes)
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Usar el modelo cargado para hacer predicciones
def predict_sentiment_loaded_model(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = loaded_model.predict(padded)
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    return sentiment

# Ejemplo de uso con el modelo cargado
predicted_sentiment = predict_sentiment_loaded_model("This product is great!")
print(f"The sentiment of the review is: {predicted_sentiment}")