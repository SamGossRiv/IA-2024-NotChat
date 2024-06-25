import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from huggingface_hub import hf_hub_download

# Load the dataset
dataset = pd.read_csv(hf_hub_download(repo_id='jhan21/amazon-food-reviews-dataset', filename='Reviews.csv', repo_type="dataset"))

# Display the first few rows of the dataset
print(dataset.head())

# Display dataset information
print(dataset.info())

# Display basic statistics
print(dataset.describe())

# Add a 'Sentiment' column based on 'Score'
dataset['Sentiment'] = dataset['Score'].apply(lambda x: 'positive' if x > 3 else 'negative')

# Convert 'Sentiment' to binary
dataset['Sentiment'] = dataset['Sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Display the distribution of sentiments
print(dataset['Sentiment'].value_counts())

# Plot the distribution of sentiments
sns.countplot(x='Sentiment', data=dataset)
plt.title('Distribution of Sentiments')
plt.show()

# Calculate the length of each review
dataset['TextLength'] = dataset['Text'].apply(len)

# Plot the distribution of review lengths
sns.histplot(dataset['TextLength'], bins=50)
plt.title('Distribution of Review Lengths')
plt.show()

# Generate word clouds for positive and negative reviews
positive_reviews = dataset[dataset['Sentiment'] == 1]['Text']
negative_reviews = dataset[dataset['Sentiment'] == 0]['Text']

positive_words = ' '.join(positive_reviews)
negative_words = ' '.join(negative_reviews)

positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_words)
negative_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(negative_words)

# Display word cloud for positive reviews
plt.figure(figsize=(10, 5))
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Frequent Words in Positive Reviews')
plt.show()

# Display word cloud for negative reviews
plt.figure(figsize=(10, 5))
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Frequent Words in Negative Reviews')
plt.show()

# Compute and display the correlation matrix
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Analyze bigrams (pairs of words) frequency
vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
bigrams = vectorizer.fit_transform(dataset['Text'])
bigram_freq = pd.DataFrame(bigrams.sum(axis=0), columns=vectorizer.get_feature_names_out()).T
bigram_freq.columns = ['Frequency']
bigram_freq = bigram_freq.sort_values(by='Frequency', ascending=False).head(10)

# Display the top 10 most frequent bigrams
print(bigram_freq)