# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords from NLTK
nltk.download('stopwords')

# Sample dataset: Positive and Negative Sentences
# A larger dataset to improve performance
data = {
    "text": [
        "The service was fantastic, highly recommend!",
        "Worst experience ever, do not buy.",
        "I love this product, it works perfectly.",
        "Terrible quality, I want a refund.",
        "Very satisfied with my purchase, excellent service.",
        "The item broke after one use, very disappointed.",
        "Amazing! Exceeded my expectations.",
        "Not worth the money, horrible experience.",
        "Fast delivery and great customer support.",
        "The packaging was damaged, not happy at all.",
        "This is the best thing I've bought this year.",
        "I regret buying this, it was a complete waste.",
    ],
    "sentiment": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: Positive, 0: Negative
}

# Load the dataset into a DataFrame
df = pd.DataFrame(data)

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = text.split()
    text = " ".join(word for word in words if word not in stop_words)
    return text

# Apply preprocessing to the text column
df['text'] = df['text'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Convert text data to numerical data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the model using Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Test the model with new inputs
new_texts = [
    "The product was amazing and I loved it!",
    "Horrible experience, would never recommend.",
    "Fast shipping and the product is great.",
    "Not good at all, I am disappointed."
]
# Preprocess and vectorize the new texts
new_texts_preprocessed = [preprocess_text(text) for text in new_texts]
new_texts_vectorized = vectorizer.transform(new_texts_preprocessed)

# Predict sentiment for the new texts
predictions = model.predict(new_texts_vectorized)
for text, sentiment in zip(new_texts, predictions):
    sentiment_label = "Positive" if sentiment == 1 else "Negative"
    print(f"Text: '{text}' -> Sentiment: {sentiment_label}")
