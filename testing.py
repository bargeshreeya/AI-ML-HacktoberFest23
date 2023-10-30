from joblib import load

# Load the trained model and TF-IDF vectorizer
model = load('sentiment_model.pkl')
tfidf_vectorizer = load('tfidf_vectorizer.pkl')

# Function to analyze sentiment
def analyze_sentiment(text):
    # Transform the text into TF-IDF features using the loaded vectorizer
    tfidf_features = tfidf_vectorizer.transform([text])
    sentiment = model.predict(tfidf_features)[0]
    return sentiment

# Sample text for testing
new_text = "This is a great movie, I loved it!"
new_text1 = "The food was horrible but the service was great "
new_text2 = "I'm okay"

predicted_sentiment = analyze_sentiment(new_text)
print(f"Predicted sentiment: {predicted_sentiment}")
predicted_sentiment = analyze_sentiment(new_text1)
print(f"Predicted sentiment: {predicted_sentiment}")
predicted_sentiment = analyze_sentiment(new_text2)
print(f"Predicted sentiment: {predicted_sentiment}")
