import pickle

# Load saved model
model = pickle.load(open("models/fake_news_model.pkl", "rb"))

# Load saved vectorizer
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Take news from user
news = input("Enter the news: ")

# Convert text into numbers
news_vector = vectorizer.transform([news])

# Predict
prediction = model.predict(news_vector)

# Show result
if prediction[0] == 0:
    print("\nPrediction: FAKE NEWS")
else:
    print("\nPrediction: REAL NEWS")