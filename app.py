from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the saved Machine Learning model
model = pickle.load(open("models/fake_news_model.pkl", "rb"))

# Load the saved TF-IDF vectorizer
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        news = request.form["news"]

        # Convert news into numbers
        news_vector = vectorizer.transform([news])

        # Make prediction
        result = model.predict(news_vector)

        if result[0] == 0:
            prediction = "🔴 FAKE NEWS"
        else:
            prediction = "🟢 REAL NEWS"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)