# Fake News Detection

A Machine Learning project that detects whether a news article is **Fake** or **Real**.

## About the Project

This project uses Machine Learning and Natural Language Processing (NLP) to classify news articles.

The text of the news article is converted into numerical features using **TF-IDF Vectorization** and then classified using **Logistic Regression**.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Machine Learning
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression

## Model Accuracy

The trained model achieved approximately **98.4% accuracy** on the test dataset.

## Project Structure

```text
fake-news-detection/
│
├── main.py
├── predict.py
├── requirements.txt
├── README.md
│
└── models/
    ├── fake_news_model.pkl
    └── vectorizer.pkl
## dataset
The datset contains the fake and real news article
The datset files re not included in this repository because they are larger tha Github's normal file upload limit
## How to Run

1. Install Python.
2. Install the required libraries:

   pip install -r requirements.txt

3. Run the prediction program:

   python predict.py

## Dataset

The dataset contains fake and real news articles.

The dataset files are not included in this repository because of their large file size.

## Project Structure

fake-news-detection/
├── models/
│   ├── fake_news_model.pkl
│   ├── vectorizer.pkl
│   └── README.md
├── main.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore


