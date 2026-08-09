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

### For prediction

1. Install Python.
2. Install the required libraries:

pip install -r requirements.txt

3. Run:

python predict.py

4. Enter a news article when prompted.

### For training the model

The original dataset is not included in this repository because the CSV files are larger than GitHub's normal upload limit.

If you have the dataset locally, place the files here:

dataset/
├── Fake.csv
└── True.csv

Then run:

python main.py


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


