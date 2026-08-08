import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
fake=pd.read_csv("dataset/Fake.csv")
true=pd.read_csv("dataset/True.csv")
fake["label"]=0
true["label"]=1
news=pd.concat([fake,true],ignore_index=True)
news=news.sample(frac=1,random_state=42)
news=news[["text","label"]]
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z\s]',' ', text)
    text=re.sub(r'\s+',' ',text)
    return text
news["text"]=news["text"].apply(clean_text)
print("missing values")
print(news.isnull().sum())
print("\nCleaned dataset:")
print(news.head())

X=news["text"]
y=news["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)
print("\nTraining Data:",len(X_train))
print("Testing Data:",len(X_test))

vectorizer=TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vector=vectorizer.fit_transform(X_train)
X_test_vector=vectorizer.transform(X_test)
print("\nText converted into numbers successfully!")
model=LogisticRegression()
model.fit(X_train_vector ,y_train)
print("Model trained successfully!")


prediction=model.predict(X_test_vector)
accuracy=accuracy_score(y_test,prediction)
print ("\nModel Accuracy:",accuracy) 

pickle.dump(model, open("models/fake_news_model.pkl","wb"))
pickle.dump(vectorizer,open("models/vectorizer.pkl","wb"))
print("Model saved successfully!")



