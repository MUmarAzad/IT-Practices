import pandas as pd 
import numpy as np 
import re 
import nltk 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, classification_report 

# Load dataset with correct column names and handle extra commas
df = pd.read_csv("spam.csv", encoding="ISO-8859-1", usecols=[0, 1], names=["label", "message"], skiprows=1)

# Convert labels to binary (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text Preprocessing
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 

ps = PorterStemmer()
corpus = []
stop_words = set(stopwords.words('english'))  # Load stopwords only once

for msg in df['message'].dropna():  # Handle potential missing values
    msg = re.sub(r'[^a-zA-Z]', ' ', msg).lower().split()
    msg = [ps.stem(word) for word in msg if word not in stop_words]
    corpus.append(" ".join(msg))

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus).toarray()
y = df['label'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
