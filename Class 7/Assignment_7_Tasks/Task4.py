import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

try:
    df = pd.read_csv('customer_churn_data.csv')  
except FileNotFoundError:
    st.error("Error: Dataset file not found. Please check the file path.")
    st.stop()

st.write("Dataset Preview:", df.head())

if df.isnull().sum().any():
    st.warning("Missing values detected. Handling them...")
    df.dropna(inplace=True)  

if 'churn' not in df.columns:
    st.error("Error: 'churn' column not found in dataset.")
    st.stop()

df['churn'] = df['churn'].astype(str).str.strip().map({'False': 0, 'True': 1})

categorical_columns = ['international_plan', 'voice_mail_plan']
for col in categorical_columns:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

columns_to_drop = ['Id', 'state', 'phone_number']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

if df.empty:
    st.error("Error: Dataset is empty after preprocessing.")
    st.stop()

X = df.drop(columns=['churn'])
y = df['churn']

if X.empty or y.empty:
    st.error("Error: Features or target variable is empty.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

st.write(f'Accuracy: {accuracy:.2f}')
st.text(report)

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig)

st.sidebar.header('Predict Customer Churn')
features = {}
for col in X.columns:
    features[col] = st.sidebar.number_input(f'Enter {col}:', float(X[col].min()), float(X[col].max()))

if st.sidebar.button('Predict'):
    input_data = np.array([features[col] for col in X.columns]).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    st.sidebar.write(f'Predicted Churn: {"Yes" if prediction == 1 else "No"}')
