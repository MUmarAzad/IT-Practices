import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
import streamlit as st 

file_path = "owid-energy-data.csv" 
df = pd.read_csv(file_path) 

st.write("Dataset Columns:", df.columns.tolist()) 

energy_columns = [col for col in df.columns if "consumption" in col.lower()] 
if not energy_columns: 
    raise KeyError("No column related to energy consumption found in the dataset.") 

energy_column = energy_columns[0] 
st.write(f"Using '{energy_column}' as the target variable.") 

df.dropna(inplace=True) 

categorical_cols = df.select_dtypes(include=['object']).columns.tolist() 
if categorical_cols: 
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True) 

X = df.drop(columns=[energy_column]) 
y = df[energy_column] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

model = LinearRegression() 
model.fit(X_train, y_train) 

y_pred = model.predict(X_test) 

mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred) 

st.title('Energy Consumption Prediction') 
st.write(f'MSE: {mse:.2f}, R-squared: {r2:.2f}') 

fig, ax = plt.subplots() 
ax.scatter(y_test, y_pred, alpha=0.5, color='blue') 
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2) 
ax.set_xlabel('Actual Energy Consumption') 
ax.set_ylabel('Predicted Energy Consumption') 
st.pyplot(fig) 

st.sidebar.header('Predict Energy Consumption') 
features = {col: st.sidebar.number_input(f'Enter {col}:', 
float(df[col].min()), float(df[col].max())) for col in X.columns} 
 
if st.sidebar.button('Predict'): 
    input_data = np.array([features[col] for col in X.columns]).reshape(1, -1) 
    prediction = model.predict(input_data)[0] 
    st.sidebar.write(f'Predicted Energy Consumption: {prediction:.2f} kWh')