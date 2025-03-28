import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 

# df = pd.read_csv('Class 7/Assignment_7_Tasks/house_data.csv')
df = pd.read_csv('house_data.csv')

non_numeric_columns = df.select_dtypes(include=['object']).columns.tolist() 
print(f"Non-numeric columns: {non_numeric_columns}") 

if non_numeric_columns:
    df = pd.get_dummies(df, columns=non_numeric_columns, drop_first=True) 

if 'price' not in df.columns: 
    raise KeyError("The dataset does not contain a 'price' column. Please check the CSV file.") 

X = df.drop(columns=['price'])
y = df['price']

if not np.issubdtype(X.dtypes.values[0], np.number): 
    raise ValueError("Some features are still non-numeric. Check the dataset preprocessing.") 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

model = LinearRegression() 
model.fit(X_train, y_train) 

y_pred = model.predict(X_test) 


mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred) 
 
print(f'Model Evaluation:\nMSE: {mse:.2f}, R-squared: {r2:.2f}')