import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 

df = pd.read_csv('salary_data.csv') 

df.dropna(inplace=True)
categorical_columns = ['degree', 'job_role', 'location'] 
existing_categorical_columns = [col for col in categorical_columns if col in 
df.columns] 

if existing_categorical_columns: 
    df = pd.get_dummies(df, columns=existing_categorical_columns, 
drop_first=True) 

if 'Salary' in df.columns: 
    X = df.drop(columns=['Salary']) 
    y = df['Salary'] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42) 

    model = LinearRegression() 
    model.fit(X_train, y_train) 

    y_pred = model.predict(X_test) 

    mse = mean_squared_error(y_test, y_pred) 
    r2 = r2_score(y_test, y_pred) 
 
    print(f'MSE: {mse}, R-squared: {r2}') 

    sample_input = X_test.iloc[[0]]
    predicted_salary = model.predict(sample_input) 
    print(f'Predicted Salary for sample input: {predicted_salary[0]}') 
 
else: 
    print("Error: The 'Salary' column is missing from the dataset.")