import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

data_sales = pd.read_csv("retail_sales.csv")
data_sales['Total Amount'] = data_sales['Quantity'] * data_sales['Price per Unit']

categorical_features = ['Gender', 'Product Category']
numerical_features = ['Age', 'Quantity', 'Price per Unit']

ohe = OneHotEncoder(drop='first', sparse_output=False)
categorical_encoded = ohe.fit_transform(data_sales[categorical_features])
categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=ohe.get_feature_names_out())

X_sales = pd.concat([data_sales[numerical_features], categorical_encoded_df], axis=1)
y_sales = data_sales['Total Amount']

X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(X_sales, y_sales, test_size=0.2, random_state=42)
model_sales = LinearRegression()
model_sales.fit(X_train_sales, y_train_sales)

y_pred_sales = model_sales.predict(X_test_sales)
mse_sales = mean_squared_error(y_test_sales, y_pred_sales)
r2_sales = r2_score(y_test_sales, y_pred_sales)

print(f"Sales Forecast - MSE: {mse_sales}")
print(f"Sales Forecast - R-squared: {r2_sales}")

plt.scatter(y_test_sales, y_pred_sales)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
