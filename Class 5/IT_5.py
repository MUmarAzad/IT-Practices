# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ykuFHZTlk4n9dB8NCjgi9ORuE361rudV
"""

# dot product

import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot_product = np.dot(a, b)
print(dot_product)

# cross product

# import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

cross_product = np.cross(a, b)
print(cross_product)

# magnitude of vectors

# import numpy as np

a = np.array([1, 2, 3])
magnitude = np.linalg.norm(a)
print(magnitude)  # sqrt(1^2 + 2^2 + 3^2)

# matrix multiplication

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

product_1 = a@b
product_2 = np.matmul(a, b)
print(product_1)
print(product_2)

# Transpose of Matrix

a = np.array([[1, 2, 3], [4, 5, 6]])
transpose = a.T
print(transpose)

# Determinant of Matrix

a = np.array([[1, 2], [3, 4]])
determinant = np.linalg.det(a)
print(determinant)

# Inverse of Matrix

# a = np.array([[1, 2], [3, 4]])
# inverse = np.linalg.inv(a)
# print(inverse)
def inverse_matrix():
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))

    print(f"Enter the {rows}x{cols} matrix row by row:")
    matrix = []
    for i in range(rows):
        row = list(map(float, input().split()))
        matrix.append(row)

    a = np.array(matrix)

    inverse_a = np.linalg.inv(a)
    return inverse_a

result = inverse_matrix()
print(result)

# Eigen Values and Vectors functions

a = np.array([[1, 2], [3, 4]])
eigenvalues, eigenvectors = np.linalg.eig(a)
print(eigenvalues)
print(eigenvectors)

# Concatenate

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = np.concatenate((a, b))
print(c)

# Vertical Stacking

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = np.vstack((a, b))
print(c)

# Horizontal Stacking

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = np.hstack((a, b))
print(c)

# Depth Stacking

a = np.array([[1, 2],
             [3, 4]])
b = np.array([[5, 6],
             [7, 8]])

c = np.dstack((a, b))
print(c.ndim)

# Spliting the arrays

arr = np.array([1, 2, 3, 4, 5, 6])
split = np.split(arr,3)

print(split)

# Vertical Split

a = np.array([[1, 2],
              [3, 4],
              [5, 6]])
v_split = np.vsplit(a, 3)
print(v_split)

# Horizontal Split

a = np.array([[1, 2, 3], [4, 5, 6]])
h_split = np.hsplit(a, 3)
print(h_split)

# Tiling and Repeating

a = np.array([[1, 2], [3, 4]])
tile = np.tile(a, (2, 4))
print(tile)

repeat = np.repeat(a, 3)
print(repeat)

# Pandas Library

import pandas as pd

df = pd.read_csv('data.csv', delimiter = ';')
print(df)

df_excel = pd.read_excel('data.xls')
print(df_excel)

# import json library

import json

df_json = pd.read_json('data.json')
print(df_json)

import sqlite3

conn = sqlite3.connect('data.db')
df_sql = pd.read_sql_query('SELECT * FROM Car_Parts', conn)
print(df_sql)

# Handling large dataset

# df = pd.read_csv('data.csv', chunksize = 5)
# for chunk in df:
#     print(chunk)

chunksize = 200

for chunk in pd.read_csv('data.csv', delimiter = ';', chunksize = chunksize):
    print(chunk.head())     # First 5 rows

# Optimize Memory Usage (Converting 1 data type into another)

df['Username'] = df['Username'].astype('category')
# df['Age'] = df['Age'].astype('float16')
print(df)

# Save all the csp files into Backup folder

import os, glob, shutil
# import pandas as pd

csv_files = glob.glob('*.csv')

for file in csv_files:
    shutil.copy(file, 'Backup')

# Sample Dataframe

data = {'Name' : ['Alice', 'Bob', 'Charlie', None]}

print(df.info)

print(df.dtypes)

print(df.head(2))

print(df.tail(2))

print(df.sample(2))

print(df.isnull().sum) # Detecting missing values

print(df.duplicated().sum) # Detecting duplicate values

boxplot = df.boxplot()

# print(df.dropna)
# df_filled = df.fillna(0)

# print(df_filled)

# print(df.groupby('Name').mean())