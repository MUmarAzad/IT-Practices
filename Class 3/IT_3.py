# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1n8XBHpDGg29ecpgcr0mbcf-MOG1eyaVS
"""

age = input("Enter your age: ")
if age.isnumeric():
  if int(age) >= 21:
    print("You are eligible for vote")
  else:
    print("The minimum age required is 21")
else:
  print("please Enter a number")

# if-else statements for grades assignment to marks

marks = 60
if marks < 50:
  print("Grade is F")
elif marks < 60:
  print("Grade is E")
elif marks < 70:
  print("Grade is D")
elif marks < 80:
  print("Grade is C")
elif marks < 90:
  print("Grade is B")
else:
  print("Grade is A")

# nested if else example

num = int(input("Enter Number: "))

if num > 5:
  if num % 2 == 0:
    print(num, "is even and greater than 5")
  else:
    print(num, "is odd and greater than 5")
else:
  print(num, "is less than 5")

is_adult = True
has_voter_id = False
if is_adult and has_voter_id:
  print("You are eligible to vote")
else:
  print("You are not eligible to vote")

# Ternary Operator

age = 30
message = "Adult" if age >= 18 else "Not Adult"
print(message)

# match-case statement

color = input("Enter the color: ").lower()

match color:
  case "red":
    print("Stop")
  case "orange":
    print("Ready")
  case "green":
    print("Go")
  case _:
    print("Invalid Color")

# for loop examples

fruits = ["Apples", "Bananas", "Mangos", "Oranges"]
# fruits = "Apples"

for fruit in fruits:
  print(fruit)

# for loop on dictionary

fruits = {
    "Fruit":"Apples",
    "Car":"Suzuki",
    "City":"Tokyo"
}

for key, value in fruits.items():
  print(f"{key}: {value}")

# for loop range

for i in range(10):
  print(i)

# while loop

count = 0
while count < 5:
  print(f"Count is {count}")
  count += 1

# while loop infinite

count = 0

while True:
  print("In Loop")
  if count >= 10:
    break
  count += 1

for i in range(10):
  if i == 5:
    continue
  print(i)

  # pass statement just lets the loop run

# nested loop

# for i in range(3):
#   for j in range(3):
#     print(f"{i},{j}")


for i in range(1,6):
  for j in range(1,11):
    print(f"{i} x {j} = {i*j}")
  print()

# loop else statements

nums = [1, 2, 3, 4, 5]
find = 7

for num in nums:
  if num == find:
    print(f"Found {find}")
else:
  print("Number not Found")

# define functions

def greet(name):
  print(f"Hi {name}! I am learning Python Language")

greet("Umar")

# def add(a, b):
#   return a+b

# print(add(6, 7))

def display_info(name, age = 20): # default age = 20
  print(f"Name: {name}, Age: {age}")

display_info("Umar", 21)

def my_func():
  pass

result = my_func()
print(result)

# Abitrary Argument

# def print_args(*args):
#   for arg in args:
#     print(arg)

# print_args(1, 2, 3, 4, 5, 6)

# Keyword Arguments

def print_kw_args(**kw_args):
  for key, value in kw_args.items():
    print(f"Key: {key}, Value: {value}")

print_kw_args(name = "Umar", age = 21)

# lambda function
# One-liner function

square = lambda x: x**2
print(square(5))