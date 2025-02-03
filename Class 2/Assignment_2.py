# ---------------Name Formatter and Length Calculator---------------

first_name = input("Enter your first name: ")
last_name = input("Enter your last name: ")

first_name = first_name.upper()
last_name = last_name.lower()

total_letters = len(first_name) + len(last_name)

print(f"First name (upper): {first_name}")
print(f"Last name (lower): {last_name}")
print("Sum of letters in your first and last name:", total_letters)

# ---------------Simple Area Calculator---------------

print("Calculate the area of the following shapes:")
print("1. Circle")
print("2. Rectangle")
print("3. Square")
print("4. Triangle")

# Circle
radius = float(input("Enter the radius of the circle: "))
area_circle = 3.1416 * (radius ** 2)
print("The area of the circle is:", area_circle)

# Rectangle
length = float(input("Enter the length of the rectangle: "))
width = float(input("Enter the width of the rectangle: "))
area_rectangle = length * width
print("The area of the rectangle is:", area_rectangle)

# Square
side = float(input("Enter the side length of the square: "))
area_square = side ** 2
print("The area of the square is:", area_square)

# Triangle
base = float(input("Enter the base of the triangle: "))
height = float(input("Enter the height of the triangle: "))
area_triangle = 0.5 * base * height
print("The area of the triangle is:", area_triangle)

# ---------------Random Color-Based Password Generator---------------

import random

colors = ["red", "blue", "green", "yellow", "orange", "purple"]
index = random.randint(0, len(colors) - 1)
selected_color = colors[index]
password = selected_color[::-1]

# Step 6: Print results
print("Selected Color:", selected_color)
print("Generated Password:", password)
