#--------------------------To-do list---------------------

tasks = []

def add_task(task):
    tasks.append(task)

def remove_task(task):
    if task in tasks:
        tasks.remove(task)

def view_tasks():
    for task in tasks:
        print(task)

add_task("Finish homework")
add_task("Buy groceries")
view_tasks()
remove_task("Buy groceries")
view_tasks()

#--------------------------Find Maximum and Minimum Values---------------------

def find_max_min(lst):
    max_val = lst[0]
    min_val = lst[0]
    for num in lst:
        if num > max_val:
            max_val = num
        if num < min_val:
            min_val = num
    return max_val, min_val

numbers = [3, 9, 2, 8, 1]
max_val, min_val = find_max_min(numbers)
print(f"Max: {max_val}, Min: {min_val}")

#--------------------------Phonebook---------------------

phonebook = {}

def add_contact(name, number):
    phonebook[name] = number

def get_number(name):
    return phonebook.get(name, "Contact not found")

add_contact("Alice", "1234567890")
add_contact("Bob", "9876543210")
print(get_number("Alice"))
print(get_number("Charlie"))

#--------------------------Inventory System---------------------

inventory = {}

def add_item(item, quantity):
    if item in inventory:
        inventory[item] += quantity
    else:
        inventory[item] = quantity

def remove_item(item, quantity):
    if item in inventory and inventory[item] >= quantity:
        inventory[item] -= quantity
        if inventory[item] == 0:
            del inventory[item]
    else:
        print(f"Insufficient quantity of {item}.")

def view_inventory():
    for item, quantity in inventory.items():
        print(f"{item}: {quantity}")

add_item("Apple", 10)
add_item("Banana", 5)
remove_item("Apple", 3)
view_inventory()

#--------------------------ATM System---------------------

def deposit(balance, amount):
    balance += amount
    print(f"Deposited: {amount}. New balance: {balance}")
    return balance

def withdraw(balance, amount):
    if amount <= balance:
        balance -= amount
        print(f"Withdrew: {amount}. New balance: {balance}")
    else:
        print("Insufficient funds")
    return balance

def check_balance(balance):
    print(f"Current balance: {balance}")

balance = 1000
balance = deposit(balance, 500)
balance = withdraw(balance, 200)
check_balance(balance)
