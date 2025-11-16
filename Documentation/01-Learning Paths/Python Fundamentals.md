# Python Fundamentals for Beginners

## 🐍 Introduction to Python

### What is Python?
Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in various fields including web development, data analysis, artificial intelligence, and quantitative finance.

### Why Learn Python?
- Easy to learn and read
- Large standard library
- Cross-platform compatibility
- Strong community support
- Extensive third-party packages
- Excellent for data analysis and scientific computing

## 📝 Basic Syntax

### Hello World
```python
print("Hello, World!")
```

### Variables and Data Types
```python
# Numbers
x = 5           # integer
y = 3.14        # float

# Strings
name = "Alice"  # string

# Boolean
is_active = True  # boolean (True/False)

# None
value = None     # represents absence of value
```

### Basic Operators
```python
# Arithmetic
sum = 5 + 3      # 8
diff = 5 - 3     # 2
product = 5 * 3  # 15
quotient = 5 / 3 # 1.666...

# Comparison
is_equal = (5 == 5)  # True
is_greater = (5 > 3) # True

# Logical
and_result = (True and False)  # False
or_result = (True or False)    # True
not_result = not True          # False
```

## 📚 Data Structures

### Lists
```python
# Create a list
fruits = ["apple", "banana", "cherry"]

# Access elements
first_fruit = fruits[0]  # "apple"

# Modify elements
fruits[1] = "blueberry"

# List methods
fruits.append("orange")    # Add to end
fruits.insert(1, "mango")  # Insert at index
fruits.remove("banana")    # Remove first occurrence
```

### Dictionaries
```python
# Create a dictionary
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}

# Access values
name = person["name"]  # "Alice"

# Add/Modify
person["email"] = "alice@example.com"

# Dictionary methods
keys = person.keys()    # Get all keys
values = person.values() # Get all values
```

### Tuples
```python
# Create a tuple
coordinates = (10, 20)

# Access elements
x = coordinates[0]  # 10

# Tuples are immutable
# coordinates[0] = 5  # This would raise an error
```

### Sets
```python
# Create a set
unique_numbers = {1, 2, 3, 3, 4}  # {1, 2, 3, 4}

# Set operations
set1 = {1, 2, 3}
set2 = {3, 4, 5}

union = set1 | set2       # {1, 2, 3, 4, 5}
intersection = set1 & set2 # {3}
```

## 🔄 Control Flow

### If-Else Statements
```python
temperature = 25

if temperature > 30:
    print("It's hot!")
elif temperature > 20:
    print("It's warm.")
else:
    print("It's cool.")
```

### For Loops
```python
# Loop through a list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(f"I like {fruit}")

# Loop with range
for i in range(5):  # 0 to 4
    print(i)
```

### While Loops
```python
count = 0
while count < 5:
    print(count)
    count += 1
```

## 📦 Functions

### Basic Function
```python
def greet(name):
    """This function greets the person passed in as a parameter"""
    return f"Hello, {name}!"

message = greet("Alice")
print(message)  # "Hello, Alice!"
```

### Default Parameters
```python
def power(base, exponent=2):
    return base ** exponent

print(power(3))      # 9 (uses default exponent=2)
print(power(3, 3))   # 27
```

## 📂 File Operations

### Reading Files
```python
# Read entire file
with open('example.txt', 'r') as file:
    content = file.read()
    print(content)

# Read line by line
with open('example.txt', 'r') as file:
    for line in file:
        print(line.strip())  # strip() removes newline characters
```

### Writing Files
```python
# Write to a file
with open('output.txt', 'w') as file:
    file.write("Hello, World!\n")
    file.write("This is a new line.")
```

## 📊 Working with External Libraries

### Installing Packages
```bash
pip install numpy pandas matplotlib
```

### Importing Libraries
```python
# Standard import
import math
print(math.sqrt(16))  # 4.0

# Import with alias
import numpy as np
array = np.array([1, 2, 3])

# Import specific functions
from datetime import datetime
today = datetime.now()

# Import all (not recommended)
# from math import *
```

## 🚀 Next Steps

Now that you've learned the basics of Python, you can explore:

1. Object-Oriented Programming (OOP) in Python
2. Working with external APIs
3. Data analysis with pandas
4. Web development with Flask/Django
5. Machine learning with scikit-learn

Check out the other documentation in this repository for more advanced topics!
