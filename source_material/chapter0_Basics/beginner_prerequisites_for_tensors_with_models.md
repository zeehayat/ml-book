# Beginner Prerequisites for Reading an ML Book that Starts with Tensors

**Audience:** A complete beginner in mathematics who is comfortable only with addition, subtraction, multiplication, division, and percentages.

**Goal:** To prepare you for the prerequisite list you shared, especially for a machine-learning book that begins with **tensors**.

This document does **not** assume algebra, square roots, LCM, calculus, linear algebra, or prior computer science. It builds slowly from ordinary arithmetic and plain language.

---

## How to Use This Document

Read it in order. The order matters.

The book you are about to read likely assumes that you can look at symbols such as:

```text
f: A -> B
x in A
sum over i
product over i
for all n in N
ceil(x)
```

and not panic. It also assumes that you understand that computers store data in memory, and that Python libraries such as NumPy often use fast compiled code underneath.

This guide is designed to make those ideas familiar.

You do **not** need to become a mathematician. You need enough mathematical and computer-science vocabulary to follow the early chapters of a machine-learning text.

---

# Part 1 — The Big Picture

## 1. Why These Prerequisites Matter for Machine Learning

Machine learning programs work with data. Data may be:

- temperatures measured every hour,
- pixels in an image,
- words in a document,
- prices over time,
- features of households,
- measurements from hospitals,
- numbers representing sound,
- numbers representing text.

A machine-learning program usually turns all this into numbers arranged in a regular structure.

That structure is often called a **tensor**.

For now, think of a tensor as:

> A rectangular arrangement of numbers that a computer can store and process efficiently.

Examples:

```text
Single number:
5

List of numbers:
[5, 7, 9]

Table of numbers:
[
  [5, 7, 9],
  [2, 4, 6]
]
```

Machine-learning books use compact notation to talk about these structures. They also expect you to understand how computers store and process these numbers. That is why the prerequisites include both mathematics and computer science.

---

# Part 2 — Essential Arithmetic Ideas

You already know addition, subtraction, multiplication, division, and percentages. We will build from there.

## 2. Numbers as Counts, Measurements, and Codes

Numbers can mean different things depending on context.

### 2.1 Numbers as counts

A count tells how many things there are.

Examples:

```text
3 apples
12 rooms
400 schools
```

Counts are usually whole numbers.

### 2.2 Numbers as measurements

A measurement tells the size, amount, or degree of something.

Examples:

```text
5.5 kilograms
27.3 degrees Celsius
1.75 meters
```

Measurements may include decimals.

### 2.3 Numbers as codes

Sometimes numbers are labels, not quantities.

Examples:

```text
Room 101
District code 7
Category 0 = no, category 1 = yes
```

Here, the number may not be something you add or average. It is being used as a code.

This matters in machine learning because data tables often contain all three kinds of numbers.

---

## 3. Whole Numbers, Negative Numbers, and Decimals

### 3.1 Whole numbers

Whole numbers are numbers like:

```text
0, 1, 2, 3, 4, 5, ...
```

They are used for counting.

### 3.2 Negative numbers

Negative numbers are numbers below zero.

Examples:

```text
-1, -2, -3, -10
```

They are useful when something can go below a reference point.

Examples:

```text
Temperature: -5 degrees
Profit/loss: -100 means a loss of 100
Bank balance: -500 means overdrawn by 500
```

### 3.3 Decimals

Decimals are numbers with a decimal point.

Examples:

```text
1.5
2.75
0.25
13.8
```

In machine learning, decimals are very common because models use measurements, probabilities, and calculated values.

---

## 4. Division and Remainders

You know division. But computer science often uses a special form of division: **integer division**.

Let us first review ordinary division.

### 4.1 Ordinary division

```text
10 divided by 2 = 5
10 divided by 4 = 2.5
```

Ordinary division allows decimals.

### 4.2 Division with a remainder

Before decimals, many people learn division like this:

```text
10 divided by 4 = 2 remainder 2
```

Why?

Because 4 fits into 10 two full times:

```text
4 + 4 = 8
```

Then 2 is left over:

```text
10 - 8 = 2
```

So:

```text
10 = 4 * 2 + 2
```

This is very important in computing.

### 4.3 Integer division

**Integer division** means division where we keep only the whole-number part.

Examples:

```text
10 // 4 = 2
11 // 4 = 2
12 // 4 = 3
13 // 4 = 3
```

In Python, the symbol for integer division is:

```python
//
```

Example:

```python
print(10 // 4)  # 2
print(10 / 4)   # 2.5
```

The difference is:

```text
/  gives ordinary division
// gives integer division
```

### 4.4 Why integer division matters

Computers often divide items into groups or blocks.

Example:

You have 10 items and each box holds 4 items.

How many full boxes can you fill?

```text
10 // 4 = 2 full boxes
```

How many items are left over?

```text
2 items left over
```

This matters for memory, batching, array indexing, and tensors.

---

## 5. The Ceiling Function

The prerequisite list mentions:

```text
ceil(x)
```

or in mathematical notation:

```text
⌈x⌉
```

This is called the **ceiling** of x.

### 5.1 Meaning of ceiling

The ceiling of a number means:

> The smallest whole number that is greater than or equal to the number.

That sounds abstract, so let us make it plain.

If the answer has a decimal and you need a whole number large enough to cover it, you round **up**.

Examples:

```text
ceil(2.1) = 3
ceil(2.5) = 3
ceil(2.9) = 3
ceil(3.0) = 3
ceil(3.1) = 4
```

Notice:

```text
ceil(3.0) is 3, not 4
```

because 3 is already a whole number.

### 5.2 Everyday example

Suppose one taxi can carry 4 people.

You have 10 people.

How many taxis do you need?

Ordinary division:

```text
10 / 4 = 2.5
```

But you cannot hire 2.5 taxis.

You need 3 taxis.

So:

```text
ceil(10 / 4) = 3
```

### 5.3 Python example

Python has a ceiling function in the `math` module.

```python
import math

print(math.ceil(10 / 4))  # 3
print(math.ceil(8 / 4))   # 2
print(math.ceil(9 / 4))   # 3
```

### 5.4 Why ceiling matters in ML and tensors

Machine-learning data is often processed in **batches**.

Suppose you have 103 examples and each batch contains 20 examples.

How many batches are needed?

```text
103 / 20 = 5.15
```

You need 6 batches, because the last batch will contain the remaining examples.

```text
ceil(103 / 20) = 6
```

---

# Part 3 — Mathematical Notation from Zero

Mathematical notation is a compressed language. It looks frightening because it packs a lot into a few symbols.

The goal is not to admire the symbols. The goal is to translate them into ordinary language.

---

## 6. What Is a Set?

A **set** is a collection of things.

The things inside a set are called **elements** or **members**.

Example:

```text
A = {1, 2, 3}
```

Read this as:

```text
A is the set containing 1, 2, and 3.
```

The set is named `A`.

The elements are:

```text
1, 2, 3
```

### 6.1 Curly braces

Sets are often written using curly braces:

```text
{ }
```

Example:

```text
{apple, banana, cherry}
```

This is a set of fruits.

### 6.2 Sets of numbers

A set can contain numbers:

```text
{2, 4, 6, 8}
```

This set contains four numbers.

### 6.3 Order does not matter in a set

In a set, order does not matter.

These are the same set:

```text
{1, 2, 3}
{3, 2, 1}
```

This is different from a Python list, where order matters.

Python list:

```python
[1, 2, 3]
```

is not the same ordering as:

```python
[3, 2, 1]
```

But mathematical sets do not care about order.

### 6.4 Repetition does not matter in a set

This:

```text
{1, 1, 2, 3}
```

is normally treated as:

```text
{1, 2, 3}
```

because a set either contains an element or it does not.

---

## 7. The Membership Symbol: ∈

The symbol:

```text
∈
```

means:

```text
is an element of
```

or more simply:

```text
is in
```

Example:

```text
2 ∈ {1, 2, 3}
```

Read it as:

```text
2 is in the set {1, 2, 3}.
```

This statement is true.

Another example:

```text
5 ∈ {1, 2, 3}
```

Read it as:

```text
5 is in the set {1, 2, 3}.
```

This statement is false.

### 7.1 Not in: ∉

The symbol:

```text
∉
```

means:

```text
is not in
```

Example:

```text
5 ∉ {1, 2, 3}
```

Read it as:

```text
5 is not in the set {1, 2, 3}.
```

This statement is true.

### 7.2 Why ∈ matters in ML books

A machine-learning book may write:

```text
i ∈ {1, 2, ..., n}
```

This means:

```text
i is one of the numbers from 1 to n.
```

If `n = 5`, then:

```text
i ∈ {1, 2, 3, 4, 5}
```

So `i` could be:

```text
1 or 2 or 3 or 4 or 5
```

---

## 8. Natural Numbers: ℕ

The symbol:

```text
ℕ
```

usually means the set of **natural numbers**.

Natural numbers are counting numbers.

Depending on the author, natural numbers may mean either:

```text
1, 2, 3, 4, ...
```

or:

```text
0, 1, 2, 3, 4, ...
```

Different books use different conventions.

If the book is careful, it will say which one it means.

### 8.1 The three dots

The dots:

```text
...
```

mean:

```text
and so on
```

So:

```text
1, 2, 3, 4, ...
```

means:

```text
1, 2, 3, 4, continuing forever
```

### 8.2 Example with ℕ

```text
n ∈ ℕ
```

Read this as:

```text
n is a natural number.
```

In ordinary words:

```text
n is a counting number.
```

In tensor books, `n` often means a size or count.

Example:

```text
n = number of rows
m = number of columns
```

Rows and columns are counts, so they are natural numbers.

---

## 9. Integers: ℤ

The symbol:

```text
ℤ
```

means the set of **integers**.

Integers are whole numbers including negative numbers, zero, and positive numbers.

```text
..., -3, -2, -1, 0, 1, 2, 3, ...
```

### 9.1 Example with ℤ

```text
k ∈ ℤ
```

Read this as:

```text
k is an integer.
```

That means `k` could be:

```text
-10, -3, -1, 0, 1, 2, 100
```

but not:

```text
2.5
3.7
```

because those are decimals.

### 9.2 Why integers matter in computing

Computer indexes are usually integers.

Example:

```python
items = [10, 20, 30]
```

The positions are:

```text
0, 1, 2
```

These are integers.

You cannot ask for position `1.5` in a list.

```python
items[1.5]  # error
```

---

## 10. The Universal Symbol: ∀

The symbol:

```text
∀
```

means:

```text
for all
```

or:

```text
for every
```

Example:

```text
∀ x ∈ A
```

Read this as:

```text
for every x in A
```

Suppose:

```text
A = {1, 2, 3}
```

Then:

```text
∀ x ∈ A
```

means:

```text
for x = 1, for x = 2, and for x = 3
```

### 10.1 Example statement

```text
∀ x ∈ {1, 2, 3}, x is less than 5
```

Read this as:

```text
For every x in the set {1, 2, 3}, x is less than 5.
```

This is true because:

```text
1 is less than 5
2 is less than 5
3 is less than 5
```

### 10.2 A false example

```text
∀ x ∈ {1, 2, 8}, x is less than 5
```

This is false because:

```text
8 is not less than 5
```

To disprove a “for all” statement, you only need one counterexample.

### 10.3 Python connection

This mathematical idea is similar to a Python loop:

```python
for x in [1, 2, 3]:
    print(x)
```

The loop says:

```text
Do something for every x in the list.
```

---

## 11. The Sum Symbol: Σ

The symbol:

```text
Σ
```

is the Greek capital letter sigma.

In mathematics it usually means:

```text
sum
```

or:

```text
add up
```

### 11.1 Plain example

Suppose you have:

```text
2 + 4 + 6 + 8
```

The sum is:

```text
20
```

Mathematics sometimes writes repeated addition compactly using Σ.

### 11.2 Index notation

You may see:

```text
Σ from i = 1 to 4 of x_i
```

Often written as:

```text
∑_{i=1}^{4} x_i
```

Read it slowly:

```text
Add x_i for i starting at 1 and ending at 4.
```

That means:

```text
x_1 + x_2 + x_3 + x_4
```

### 11.3 What does x_i mean?

The small `i` below the x is an index.

```text
x_i
```

means:

```text
the item of x at position i
```

Suppose:

```text
x_1 = 10
x_2 = 20
x_3 = 30
x_4 = 40
```

Then:

```text
∑_{i=1}^{4} x_i = 10 + 20 + 30 + 40 = 100
```

### 11.4 Python connection

In Python:

```python
x = [10, 20, 30, 40]
print(sum(x))  # 100
```

Python uses 0-based indexing, so:

```python
x[0] = 10
x[1] = 20
x[2] = 30
x[3] = 40
```

Mathematical books often use 1-based indexing:

```text
x_1, x_2, x_3, x_4
```

Python lists use 0-based indexing:

```text
x[0], x[1], x[2], x[3]
```

This difference is important.

### 11.5 Why Σ matters in ML

Machine learning uses many sums.

Example:

- add all errors,
- add all probabilities,
- add all values in a row,
- add all numbers in a tensor,
- compute an average.

An average is:

```text
sum of values / number of values
```

Example:

```text
Values: 10, 20, 30
Sum: 10 + 20 + 30 = 60
Number of values: 3
Average: 60 / 3 = 20
```

---

## 12. The Product Symbol: Π

The symbol:

```text
Π
```

is the Greek capital letter pi.

In mathematical notation it often means:

```text
product
```

or:

```text
multiply together
```

### 12.1 Plain example

Suppose you have:

```text
2 * 3 * 4
```

The product is:

```text
24
```

### 12.2 Product notation

You may see:

```text
∏_{i=1}^{3} x_i
```

Read it as:

```text
Multiply x_i for i starting at 1 and ending at 3.
```

That means:

```text
x_1 * x_2 * x_3
```

Suppose:

```text
x_1 = 2
x_2 = 3
x_3 = 4
```

Then:

```text
∏_{i=1}^{3} x_i = 2 * 3 * 4 = 24
```

### 12.3 Why Π matters for tensors

Tensor shapes are often written as several numbers.

Example:

```text
shape = (2, 3, 4)
```

This means the tensor has:

```text
2 groups,
3 rows in each group,
4 values in each row
```

The total number of values is:

```text
2 * 3 * 4 = 24
```

A book may write this as:

```text
∏ dimensions
```

meaning:

```text
multiply all the dimension sizes together
```

---

## 13. Functions from One Set to Another: f: A -> B

The prerequisite list says you should understand:

```text
f: A -> B
```

This is function notation.

### 13.1 What is a function?

A **function** is a rule that takes an input and gives an output.

Plain examples:

```text
Input: number of apples
Function: multiply by price per apple
Output: total price
```

If each apple costs 10 rupees:

```text
input 1 -> output 10
input 2 -> output 20
input 3 -> output 30
```

The function is:

```text
multiply by 10
```

### 13.2 Function as a machine

Think of a function as a machine:

```text
input goes in -> rule is applied -> output comes out
```

Example:

```text
Input: 5
Rule: multiply by 2
Output: 10
```

### 13.3 What does f: A -> B mean?

```text
f: A -> B
```

Read it as:

```text
f is a function from A to B.
```

This means:

```text
The inputs of f come from set A.
The outputs of f are in set B.
```

Example:

```text
A = {1, 2, 3}
B = {10, 20, 30}
```

Define function `f` as:

```text
f(x) = 10 times x
```

Then:

```text
f(1) = 10
f(2) = 20
f(3) = 30
```

So:

```text
f: A -> B
```

### 13.4 What does f(x) mean?

```text
f(x)
```

means:

```text
the output of function f when the input is x
```

Example:

If:

```text
f means multiply by 10
```

then:

```text
f(3) = 30
```

### 13.5 Python function example

```python
def f(x):
    return 10 * x

print(f(1))  # 10
print(f(2))  # 20
print(f(3))  # 30
```

This is the same idea.

### 13.6 Why functions matter in ML

A machine-learning model is often a function.

It takes input data and produces output.

Example:

```text
Input: data about a house
Output: predicted price
```

or:

```text
Input: image pixels
Output: probability that the image contains a cat
```

In tensor language:

```text
model: input tensor -> output tensor
```

---

# Part 4 — Python Foundations Needed for the Book

This section assumes you know some Python but want the relevant concepts explained carefully.

---

## 14. Python Values, Variables, and Names

In ordinary conversation, we say:

```python
x = 5
```

and then say:

```text
x is a variable containing 5.
```

That is acceptable at beginner level.

But for understanding memory, we should be slightly more precise.

In Python:

```text
x is a name that refers to an object.
```

The object is the value `5`.

So:

```python
x = 5
```

means:

```text
Bind the name x to the integer object 5.
```

You do not need to speak this way every day, but it helps explain why Python behaves as it does.

### 14.1 Assignment

Assignment uses `=`.

```python
x = 5
```

This does not mean “x equals 5 forever.”

It means:

```text
From now on, the name x refers to 5, unless we reassign it.
```

Example:

```python
x = 5
print(x)  # 5

x = 10
print(x)  # 10
```

The name `x` first referred to 5. Then it referred to 10.

---

## 15. Python Lists

A Python list is an ordered collection of items.

Example:

```python
numbers = [10, 20, 30]
```

This list has three items.

### 15.1 Indexing

Python indexes start at 0.

```python
numbers = [10, 20, 30]

print(numbers[0])  # 10
print(numbers[1])  # 20
print(numbers[2])  # 30
```

The positions are:

```text
position 0 -> 10
position 1 -> 20
position 2 -> 30
```

### 15.2 Changing a list

Lists are mutable.

Mutable means:

```text
can be changed after creation
```

Example:

```python
numbers = [10, 20, 30]
numbers[1] = 99
print(numbers)  # [10, 99, 30]
```

### 15.3 List length

Use `len()` to get the number of items.

```python
numbers = [10, 20, 30]
print(len(numbers))  # 3
```

### 15.4 Lists can hold mixed objects

Python lists can hold different kinds of values:

```python
items = [10, "hello", 3.5]
```

This flexibility is convenient but not always efficient.

For numerical machine learning, arrays/tensors usually store many values of the same type.

---

## 16. Python Tuples

A tuple is also an ordered collection.

Example:

```python
point = (10, 20)
```

The main difference is that tuples are immutable.

Immutable means:

```text
cannot be changed after creation
```

Example:

```python
point = (10, 20)
print(point[0])  # 10
print(point[1])  # 20
```

But this will fail:

```python
point[0] = 99  # error
```

### 16.1 Why tuples matter for tensors

Tensor shapes are often written as tuples.

Example:

```python
shape = (2, 3)
```

This may mean:

```text
2 rows and 3 columns
```

Another example:

```python
shape = (4, 28, 28)
```

This may mean:

```text
4 images, each with 28 rows and 28 columns
```

A shape should not casually change, so a tuple is a natural way to represent it.

---

## 17. Generator Functions

A generator function is a function that produces values one at a time instead of building a full list immediately.

### 17.1 Normal function returning a list

```python
def make_numbers():
    return [1, 2, 3]

numbers = make_numbers()
print(numbers)  # [1, 2, 3]
```

This creates the full list before returning it.

### 17.2 Generator using yield

```python
def generate_numbers():
    yield 1
    yield 2
    yield 3

for number in generate_numbers():
    print(number)
```

Output:

```text
1
2
3
```

The keyword `yield` means:

```text
produce this value now, but remember where you stopped so you can continue later
```

### 17.3 Why generators matter

Suppose you have a very large dataset.

A normal function might try to load everything into memory at once.

A generator can produce one item or one batch at a time.

This is useful in machine learning because datasets can be very large.

### 17.4 Generator example with batches

```python
def batches(data, batch_size):
    start = 0
    while start < len(data):
        end = start + batch_size
        yield data[start:end]
        start = end

numbers = [1, 2, 3, 4, 5, 6, 7]

for batch in batches(numbers, 3):
    print(batch)
```

Output:

```text
[1, 2, 3]
[4, 5, 6]
[7]
```

The last batch has only one item because the data size is not exactly divisible by the batch size.

This connects to the ceiling function:

```text
ceil(7 / 3) = 3 batches
```

---

## 18. Dataclasses

A dataclass is a convenient way to create a simple class mostly used to store data.

### 18.1 Why classes exist

Suppose you want to store information about a student.

Without a class:

```python
student_name = "Amina"
student_age = 20
student_score = 85
```

This works, but if you have many students, it becomes messy.

A class lets you group related information.

### 18.2 Dataclass example

```python
from dataclasses import dataclass

@dataclass
class Student:
    name: str
    age: int
    score: float

student = Student(name="Amina", age=20, score=85.0)

print(student.name)   # Amina
print(student.age)    # 20
print(student.score)  # 85.0
```

### 18.3 What @dataclass does

The line:

```python
@dataclass
```

asks Python to automatically create useful methods for the class.

For example, it creates an initializer so that this works:

```python
Student(name="Amina", age=20, score=85.0)
```

Without dataclass, you would have to write more code yourself.

### 18.4 Why dataclasses matter for ML books

A book may use dataclasses to store configuration.

Example:

```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
```

Then:

```python
config = TrainingConfig(
    batch_size=32,
    learning_rate=0.001,
    num_epochs=10,
)
```

This groups training settings into one object.

---

## 19. __getitem__ and __setitem__

These names look strange because they have double underscores.

```python
__getitem__
__setitem__
```

Python uses double-underscore methods for special behavior.

They are sometimes called “dunder” methods because double underscore sounds like “dunder.”

### 19.1 __getitem__

`__getitem__` controls what happens when you use square brackets to get an item.

Example:

```python
x[0]
```

When Python sees this, it internally calls something like:

```python
x.__getitem__(0)
```

### 19.2 Simple class with __getitem__

```python
class MyListLikeThing:
    def __init__(self):
        self.data = [10, 20, 30]

    def __getitem__(self, index):
        return self.data[index]

obj = MyListLikeThing()

print(obj[0])  # 10
print(obj[1])  # 20
```

The object behaves like something you can index.

### 19.3 __setitem__

`__setitem__` controls what happens when you assign using square brackets.

Example:

```python
x[0] = 99
```

Python internally calls something like:

```python
x.__setitem__(0, 99)
```

### 19.4 Simple class with __setitem__

```python
class MyListLikeThing:
    def __init__(self):
        self.data = [10, 20, 30]

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

obj = MyListLikeThing()

print(obj[1])  # 20
obj[1] = 99
print(obj[1])  # 99
```

### 19.5 Why this matters for tensors

Tensor libraries use square brackets heavily.

Example:

```python
x[0]
x[0, 1]
x[:, 2]
```

These indexing operations are made possible by methods like `__getitem__` and `__setitem__`.

You do not need to write these methods at first. But you should understand that bracket notation is not magic. Python objects can define how bracket access works.

---

## 20. Basic Type Hinting with typing

Python does not require type hints, but they help readers and tools understand what kind of values are expected.

### 20.1 Function without type hints

```python
def add(a, b):
    return a + b
```

This works, but we do not know what `a` and `b` are supposed to be.

Are they integers?

Decimals?

Strings?

Lists?

### 20.2 Function with type hints

```python
def add(a: int, b: int) -> int:
    return a + b
```

Read this as:

```text
a should be an integer.
b should be an integer.
The function returns an integer.
```

The arrow:

```python
-> int
```

means:

```text
returns an integer
```

### 20.3 Type hints do not automatically enforce types

This is important.

Python type hints are mostly for humans and tools.

Example:

```python
def add(a: int, b: int) -> int:
    return a + b

print(add(2, 3))        # 5
print(add("a", "b"))  # ab
```

Python may still run the second example, even though the type hints said `int`.

Type hints are guidance, not always strict runtime rules.

### 20.4 Common type hints

```python
x: int = 5
name: str = "Amina"
price: float = 10.5
active: bool = True
```

Meanings:

```text
int   whole number
str   text string
float decimal number
bool  True or False
```

### 20.5 Lists in type hints

Modern Python:

```python
numbers: list[int] = [1, 2, 3]
```

Older style using `typing`:

```python
from typing import List

numbers: List[int] = [1, 2, 3]
```

Both mean:

```text
numbers is a list of integers
```

### 20.6 Tuples in type hints

```python
shape: tuple[int, int] = (2, 3)
```

Means:

```text
shape is a tuple containing two integers
```

For a tensor shape, you may also see:

```python
shape: tuple[int, ...]
```

The `...` means:

```text
any number of integer entries
```

So it could be:

```python
(3,)
(2, 3)
(4, 28, 28)
```

### 20.7 Why type hints matter for ML books

Machine-learning code often has many objects:

- datasets,
- batches,
- tensors,
- models,
- training configurations,
- optimizers.

Type hints help the reader know what kind of object is being passed around.

---

# Part 5 — Computer Memory from the Beginning

The prerequisite list says:

> You understand that computer memory is a large, flat array of addressable bytes. Every variable your program uses — an integer, a float, a list — occupies a contiguous range of bytes at a specific address.

This idea is mostly true at a low level, but Python adds some important details. We will build the idea carefully.

---

## 21. What Is Computer Memory?

Computer memory is where a running program keeps information it needs right now.

When a program runs, it needs somewhere to store:

- numbers,
- text,
- lists,
- images,
- temporary results,
- instructions,
- objects.

That place is memory.

### 21.1 Memory as a long row of boxes

Imagine computer memory as a very long row of small boxes.

Each box can hold a tiny piece of information.

```text
+----+----+----+----+----+----+----+----+
|    |    |    |    |    |    |    |    |
+----+----+----+----+----+----+----+----+
  0    1    2    3    4    5    6    7
```

Each box has an address.

The addresses above are:

```text
0, 1, 2, 3, 4, 5, 6, 7
```

Real memory has billions of such boxes.

### 21.2 What is a byte?

A byte is a small unit of computer memory.

A modern computer usually treats memory as addressable bytes.

That means:

```text
Each address points to one byte.
```

You do not need to understand binary deeply yet. For this book, it is enough to know:

```text
A byte is a small storage unit.
Many bytes together can store larger things.
```

### 21.3 Addressable bytes

“Addressable” means each byte has an address, like a house number.

If memory is like a long street, each byte is like a house on that street.

```text
address 1000 -> one byte
address 1001 -> next byte
address 1002 -> next byte
```

Programs use addresses to find data.

---

## 22. Contiguous Memory

Contiguous means:

```text
next to each other, without gaps
```

If a value occupies addresses:

```text
1000, 1001, 1002, 1003
```

then it occupies 4 contiguous bytes.

They are side by side.

### 22.1 Example

Suppose a number needs 4 bytes.

It might be stored like this:

```text
Address: 1000  1001  1002  1003
Byte:      ?     ?     ?     ?
```

The number occupies a contiguous range of 4 bytes.

### 22.2 Why contiguous memory matters

Computers are fast when data is stored close together.

If a tensor stores many numbers in a contiguous block, the computer can process them efficiently.

This is one reason NumPy arrays and tensors are faster than ordinary Python lists for numerical work.

---

## 23. Python Variables and Memory: The Important Nuance

At a low level, every object exists somewhere in memory.

But in Python, a variable name is not exactly the same as a raw memory slot.

Example:

```python
x = 5
```

In Python, `x` is a name referring to an object representing the integer 5.

The object lives in memory. The name points to it.

### 23.1 Names and objects

Think of it like this:

```text
x  --->  integer object 5
```

If you write:

```python
y = x
```

then both names refer to the same value object:

```text
x  ----\
        ---> integer object 5
y  ----/
```

For immutable objects like integers, this usually does not create problems.

### 23.2 Lists contain references

A Python list does not necessarily store all raw values directly side by side.

A Python list stores references to objects.

Example:

```python
items = [10, 20, 30]
```

Conceptually:

```text
items list object ---> [ reference, reference, reference ]
                         |          |          |
                         v          v          v
                         10         20         30
```

This means a Python list is flexible, but it has overhead.

### 23.3 NumPy arrays are different

A NumPy array of numbers is closer to the raw contiguous-memory idea.

Example:

```python
import numpy as np

arr = np.array([10, 20, 30], dtype=np.int32)
```

This stores the numbers in a more compact block of memory, all with the same data type.

Conceptually:

```text
arr data buffer ---> 10 | 20 | 30
```

The values are stored in a regular format, making fast numerical operations possible.

### 23.4 Why the nuance matters

When the prerequisite says:

> every variable occupies a contiguous range of bytes

for a low-level language like C, this is a useful simplification.

For Python, a more accurate statement is:

> Python names refer to objects. Objects live in memory. Numerical arrays and tensors often store their raw data in contiguous memory blocks.

This distinction will help you understand why Python lists and tensor arrays behave differently.

---

## 24. Data Types and Sizes

A computer must know how to interpret bytes.

The same bytes could mean different things depending on the data type.

A data type tells the computer:

```text
what kind of value this is and how to interpret its bytes
```

Common numerical data types:

```text
integer       whole number
float         decimal number
boolean       True or False
```

### 24.1 Integer

An integer is a whole number.

Examples:

```text
-3, -2, -1, 0, 1, 2, 3
```

### 24.2 Float

A float is a decimal number.

Examples:

```text
1.5
2.75
0.001
```

Machine-learning libraries use floats very often.

### 24.3 Boolean

A boolean has one of two values:

```text
True
False
```

Example:

```python
is_active = True
is_finished = False
```

### 24.4 dtype

In NumPy and tensor libraries, you often see `dtype`.

`dtype` means:

```text
data type
```

Example:

```python
import numpy as np

x = np.array([1, 2, 3], dtype=np.int32)
y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
```

This tells the library how to store and interpret the numbers.

### 24.5 Why same dtype matters

An ordinary Python list can contain mixed types:

```python
items = [1, 2.5, "hello"]
```

A numerical tensor usually contains one kind of number.

Example:

```text
all float32
or all int64
```

This regularity helps the computer process data quickly.

---

## 25. Memory Layout and Indexing

Suppose we have this table:

```text
[
  [10, 20, 30],
  [40, 50, 60]
]
```

It has:

```text
2 rows
3 columns
```

Shape:

```text
(2, 3)
```

But computer memory is flat: a long row of bytes.

So how can a 2D table fit into 1D memory?

The answer: the computer lays it out in a flat sequence.

One common layout is row-major order:

```text
10, 20, 30, 40, 50, 60
```

The first row comes first, then the second row.

### 25.1 From 2D position to flat position

For a table with 3 columns:

```text
flat position = row * number_of_columns + column
```

Use Python-style indexing, where rows and columns start at 0.

Example table:

```text
row 0: 10, 20, 30
row 1: 40, 50, 60
```

The value at row 1, column 2 is 60.

Flat position:

```text
row * number_of_columns + column
= 1 * 3 + 2
= 5
```

Flat sequence:

```text
position 0 -> 10
position 1 -> 20
position 2 -> 30
position 3 -> 40
position 4 -> 50
position 5 -> 60
```

So position 5 contains 60.

### 25.2 Why this matters

Tensors may look multi-dimensional, but the raw memory underneath is often a flat block.

The tensor object remembers:

```text
shape
strides
 dtype
location of data
```

You do not need to master strides yet, but you should know:

```text
multi-dimensional indexing is mapped to flat memory
```

---

# Part 6 — C Extensions and Why NumPy Is Fast

The prerequisite list says:

> You understand what a C extension is at a conceptual level: a Python library whose performance-critical routines are compiled to native machine code and called from Python. You do not need to write C, but you should know that when you call numpy.sum(), the actual loop over bytes is running compiled C, not Python.

Let us unpack this slowly.

---

## 26. Python Is Convenient but Not Always Fast

Python is easy to read and write.

Example:

```python
total = 0
for x in [1, 2, 3, 4]:
    total += x

print(total)
```

This is clear.

But Python loops have overhead. For every item, Python has to manage objects, types, and interpreter steps.

For small lists this does not matter.

For millions of numbers, it matters a lot.

---

## 27. What Is C?

C is an older programming language that is closer to the machine than Python.

C code is usually compiled into machine code.

Machine code is the kind of instruction the processor can run directly.

You do not need to write C for this book. But you should know this chain:

```text
C source code -> compiler -> machine code -> runs fast on processor
```

Python usually works differently:

```text
Python code -> Python interpreter -> actions performed step by step
```

That makes Python flexible, but often slower for tight loops over huge data.

---

## 28. What Is a C Extension?

A C extension is a piece of compiled C code that Python can call.

From the user's point of view, it feels like calling a normal Python function.

Example:

```python
import numpy as np

x = np.array([1, 2, 3, 4])
print(np.sum(x))
```

You write:

```python
np.sum(x)
```

But the heavy work is done by compiled code underneath.

### 28.1 The key idea

You use Python as the convenient control language.

The library uses C or another compiled language for the heavy numerical work.

So the pattern is:

```text
Python for expression and organization
Compiled code for speed
```

---

## 29. Why numpy.sum Is Faster Than a Python Loop

Suppose you want to add one million numbers.

Python loop:

```python
total = 0
for x in numbers:
    total += x
```

This loop is controlled by Python.

NumPy:

```python
total = np.sum(array)
```

The loop over the raw bytes is handled by optimized compiled code.

### 29.1 Plain analogy

Imagine you need to move 10,000 bricks.

Pure Python loop is like giving one instruction at a time:

```text
Pick brick 1.
Move brick 1.
Pick brick 2.
Move brick 2.
...
```

Compiled NumPy code is like giving one instruction to a trained crew:

```text
Move all these bricks using your efficient method.
```

You still requested the work from Python, but the detailed repeated work happens elsewhere.

---

# Part 7 — From Lists to Arrays to Tensors

Now we connect mathematics, Python, and memory to tensors.

---

## 30. Scalar, Vector, Matrix, Tensor

These words sound more advanced than they are.

### 30.1 Scalar

A scalar is a single number.

Example:

```text
5
```

In Python:

```python
x = 5
```

### 30.2 Vector

A vector is a one-dimensional list of numbers.

Example:

```text
[10, 20, 30]
```

It has length 3.

Shape:

```text
(3,)
```

The comma is used because it is a tuple with one entry.

### 30.3 Matrix

A matrix is a two-dimensional table of numbers.

Example:

```text
[
  [10, 20, 30],
  [40, 50, 60]
]
```

It has:

```text
2 rows
3 columns
```

Shape:

```text
(2, 3)
```

### 30.4 Tensor

A tensor is a general word for these number arrangements.

In machine learning, a tensor may be:

```text
0-dimensional: a single number
1-dimensional: a vector
2-dimensional: a matrix
3-dimensional: a stack of matrices
4-dimensional or more: larger structured numerical data
```

Do not worry about the deeper mathematical meaning of tensor at this stage. In most introductory ML code, tensor means:

```text
a multi-dimensional array of numbers
```

---

## 31. Shape

The shape of a tensor tells you how many items exist along each dimension.

### 31.1 Shape of a vector

```text
[10, 20, 30]
```

There are 3 values.

Shape:

```text
(3,)
```

### 31.2 Shape of a matrix

```text
[
  [10, 20, 30],
  [40, 50, 60]
]
```

There are:

```text
2 rows
3 columns
```

Shape:

```text
(2, 3)
```

### 31.3 Shape of a 3D tensor

Imagine 2 tables, each with 3 rows and 4 columns.

Shape:

```text
(2, 3, 4)
```

Total number of values:

```text
2 * 3 * 4 = 24
```

This is where product notation is useful.

---

## 32. Dimension, Axis, and Rank

Different books use these words slightly differently, but in practical ML code:

### 32.1 Dimension or axis

An axis is one direction of a tensor.

For a matrix:

```text
axis 0 -> rows
axis 1 -> columns
```

Example matrix shape:

```text
(2, 3)
```

This means:

```text
axis 0 has size 2
axis 1 has size 3
```

### 32.2 Rank or order

The rank/order of a tensor often means the number of axes.

Examples:

```text
scalar: shape (), rank 0
vector: shape (3,), rank 1
matrix: shape (2, 3), rank 2
3D tensor: shape (2, 3, 4), rank 3
```

Some books use “rank” differently in linear algebra. In tensor libraries, “rank” often means number of dimensions. Always check the author’s convention.

---

## 33. Indexing Tensors

Indexing means selecting a value by position.

### 33.1 Vector indexing

```python
x = [10, 20, 30]

print(x[0])  # 10
print(x[1])  # 20
print(x[2])  # 30
```

### 33.2 Matrix indexing

In NumPy:

```python
import numpy as np

x = np.array([
    [10, 20, 30],
    [40, 50, 60],
])

print(x[0, 0])  # 10
print(x[0, 1])  # 20
print(x[1, 0])  # 40
print(x[1, 2])  # 60
```

Read:

```text
x[row, column]
```

### 33.3 3D indexing

For a tensor with shape:

```text
(2, 3, 4)
```

an index may look like:

```python
x[1, 2, 3]
```

Read it as:

```text
from group 1, row 2, column 3
```

Remember Python indexing starts at 0.

So if an axis has size 4, valid indexes are:

```text
0, 1, 2, 3
```

Not 4.

---

## 34. Slicing

Slicing means selecting a range of items.

### 34.1 List slicing

```python
x = [10, 20, 30, 40, 50]

print(x[1:4])  # [20, 30, 40]
```

Read:

```text
start at index 1, stop before index 4
```

Important:

```text
Python includes the start index but excludes the stop index.
```

So:

```python
x[1:4]
```

includes:

```text
index 1, index 2, index 3
```

but not index 4.

### 34.2 Colon alone

A colon alone means:

```text
take everything along this axis
```

Example:

```python
x[:, 0]
```

For a matrix, this means:

```text
take all rows, column 0
```

### 34.3 Example

```python
import numpy as np

x = np.array([
    [10, 20, 30],
    [40, 50, 60],
])

print(x[:, 0])  # [10 40]
print(x[0, :])  # [10 20 30]
```

Read:

```text
x[:, 0] -> all rows, first column
x[0, :] -> first row, all columns
```

---

## 35. Reshaping

Reshaping means changing the shape of data without changing the actual values.

Example:

```text
[1, 2, 3, 4, 5, 6]
```

This has shape:

```text
(6,)
```

It can be reshaped into:

```text
[
  [1, 2, 3],
  [4, 5, 6]
]
```

Shape:

```text
(2, 3)
```

Why is this allowed?

Because:

```text
2 * 3 = 6
```

The number of values stays the same.

### 35.1 Not every reshape is possible

Can 6 values be reshaped into shape `(4, 2)`?

```text
4 * 2 = 8
```

No, because you would need 8 values.

Can 6 values be reshaped into `(3, 2)`?

```text
3 * 2 = 6
```

Yes.

### 35.2 NumPy example

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6])
y = x.reshape(2, 3)

print(y)
```

Output:

```text
[[1 2 3]
 [4 5 6]]
```

---

## 36. Broadcasting: A First Intuition

Broadcasting is a rule that lets tensor libraries combine arrays of different shapes when the operation makes sense.

You do not need to master it immediately, but it appears early in many books.

### 36.1 Simple example

```python
import numpy as np

x = np.array([10, 20, 30])
y = x + 5

print(y)  # [15 25 35]
```

Here, `5` is a single number. NumPy adds it to every element.

Conceptually:

```text
[10, 20, 30] + 5
```

becomes:

```text
[10 + 5, 20 + 5, 30 + 5]
```

Result:

```text
[15, 25, 35]
```

### 36.2 Matrix plus vector

```python
import numpy as np

x = np.array([
    [10, 20, 30],
    [40, 50, 60],
])

y = np.array([1, 2, 3])

print(x + y)
```

Result:

```text
[[11 22 33]
 [41 52 63]]
```

The vector `[1, 2, 3]` is added to each row.

Broadcasting saves us from writing repeated loops.

---

# Part 8 — How the Prerequisite List Looks After Translation

Here is the original prerequisite list rewritten in beginner language.

---

## 37. Computer Science Fundamentals — Translated

### Original idea 1

> You understand that computer memory is a large, flat array of addressable bytes.

Beginner translation:

```text
A computer stores running data in memory.
Memory is like a very long row of tiny boxes.
Each box has an address, like a house number.
Each box stores one byte.
```

### Original idea 2

> Every variable your program uses — an integer, a float, a list — occupies a contiguous range of bytes at a specific address.

Beginner translation with Python nuance:

```text
Every value used by a program must be stored somewhere in memory.
In low-level languages, values often sit directly in memory locations.
In Python, names refer to objects, and those objects live in memory.
Numerical arrays and tensors often store their raw values in contiguous blocks of memory.
```

### Original idea 3

> You can read and write Python with comfort. You are familiar with Python's list, tuple, generator functions, dataclass, __getitem__, __setitem__, and basic type hinting with typing.

Beginner translation:

```text
You can use Python lists and tuples.
You understand that generators produce values one at a time.
You know dataclasses are simple ways to group data.
You understand that __getitem__ controls x[0] behavior.
You understand that __setitem__ controls x[0] = value behavior.
You can read simple type hints such as x: int or def f(x: int) -> int.
```

### Original idea 4

> You understand what a C extension is at a conceptual level.

Beginner translation:

```text
Some Python libraries are partly written in faster compiled languages such as C.
You call them from Python, but the heavy loops run in compiled machine code.
That is why NumPy and tensor libraries can be fast.
```

---

## 38. Mathematics — Translated

### Original idea 1

> You can read and interpret basic set notation: ∈, ∏, ∑, ∀, ℕ, ℤ.

Beginner translation:

```text
You can translate these symbols:

∈  means "is in"
∑  means "add these things"
∏  means "multiply these things"
∀  means "for every"
ℕ  means natural/counting numbers
ℤ  means integers: negative whole numbers, zero, and positive whole numbers
```

### Original idea 2

> You are comfortable with integer division and the ceiling function.

Beginner translation:

```text
Integer division keeps the whole-number part of division.
Ceiling means round up to the next whole number when needed.
```

Example:

```text
10 // 4 = 2
ceil(10 / 4) = 3
```

### Original idea 3

> You understand the concept of a function from one set to another: f: A -> B.

Beginner translation:

```text
A function takes inputs from one set and gives outputs in another set.
f: A -> B means function f takes inputs from A and produces outputs in B.
```

---

# Part 9 — Practice Exercises

Do these slowly. The goal is not speed. The goal is comfort.

---

## 39. Set Notation Practice

### Exercise 1

Let:

```text
A = {2, 4, 6, 8}
```

Are these true or false?

```text
2 ∈ A
5 ∈ A
8 ∈ A
10 ∉ A
```

### Solution

```text
2 ∈ A      true, because 2 is in A
5 ∈ A      false, because 5 is not in A
8 ∈ A      true, because 8 is in A
10 ∉ A     true, because 10 is not in A
```

---

## 40. Natural Numbers and Integers Practice

### Exercise 2

Which of these are integers?

```text
-3
0
2
2.5
10
```

### Solution

Integers are whole numbers, including negative whole numbers and zero.

```text
-3   integer
0    integer
2    integer
2.5  not an integer
10   integer
```

---

## 41. Sum Symbol Practice

### Exercise 3

Suppose:

```text
x_1 = 5
x_2 = 10
x_3 = 15
```

Find:

```text
∑_{i=1}^{3} x_i
```

### Solution

This means:

```text
x_1 + x_2 + x_3
```

So:

```text
5 + 10 + 15 = 30
```

Answer:

```text
30
```

---

## 42. Product Symbol Practice

### Exercise 4

Suppose:

```text
x_1 = 2
x_2 = 3
x_3 = 5
```

Find:

```text
∏_{i=1}^{3} x_i
```

### Solution

This means:

```text
x_1 * x_2 * x_3
```

So:

```text
2 * 3 * 5 = 30
```

Answer:

```text
30
```

---

## 43. Ceiling Practice

### Exercise 5

A van carries 6 people. You have 25 people.

How many vans are needed?

### Solution

Ordinary division:

```text
25 / 6 = 4.166...
```

Four vans carry:

```text
4 * 6 = 24 people
```

One person remains.

So you need 5 vans.

```text
ceil(25 / 6) = 5
```

---

## 44. Function Practice

### Exercise 6

Let:

```text
A = {1, 2, 3}
B = {2, 4, 6}
```

Function `f` doubles the input.

Find:

```text
f(1)
f(2)
f(3)
```

### Solution

```text
f(1) = 2
f(2) = 4
f(3) = 6
```

So `f` maps inputs from `A` to outputs in `B`.

```text
f: A -> B
```

---

## 45. Python List Practice

### Exercise 7

Given:

```python
numbers = [10, 20, 30, 40]
```

What are:

```python
numbers[0]
numbers[2]
len(numbers)
```

### Solution

Python indexing starts at 0.

```python
numbers[0]  # 10
numbers[2]  # 30
len(numbers)  # 4
```

---

## 46. Shape Practice

### Exercise 8

What is the shape of this table?

```text
[
  [1, 2, 3],
  [4, 5, 6]
]
```

### Solution

There are:

```text
2 rows
3 columns
```

Shape:

```text
(2, 3)
```

Total values:

```text
2 * 3 = 6
```

---

## 47. Reshape Practice

### Exercise 9

Can 12 values be reshaped into shape `(3, 4)`?

### Solution

Multiply the shape numbers:

```text
3 * 4 = 12
```

Yes, 12 values can be reshaped into `(3, 4)`.

### Exercise 10

Can 12 values be reshaped into shape `(5, 3)`?

### Solution

```text
5 * 3 = 15
```

No. Shape `(5, 3)` needs 15 values, not 12.

---

# Part 10 — A Gentle First Reading of Tensor Notation

Here are examples of statements you may see in an ML book, translated into plain language.

---

## 48. Example: x ∈ ℝⁿ

You may see:

```text
x ∈ R^n
```

This uses a symbol not in your prerequisite list, but it is common.

`R` usually means real numbers, which include ordinary decimals.

`R^n` usually means:

```text
all vectors of length n containing real numbers
```

Beginner translation:

```text
x is a list of n decimal-type numbers.
```

If:

```text
n = 3
```

then `x` might be:

```text
[1.5, 2.0, 7.25]
```

Do not worry about the deeper mathematics yet.

---

## 49. Example: X ∈ ℝ^{m×n}

You may see:

```text
X ∈ R^{m x n}
```

Beginner translation:

```text
X is a table with m rows and n columns, containing decimal-type numbers.
```

If:

```text
m = 2
n = 3
```

then `X` could be:

```text
[
  [1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0]
]
```

Shape:

```text
(2, 3)
```

---

## 50. Example: Tensor T with shape (b, h, w, c)

In image machine learning, you may see:

```text
T has shape (b, h, w, c)
```

This often means:

```text
b = batch size
h = height
w = width
c = channels
```

For example:

```text
(32, 224, 224, 3)
```

may mean:

```text
32 images
224 rows of pixels per image
224 columns of pixels per image
3 color channels per pixel
```

Total numbers:

```text
32 * 224 * 224 * 3
```

You do not need to calculate this immediately. The important idea is:

```text
multiply the dimensions to know how many stored values there are
```

---

# Part 11 — Minimum Python You Should Be Able to Read

Before starting the ML book, make sure you can read the following examples.

---

## 51. Lists and Loops

```python
numbers = [10, 20, 30]

total = 0
for number in numbers:
    total = total + number

print(total)
```

Plain meaning:

```text
Start with total = 0.
Go through each number.
Add it to total.
Print the final total.
```

Output:

```text
60
```

---

## 52. Function

```python
def double(x):
    return 2 * x

print(double(5))
```

Plain meaning:

```text
Define a function named double.
It takes x as input.
It returns 2 times x.
```

Output:

```text
10
```

---

## 53. Type-Hinted Function

```python
def double(x: int) -> int:
    return 2 * x
```

Plain meaning:

```text
This function expects x to be an integer.
It returns an integer.
```

---

## 54. Dataclass

```python
from dataclasses import dataclass

@dataclass
class Point:
    row: int
    column: int

p = Point(row=2, column=3)
print(p.row)
print(p.column)
```

Plain meaning:

```text
Create a simple data structure called Point.
It stores a row and a column.
Make one Point with row 2 and column 3.
Print its row and column.
```

---

## 55. Generator

```python
def count_up_to(n):
    i = 1
    while i <= n:
        yield i
        i = i + 1

for value in count_up_to(3):
    print(value)
```

Plain meaning:

```text
Generate numbers one at a time from 1 to n.
```

Output:

```text
1
2
3
```

---

## 56. NumPy Array

```python
import numpy as np

x = np.array([10, 20, 30])
print(x + 5)
```

Plain meaning:

```text
Create a numerical array.
Add 5 to each element.
```

Output:

```text
[15 25 35]
```

---

# Part 12 — Common Confusions and Clear Answers

---

## 57. Is a Python list the same as a tensor?

No.

A Python list is a general-purpose container.

A tensor is usually a numerical multi-dimensional array with:

```text
regular shape
same data type
fast operations
memory-efficient storage
```

Python list:

```python
[1, "hello", 3.5]
```

Tensor-like array:

```python
np.array([1.0, 2.0, 3.0])
```

---

## 58. Is a matrix the same as a tensor?

A matrix is a 2-dimensional tensor in practical ML language.

```text
single number -> 0D tensor
vector        -> 1D tensor
matrix        -> 2D tensor
higher array  -> 3D, 4D, etc. tensor
```

---

## 59. Why does indexing start at 0 in Python?

Python follows a convention common in many programming languages.

The first element is at offset 0 from the beginning.

Think of a row of boxes:

```text
start -> [10] [20] [30]
          0    1    2
```

The first item is zero steps from the start.

So its index is 0.

---

## 60. Why does shape use tuples?

Shape is a fixed description of dimensions.

Example:

```python
shape = (2, 3)
```

This means:

```text
2 rows, 3 columns
```

Tuples are often used because they are simple, ordered, and immutable.

---

## 61. Why are tensors fast?

Tensors are fast because:

```text
1. They store data in regular blocks.
2. They use one data type at a time.
3. Operations are implemented in compiled code.
4. Many operations avoid slow Python loops.
```

---

## 62. Do I need to understand C?

No.

You only need the concept:

```text
Python may call fast compiled routines written in C/C++/CUDA or similar languages.
```

When using NumPy, PyTorch, TensorFlow, or JAX, you often write Python but the heavy numerical work runs underneath in optimized code.

---

# Part 13 — A 7-Day Preparation Plan

This is a suggested plan before starting the ML book.

---

## Day 1 — Sets and Symbols

Read:

- set,
- element,
- ∈,
- ∉,
- ℕ,
- ℤ,
- ∀.

Practice translating symbols into English.

Goal:

```text
When you see x ∈ A, you can read it as "x is in A."
```

---

## Day 2 — Sum and Product Notation

Read:

- ∑,
- ∏,
- index notation,
- x_i.

Practice expanding notation.

Goal:

```text
∑ from i=1 to 3 of x_i becomes x_1 + x_2 + x_3.
∏ from i=1 to 3 of x_i becomes x_1 * x_2 * x_3.
```

---

## Day 3 — Division, Integer Division, and Ceiling

Read:

- ordinary division,
- remainder,
- `//`,
- ceiling.

Practice batch examples.

Goal:

```text
You can explain why 103 examples with batch size 20 require 6 batches.
```

---

## Day 4 — Python Containers

Read:

- list,
- tuple,
- indexing,
- slicing,
- len.

Practice with small examples.

Goal:

```text
You can read x[0], x[1:4], and shape = (2, 3).
```

---

## Day 5 — Functions, Dataclasses, and Type Hints

Read:

- function,
- f: A -> B,
- Python `def`,
- dataclass,
- type hints.

Goal:

```text
You can read def f(x: int) -> int.
```

---

## Day 6 — Memory and C Extensions

Read:

- memory as addressable bytes,
- contiguous memory,
- Python names and objects,
- NumPy arrays,
- C extensions.

Goal:

```text
You can explain why np.sum(array) is usually faster than a Python for-loop.
```

---

## Day 7 — Tensors

Read:

- scalar,
- vector,
- matrix,
- tensor,
- shape,
- axis,
- indexing,
- reshaping,
- broadcasting intuition.

Goal:

```text
You can look at shape (32, 224, 224, 3) and say what kind of structure it represents.
```

---

# Part 14 — Final Checklist Before Opening the ML Book

You are ready to begin the tensor chapter if you can answer these questions.

## Math checklist

Can you explain these?

```text
x ∈ A
x ∉ A
n ∈ ℕ
k ∈ ℤ
∀ x ∈ A
∑_{i=1}^{3} x_i
∏_{i=1}^{3} x_i
ceil(10 / 4)
f: A -> B
```

## Python checklist

Can you read these?

```python
x = [10, 20, 30]
x[0]
x[1:3]
shape = (2, 3)

def f(x: int) -> int:
    return 2 * x
```

Can you explain these at a high level?

```python
@dataclass
class Config:
    batch_size: int
    epochs: int
```

```python
def generator():
    yield 1
    yield 2
```

```python
obj.__getitem__(0)
obj.__setitem__(0, 99)
```

## Computer memory checklist

Can you explain these?

```text
memory as a long row of bytes
address
contiguous block
Python name referring to an object
NumPy array storing numerical data compactly
C extension doing fast loops underneath Python
```

## Tensor checklist

Can you explain these?

```text
scalar
vector
matrix
tensor
shape
axis
indexing
slicing
reshape
broadcasting intuition
```

---

# Part 15 — One-Page Symbol Dictionary

Keep this page near you while reading.

```text
∈      is in / is an element of
∉      is not in / is not an element of
∀      for all / for every
∑      sum / add up
∏      product / multiply together
ℕ      natural numbers / counting numbers
ℤ      integers / whole numbers including negatives and zero
⌈x⌉    ceiling of x / round upward to a whole number
f: A -> B   function f takes inputs from A and gives outputs in B
x_i    the i-th item of x
```

Python reminders:

```text
list        [10, 20, 30]
tuple       (2, 3)
indexing    x[0]
slicing     x[1:4]
shape       size along each tensor axis
dtype       data type of stored values
yield       produce one value now and continue later
```

Tensor reminders:

```text
scalar      one number
vector      one-dimensional array
matrix      two-dimensional array
tensor      general multi-dimensional numerical array
shape       tuple describing size along each axis
axis        one direction/dimension of tensor
```

---

# Closing Note

The purpose of these prerequisites is not to make you solve advanced mathematics. Their purpose is to make the notation and code readable.

When you begin the ML book, do not try to understand every sentence perfectly on the first reading. Read tensor sections with three questions in mind:

```text
1. What is the shape?
2. What kind of values are stored?
3. What operation is being applied to all or part of the tensor?
```

If you can keep track of these three things, the first tensor chapters will become much less intimidating.

---

# Part 16 — From Tensors to Machine-Learning Models

You have now learned the minimum language of tensors:

```text
data -> numbers -> arrays/tensors -> operations
```

The next step is to understand **models**.

A machine-learning model is not magic. At beginner level, think of a model as:

> A rule, learned from data, that takes input values and produces an output.

In symbols, a model is often written like this:

```text
model: input data -> prediction
```

or:

```text
f: X -> y
```

Read this as:

```text
Function f takes input X and produces output y.
```

In practical words:

```text
house details -> predicted house price
patient measurements -> risk category
flower measurements -> flower species
text message -> topic or spam/not-spam
image pixels -> digit or object label
past pollution readings -> next pollution reading
```

The tensor part tells us how data is stored.

The model part tells us what we do with that data.

---

## 63. The Three Questions Before Choosing Any Model

Before choosing a model, ask three questions.

### Question 1: What is the input?

The input is usually called:

```text
X
```

Examples:

```text
X = table of household features
X = table of flower measurements
X = image pixels
X = text converted into numbers
X = past hourly air-quality readings
```

### Question 2: What is the target?

The target is usually called:

```text
y
```

It is what you want the model to learn to predict.

Examples:

```text
y = house price
y = flower species
y = survived or did not survive
y = income above or below a threshold
y = topic of a document
y = next hour pollution level
```

### Question 3: What kind of answer do I need?

This determines the type of problem.

```text
number prediction      -> regression
category prediction    -> classification
group discovery        -> clustering
compression/summary    -> dimensionality reduction
sequence prediction    -> time-series forecasting
```

---

# Part 17 — Main Types of Machine-Learning Problems

## 64. Regression: Predicting a Number

Regression is used when the answer is a number.

Examples:

```text
Predict house price.
Predict crop yield.
Predict school attendance rate.
Predict electricity demand.
Predict pollution concentration.
Predict hotel occupancy percentage.
```

The target `y` is numerical.

Example:

```text
X = median income, house age, rooms, population
y = median house value
```

The model learns a rule like:

```text
features of area -> estimated price
```

Common regression models:

```text
Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
Neural Network Regressor
```

---

## 65. Classification: Predicting a Category

Classification is used when the answer is a category.

Examples:

```text
Will the borrower repay? yes/no
Is this flower setosa, versicolor, or virginica?
Is this email spam or not spam?
Did the passenger survive or not?
Is this image a 0, 1, 2, ..., or 9?
Is a household poor, vulnerable, or non-poor?
```

The target `y` is a label.

Example:

```text
X = measurements of a flower
y = flower species
```

Common classification models:

```text
Logistic Regression
k-Nearest Neighbors
Naive Bayes
Decision Tree Classifier
Random Forest Classifier
Gradient Boosting Classifier
Support Vector Machine
Neural Network Classifier
```

---

## 66. Clustering: Finding Groups Without Labels

Clustering is used when you do not already have the correct answer.

The model tries to discover natural groups in the data.

Examples:

```text
Group villages by livelihood patterns.
Group hotel guests by booking behavior.
Group districts by service-delivery indicators.
Group customers by purchasing pattern.
Group documents by topic.
```

There is no target `y`.

You only have:

```text
X = features
```

Common clustering models:

```text
K-Means
Hierarchical Clustering
DBSCAN
Gaussian Mixture Models
```

---

## 67. Dimensionality Reduction: Making Many Columns Simpler

Sometimes a dataset has many columns.

Dimensionality reduction tries to create a smaller number of summary columns.

Examples:

```text
Reduce 50 governance indicators to 2 or 3 summary directions.
Compress image data.
Plot high-dimensional data on a 2D chart.
Remove noise before another model.
```

Common dimensionality-reduction models:

```text
PCA
t-SNE
UMAP
Autoencoders
```

At beginner level, start with PCA.

---

## 68. Time-Series Forecasting: Predicting What Comes Next

Time-series data has a time order.

Examples:

```text
hourly electricity demand
daily hotel occupancy
monthly revenue
weekly disease cases
hourly air pollution
daily rainfall
```

The key difference is:

```text
past and future order matters
```

You should not randomly mix future data into the training set.

Common forecasting models:

```text
moving average
linear regression with lagged values
ARIMA-type models
random forest with lag features
gradient boosting with lag features
recurrent neural networks
transformers for long sequences
```

For beginners, start with:

```text
previous value -> next value
moving average -> next value
linear regression using past values
```

---

# Part 18 — The Main Models: Plain-Language Primer

This section explains the most important model families without heavy mathematics.

---

## 69. Baseline Model

Before using a serious model, always build a simple baseline.

A baseline is a simple rule.

For regression:

```text
Always predict the average house price.
```

For classification:

```text
Always predict the most common class.
```

For time series:

```text
Predict that tomorrow will be the same as today.
```

Why baseline matters:

```text
If your advanced model cannot beat the simple baseline, your model is not useful yet.
```

---

## 70. Linear Regression

### What it does

Linear regression predicts a number using a weighted combination of input features.

Plain language:

```text
Start with some base value.
Add or subtract something for each feature.
The result is the prediction.
```

Example:

```text
house price = base amount
            + effect of income
            + effect of rooms
            + effect of location
            + effect of house age
```

### Use cases

```text
price prediction
demand estimation
budget forecasting
risk scoring where output is numerical
relationship between indicators and outcomes
```

### Strengths

```text
easy to explain
fast
good first model
shows direction of relationships
```

### Weaknesses

```text
cannot easily capture complex curves unless you add features
sensitive to outliers
assumes relationships are fairly smooth/simple
```

### Best beginner dataset

```text
California Housing
```

---

## 71. Logistic Regression

Despite the name, logistic regression is usually used for classification.

It predicts the probability of a category.

Example:

```text
probability of survival = 0.78
probability of default = 0.22
probability income is above threshold = 0.64
```

Then we choose a cutoff.

Example:

```text
if probability >= 0.50, predict yes
otherwise, predict no
```

### Use cases

```text
yes/no decisions
risk classification
medical screening support
credit scoring
survival prediction
attendance/non-attendance prediction
```

### Strengths

```text
interpretable
fast
good baseline for classification
gives probabilities
```

### Weaknesses

```text
needs careful handling of categories
may miss complex patterns
probabilities may need calibration
```

### Best beginner datasets

```text
Titanic
Adult Income
Breast Cancer
```

---

## 72. k-Nearest Neighbors

k-Nearest Neighbors, often called KNN, predicts by comparing a new case to similar past cases.

Plain language:

```text
Find the k most similar examples.
Let them vote.
```

Example:

```text
A new flower is measured.
Find the 5 most similar flowers in the old data.
If most of those 5 are Iris setosa, predict Iris setosa.
```

### Use cases

```text
small classification problems
similarity-based recommendation
simple pattern recognition
teaching the idea of distance
```

### Strengths

```text
very intuitive
little training time
good for teaching
```

### Weaknesses

```text
slow when dataset is large
sensitive to feature scale
does not explain much
struggles when many features are irrelevant
```

### Best beginner dataset

```text
Iris
```

---

## 73. Naive Bayes

Naive Bayes is often used for text classification.

Plain language:

```text
Look at which words appear.
Estimate which category those words most strongly suggest.
```

Example:

```text
Words like "offer", "winner", "free" may suggest spam.
Words like "meeting", "agenda", "minutes" may suggest office communication.
```

The word “naive” means the model makes a simplifying assumption:

```text
It treats features as if they are more independent than they really are.
```

Even though this assumption is not fully true, the model can work surprisingly well.

### Use cases

```text
spam detection
document topic classification
simple sentiment classification
first model for text data
```

### Strengths

```text
fast
works well on text
good baseline
can work with many word features
```

### Weaknesses

```text
simplifying assumptions are crude
not ideal for subtle language meaning
```

### Best beginner dataset

```text
20 Newsgroups
```

---

## 74. Decision Tree

A decision tree makes predictions through a sequence of if-then questions.

Plain language:

```text
If income is high, go left.
If house age is low, go right.
If rooms are many, predict higher price.
```

For classification:

```text
If petal length is small, predict setosa.
Otherwise ask another question.
```

### Use cases

```text
explainable rules
classification
regression
policy triage
risk segmentation
```

### Strengths

```text
easy to visualize
handles non-linear patterns
handles both numbers and categories after preprocessing
```

### Weaknesses

```text
can overfit easily
small data changes can create different trees
often weaker than forests or boosting
```

### Best beginner datasets

```text
Iris
Titanic
California Housing
```

---

## 75. Random Forest

A random forest is a collection of many decision trees.

Plain language:

```text
Train many trees.
Each tree gives an answer.
Average their answers or let them vote.
```

For classification:

```text
many trees vote for the class
```

For regression:

```text
many trees produce numbers, then the model averages them
```

### Use cases

```text
strong general-purpose tabular prediction
house price prediction
risk classification
feature importance
operations and service-delivery prediction
```

### Strengths

```text
usually accurate
less overfitting than one tree
works well with tabular data
needs less scaling than KNN or SVM
```

### Weaknesses

```text
less interpretable than one tree
can be large and slower
not naturally ideal for text or images without feature engineering
```

### Best beginner datasets

```text
California Housing
Adult Income
Titanic
```

---

## 76. Gradient Boosting

Gradient boosting also uses many trees, but in a different way.

Plain language:

```text
Build one tree.
Look at its mistakes.
Build the next tree to correct some of those mistakes.
Repeat.
```

This is like a team that learns from previous errors.

Popular versions include:

```text
GradientBoosting
HistGradientBoosting
XGBoost
LightGBM
CatBoost
```

### Use cases

```text
high-performing tabular prediction
competitions
credit risk
demand forecasting
price prediction
classification with mixed features
```

### Strengths

```text
often very accurate on tabular data
captures complex patterns
can beat random forests when tuned well
```

### Weaknesses

```text
more settings to tune
can overfit if careless
less transparent than simple models
```

### Best beginner datasets

```text
California Housing
Adult Income
Air Quality
```

---

## 77. Support Vector Machine

A support vector machine, often called SVM, tries to create a strong boundary between classes.

Plain language:

```text
Find a dividing line or surface that separates categories with the widest safe margin.
```

For two flower classes, it tries to draw a boundary so that the groups are separated cleanly.

### Use cases

```text
small or medium classification datasets
high-dimensional data
image recognition before deep learning became dominant
text classification with suitable features
```

### Strengths

```text
can work well with clear boundaries
powerful for some small/medium datasets
```

### Weaknesses

```text
needs scaling
can be slow on large datasets
harder to explain
parameters can be confusing for beginners
```

### Best beginner datasets

```text
Iris
Digits
```

---

## 78. K-Means Clustering

K-Means finds groups in data.

Plain language:

```text
Choose k groups.
Put each case into the nearest group.
Move the group centers.
Repeat until the groups stabilize.
```

Example:

```text
Group districts into 3 types based on indicators.
Group guests into 4 customer segments.
Group villages into livelihood profiles.
```

### Use cases

```text
segmentation
exploratory analysis
finding patterns before formal prediction
compressing examples into representative groups
```

### Strengths

```text
simple
fast
good first clustering model
```

### Weaknesses

```text
you must choose k
sensitive to feature scale
assumes round-ish clusters
clusters may not have real-world meaning unless interpreted carefully
```

### Best beginner datasets

```text
Iris without labels
Wine
district-level indicators if you have them
```

---

## 79. PCA

PCA means Principal Component Analysis.

Plain language:

```text
Take many columns and create fewer summary directions that preserve as much variation as possible.
```

Example:

```text
50 governance indicators -> 2 summary axes for a chart
```

PCA is not mainly for prediction. It is often for:

```text
visualization
compression
noise reduction
understanding structure
```

### Use cases

```text
plotting high-dimensional data
reducing many correlated indicators
preparing data before clustering
image compression
```

### Strengths

```text
useful for visualization
fast
mathematically clean
```

### Weaknesses

```text
summary axes may be hard to interpret
only captures linear patterns
sensitive to feature scale
```

### Best beginner datasets

```text
Iris
Wine
Digits
```

---

## 80. Neural Networks

A neural network is a model made of layers.

Plain language:

```text
Each layer transforms the data.
Later layers learn more useful representations.
The final layer produces the prediction.
```

A very simple neural network may look like:

```text
input features -> hidden layer -> output prediction
```

For images, neural networks can learn from pixels.

For text, modern neural networks can learn from word or token representations.

For tabular data, simpler models often compete very well, so do not assume neural networks are always best.

### Main neural-network types

```text
MLP: ordinary feed-forward network for tabular data
CNN: convolutional neural network for images
RNN/LSTM/GRU: older sequence models for time series and text
Transformer: modern architecture for text, images, and many sequence tasks
Autoencoder: learns compressed representation of data
```

### Use cases

```text
image recognition
speech recognition
large-scale text processing
translation
language models
complex pattern recognition
```

### Strengths

```text
very flexible
can learn from raw pixels, audio, and text
strong at large-scale problems
```

### Weaknesses

```text
needs more data
needs more computation
harder to explain
more ways to go wrong
```

### Best beginner datasets

```text
Digits
MNIST
```

---

# Part 19 — The Standard Methodology

A machine-learning project follows a disciplined sequence.

Do not begin by asking:

```text
Which model should I use?
```

Begin by asking:

```text
What decision or prediction problem am I trying to solve?
```

---

## 81. Step 1: Define the Prediction Question

Bad question:

```text
Can we use AI on this data?
```

Better question:

```text
Can we predict next month’s district-level school attendance from the previous 12 months?
```

or:

```text
Can we classify households into high, medium, and low risk of dropping out of a programme?
```

or:

```text
Can we estimate hotel occupancy for the next week?
```

A good prediction question names:

```text
1. the unit of analysis
2. the input information available at prediction time
3. the target to predict
4. the time at which the prediction will be used
5. the action the prediction will support
```

---

## 82. Step 2: Identify Rows, Features, and Target

Most beginner ML data is a table.

```text
rows = examples
columns = features
target = what you want to predict
```

Example:

```text
One row = one house district
Features = income, house age, rooms, population
Target = median house value
```

Another example:

```text
One row = one passenger
Features = age, sex, ticket class, fare
Target = survived or did not survive
```

---

## 83. Step 3: Split Data into Training and Test Sets

The model must be tested on data it has not seen.

```text
training set = used to learn
test set     = used to check performance
```

A common beginner split:

```text
80% training
20% testing
```

For time-series data, do not split randomly.

Use time order:

```text
older data -> training
newer data -> testing
```

---

## 84. Step 4: Build a Baseline

Examples:

```text
Regression baseline: predict the average value.
Classification baseline: predict the most common class.
Time-series baseline: predict that the next value equals the last value.
```

Then ask:

```text
Does my model beat the baseline?
```

---

## 85. Step 5: Train the Model

In scikit-learn, the usual pattern is:

```python
model.fit(X_train, y_train)
```

This means:

```text
Let the model learn from the training data.
```

Then:

```python
predictions = model.predict(X_test)
```

This means:

```text
Use the learned model to predict unseen test data.
```

---

## 86. Step 6: Evaluate the Model

The metric depends on the problem.

### Regression metrics

```text
MAE  = average size of error
RMSE = like MAE but punishes big errors more
R²   = how much variation the model explains
```

For a beginner, MAE is easiest.

If MAE is 25,000 in a house-price problem, read it as:

```text
On average, predictions are off by about 25,000 units of money.
```

### Classification metrics

```text
accuracy  = percent correct
precision = when model says yes, how often is it right?
recall    = out of all real yes cases, how many did model find?
F1        = balance of precision and recall
```

Accuracy can mislead when classes are unbalanced.

Example:

```text
If 95% of people repay loans, a dumb model that always predicts "repay" gets 95% accuracy.
```

But it may completely fail to identify risky cases.

---

## 87. Step 7: Look at Errors

A model report is not enough.

Look at the wrong predictions.

Ask:

```text
Which cases does the model miss?
Are errors larger for some groups?
Are some features missing?
Is the target poorly defined?
Is the data old?
Is the data biased?
```

This is where practical judgment matters.

---

## 88. Step 8: Use the Model Carefully

A model should support judgment, not replace it blindly.

Ask:

```text
What action will this prediction trigger?
Who may be harmed by an error?
Can the affected person appeal?
Can the model be monitored?
Will the data distribution change over time?
```

For development, government, finance, health, and social services, this matters greatly.

---

# Part 20 — Data Sources for Beginner Exercises

These are real public datasets suitable for learning.

Some are small and safe for the first week. Others are more realistic and messy.

## 89. Recommended Dataset Links

| Dataset | Best for | Problem type | Why useful | Link |
|---|---|---|---|---|
| Iris | first classification model | classification | tiny, clean, visual | [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris) |
| California Housing | predicting a number | regression | real-world tabular data | [scikit-learn California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) |
| Titanic | yes/no prediction with missing values | classification | intuitive story, mixed columns | [OpenML Titanic](https://www.openml.org/d/40945) |
| Adult Income | classification with social variables | classification | categorical variables, ethics, fairness | [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) |
| 20 Newsgroups | text classification | classification | turns words into numbers | [scikit-learn 20 Newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html) |
| Digits | image-like data without large download | classification | small 8x8 image tensors | [scikit-learn Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) |
| MNIST | image classification | classification | classic handwritten digits | [Yann LeCun MNIST](https://yann.lecun.com/exdb/mnist/) |
| Air Quality | time series and regression | regression/time series | hourly sensor data with missing values | [UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) |

---

## 90. The Order in Which to Use These Datasets

Do not start with the hardest dataset.

Use this order:

```text
1. Iris
2. California Housing
3. Titanic
4. Adult Income
5. 20 Newsgroups
6. Digits
7. Air Quality
8. MNIST
```

Why this order?

```text
Iris teaches classification.
California Housing teaches regression.
Titanic teaches missing values and categorical variables.
Adult teaches ethics, categorical encoding, and fairness concerns.
20 Newsgroups teaches text-to-numbers.
Digits teaches image-like tensors.
Air Quality teaches time order and missing values.
MNIST teaches larger image tensors and neural networks.
```

---

# Part 21 — Exercise Build-Up on Real Data

The exercises below are written as a ladder.

Each exercise adds one new concept.

---

## 91. Exercise 1: Iris — First Classification Model

### Goal

Learn how a model predicts categories.

### Dataset

```text
Iris
```

### Question

```text
Can we predict the species of an iris plant from flower measurements?
```

### Rows, features, target

```text
row = one plant
features = sepal length, sepal width, petal length, petal width
target = species
```

### Models to try

```text
k-Nearest Neighbors
Logistic Regression
Decision Tree
```

### Method

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = load_iris(as_frame=True)

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
```

### What to understand

```text
X is a table.
y is the correct species.
The model sees training rows.
The model predicts test rows.
Accuracy tells how many it got right.
```

### Follow-up questions

```text
What happens if k = 1?
What happens if k = 15?
Which features seem most useful?
Which species are confused with each other?
```

---

## 92. Exercise 2: California Housing — First Regression Model

### Goal

Learn how a model predicts a number.

### Dataset

```text
California Housing
```

### Question

```text
Can we predict median house value from area-level features?
```

### Rows, features, target

```text
row = one California block group
features = income, house age, rooms, bedrooms, population, occupancy, latitude, longitude
target = median house value
```

### Models to try

```text
Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
```

### Method

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = fetch_california_housing(as_frame=True)

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
```

### What to understand

```text
This is regression because y is a number.
MAE is the average size of the prediction error.
A lower MAE is better.
```

### Follow-up questions

```text
Does a decision tree do better or worse?
Does a random forest improve the error?
Which features seem most important?
Are errors larger for expensive areas?
```

---

## 93. Exercise 3: Titanic — Missing Values and Categories

### Goal

Learn that real data is often messy.

### Dataset

```text
Titanic
```

### Question

```text
Can we predict whether a passenger survived?
```

### Rows, features, target

```text
row = one passenger
features = age, sex, ticket class, fare, family aboard, etc.
target = survived or not
```

### Models to try

```text
Logistic Regression
Decision Tree
Random Forest
```

### New concepts

```text
missing values
categorical variables
one-hot encoding
pipeline
```

### Method

The exact column names may vary depending on the source.

Begin with this idea:

```python
from sklearn.datasets import fetch_openml

data = fetch_openml(data_id=40945, as_frame=True)

df = data.frame
print(df.head())
print(df.columns)
```

Then identify:

```text
which column is the target
which columns are useful features
which columns have missing values
which columns are text categories
```

### What to understand

A model cannot directly understand text such as:

```text
male
female
first class
second class
Southampton
Cherbourg
```

These must be converted into numbers.

One common method is:

```text
one-hot encoding
```

Example:

```text
sex = male/female
```

becomes:

```text
sex_male     sex_female
1            0
0            1
```

### Follow-up questions

```text
What happens if you use only sex and passenger class?
What happens if you add age and fare?
Do missing ages matter?
Does accuracy hide important mistakes?
```

---

## 94. Exercise 4: Adult Income — Classification, Bias, and Fairness

### Goal

Learn that prediction is not only technical.

### Dataset

```text
Adult Income / Census Income
```

### Question

```text
Can we predict whether income is above a threshold from census variables?
```

### Rows, features, target

```text
row = one person
features = age, education, occupation, hours worked, etc.
target = income above or below threshold
```

### Models to try

```text
Logistic Regression
Random Forest
Gradient Boosting
```

### New concepts

```text
categorical encoding
class imbalance
precision and recall
fairness questions
```

### Ethical warning

This dataset contains social and demographic variables.

Do not treat the model as a neutral truth machine.

Ask:

```text
Could the data reflect historical inequality?
Could a model reproduce discrimination?
Should some variables be excluded?
Would exclusion really remove bias, or would proxies remain?
Who is affected by errors?
```

### Follow-up questions

```text
What is the accuracy baseline?
What are precision and recall?
Do errors differ across groups?
What happens if some demographic columns are removed?
What real-world decision would this model support, and should it?
```

---

## 95. Exercise 5: 20 Newsgroups — Turning Text into Numbers

### Goal

Learn how text becomes numerical features.

### Dataset

```text
20 Newsgroups
```

### Question

```text
Can we predict the topic of a document from its words?
```

### Rows, features, target

```text
row = one document/post
features = words converted into numbers
target = topic/category
```

### Models to try

```text
Naive Bayes
Logistic Regression
Linear SVM
```

### Method

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))

model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

model.fit(train.data, train.target)

pred = model.predict(test.data)
print("Accuracy:", accuracy_score(test.target, pred))
```

### What to understand

The model does not understand words as humans do.

The vectorizer converts text into numbers.

Simplified:

```text
document -> word counts or word weights -> numerical vector -> model
```

### Follow-up questions

```text
Which topics are easiest?
Which topics are confused?
What happens if headers are not removed?
Why might metadata create misleading accuracy?
```

---

## 96. Exercise 6: Digits — Image Data as Tensors

### Goal

Learn how an image becomes a grid of numbers.

### Dataset

```text
Digits
```

### Question

```text
Can we identify handwritten digits from small images?
```

### Rows, features, target

```text
row = one image
features = pixel values
target = digit 0 to 9
```

### Method

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

digits = load_digits()

print(digits.images.shape)
print(digits.data.shape)
print(digits.target.shape)

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
```

### What to understand

You may see:

```text
digits.images.shape = (1797, 8, 8)
```

This means:

```text
1797 images
each image has 8 rows
each image has 8 columns
```

The flattened version may be:

```text
digits.data.shape = (1797, 64)
```

This means:

```text
1797 rows
64 pixel columns
```

So the image tensor was reshaped into a table.

---

## 97. Exercise 7: Air Quality — Time-Series Regression

### Goal

Learn that time order changes the method.

### Dataset

```text
Air Quality
```

### Question

```text
Can we predict pollution concentration from sensor readings and previous values?
```

### Rows, features, target

```text
row = one hourly reading
features = sensor readings and time variables
target = pollutant concentration
```

### New concepts

```text
time order
lag features
missing values
train/test split by date
```

### Important warning

Do not randomly split time-series data if the goal is forecasting.

Use:

```text
earlier dates -> train
later dates   -> test
```

### Simple method

Create lag features:

```text
value at previous hour
value two hours ago
value 24 hours ago
```

Then train a regression model.

### Follow-up questions

```text
Does yesterday's value help?
Does the same hour yesterday help?
How do missing values affect results?
Does the model fail during unusual pollution episodes?
```

---

## 98. Exercise 8: K-Means and PCA on Iris

### Goal

Learn unsupervised learning.

### Dataset

```text
Iris, but hide the species labels at first.
```

### Question

```text
Can the model discover groups without being told the species?
```

### Method

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data = load_iris(as_frame=True)

X = data.data

X_scaled = StandardScaler().fit_transform(X)

clusters = KMeans(n_clusters=3, random_state=42, n_init="auto").fit_predict(X_scaled)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

print("First five PCA rows:")
print(X_2d[:5])

print("First five cluster labels:")
print(clusters[:5])
```

### What to understand

K-Means does not know the true species.

It only sees measurements.

PCA reduces four measurement columns to two summary columns for plotting.

### Follow-up questions

```text
Do the clusters match the real species?
Which species are easiest to separate?
What happens if n_clusters = 2?
What happens if n_clusters = 4?
```

---

# Part 22 — Model Selection: Which Model Should I Try First?

## 99. Simple Beginner Decision Guide

Use this table as a starting point.

| Your problem | First model | Next model | Why |
|---|---|---|---|
| Predict a number from a table | Linear Regression | Random Forest / Gradient Boosting | Start interpretable, then improve accuracy |
| Predict yes/no from a table | Logistic Regression | Random Forest / Gradient Boosting | Start with probabilities, then test non-linear models |
| Predict a category from clean measurements | KNN / Logistic Regression | SVM / Random Forest | Good for learning boundaries |
| Text classification | Naive Bayes | Logistic Regression / Linear SVM | Strong simple baselines for text |
| Image classification | Logistic Regression on flattened pixels | CNN | Start with shape understanding, then deep learning |
| Find groups without labels | K-Means | Hierarchical / DBSCAN | Start with simple clustering |
| Reduce many columns | PCA | UMAP / Autoencoder | Start with interpretable linear compression |
| Forecast future values | Last-value baseline | Lagged regression / tree models | Respect time order |

---

## 100. How to Compare Models Fairly

Do not compare models on different test sets.

Use the same train/test split.

Example:

```python
models = {
    "logistic_regression": model1,
    "random_forest": model2,
    "gradient_boosting": model3,
}
```

For each model:

```text
train on X_train, y_train
predict on X_test
evaluate against y_test
```

Then compare.

A fair comparison means:

```text
same data
same target
same metric
same test set
```

---

## 101. Overfitting and Underfitting

### Underfitting

The model is too simple.

Example:

```text
A straight line is used for a very curved relationship.
```

Symptoms:

```text
bad on training data
bad on test data
```

### Overfitting

The model memorizes the training data.

Symptoms:

```text
excellent on training data
poor on test data
```

A decision tree grown too deep often overfits.

### Good fit

The model captures useful patterns without memorizing noise.

Symptoms:

```text
good on training data
good on test data
```

---

## 102. Cross-Validation

A single train/test split can be unlucky.

Cross-validation repeats the test in several folds.

Plain language:

```text
Split the data into several parts.
Train on most parts.
Test on the remaining part.
Repeat so each part gets a turn as test data.
Average the results.
```

Use cross-validation after you understand simple train/test split.

---

# Part 23 — Practical Use Cases by Sector

## 103. Public Administration and Development

Possible use cases:

```text
predict school dropout risk
forecast demand for health facilities
identify villages with similar livelihood patterns
estimate maintenance risk for water schemes
prioritize inspections based on risk
classify grievances by topic
forecast electricity demand in micro-hydel schemes
detect unusual project expenditure patterns
```

Use caution:

```text
Models can support prioritization.
They should not replace accountability, field verification, or citizen voice.
```

---

## 104. Small Hotel or Resort Management

Possible use cases:

```text
forecast occupancy
predict food demand for buffet planning
classify guest reviews by topic
estimate probability of cancellation
segment guests by booking behavior
identify seasonal demand patterns
```

Useful models:

```text
linear regression
random forest
gradient boosting
time-series baseline models
text classification for reviews
```

---

## 105. Health and Service Delivery

Possible use cases:

```text
predict appointment no-shows
classify risk groups
forecast medicine demand
detect unusual lab readings
prioritize follow-up calls
```

Use caution:

```text
Health models need clinical review.
False positives and false negatives both matter.
A model should not be used beyond the data on which it was validated.
```

---

## 106. Agriculture and Climate

Possible use cases:

```text
predict crop yield
forecast water demand
classify plant disease images
cluster farms by soil and climate profile
forecast pest risk
estimate impact of rainfall patterns
```

Useful models:

```text
regression
random forest
gradient boosting
CNNs for images
time-series models
```

---

# Part 24 — Mini-Glossary of Model Terms

## 107. Feature

A feature is an input column.

Example:

```text
age
income
rainfall
petal length
number of rooms
previous hour pollution
```

---

## 108. Target

The target is what we want to predict.

Example:

```text
house price
survived/not survived
flower species
digit label
next hour pollution
```

---

## 109. Label

A label is a known correct answer in supervised learning.

Example:

```text
This image is digit 7.
This passenger survived.
This flower is setosa.
```

---

## 110. Prediction

A prediction is the model's output.

Example:

```text
Predicted price = 240,000
Predicted class = setosa
Predicted probability of survival = 0.72
```

---

## 111. Parameter

A parameter is learned by the model from data.

Example:

```text
the weight attached to income in linear regression
the split points inside a decision tree
the internal weights of a neural network
```

---

## 112. Hyperparameter

A hyperparameter is chosen by the user before or during training.

Examples:

```text
k in KNN
maximum depth of a decision tree
number of trees in a random forest
learning rate in gradient boosting
number of layers in a neural network
```

---

## 113. Training

Training means fitting the model to data.

```python
model.fit(X_train, y_train)
```

---

## 114. Inference

Inference means using a trained model to make predictions.

```python
model.predict(X_new)
```

---

## 115. Generalization

Generalization means the model works on new data, not just the training data.

This is the central goal of machine learning.

---

# Part 25 — A 14-Day Model Primer Plan

This plan comes after the earlier 7-day tensor preparation plan.

## Day 1 — What Is a Model?

Read:

```text
model as function
X and y
features and target
regression vs classification
```

Practice:

```text
For 5 real problems, identify X, y, and problem type.
```

---

## Day 2 — Iris Classification

Run:

```text
KNN on Iris
Logistic Regression on Iris
Decision Tree on Iris
```

Goal:

```text
Understand train/test split and accuracy.
```

---

## Day 3 — Regression with California Housing

Run:

```text
Linear Regression
Decision Tree Regressor
Random Forest Regressor
```

Goal:

```text
Understand MAE and prediction errors.
```

---

## Day 4 — Overfitting

Use a decision tree.

Try:

```text
max_depth = 2
max_depth = 5
no max_depth
```

Goal:

```text
See the difference between training score and test score.
```

---

## Day 5 — Titanic Data Cleaning

Practice:

```text
missing values
categorical variables
one-hot encoding
pipeline
```

Goal:

```text
Understand that real data needs preparation.
```

---

## Day 6 — Logistic Regression and Probabilities

Use Titanic or Adult.

Goal:

```text
Understand probability, threshold, precision, recall.
```

---

## Day 7 — Random Forest and Feature Importance

Use California Housing or Adult.

Goal:

```text
Understand why many trees can outperform one tree.
```

---

## Day 8 — Gradient Boosting

Use California Housing.

Goal:

```text
Understand sequential error correction.
```

Do not tune too much at first.

---

## Day 9 — Text Classification

Use 20 Newsgroups.

Goal:

```text
Understand text -> numbers -> model.
```

---

## Day 10 — Clustering

Use Iris without labels.

Goal:

```text
Understand unsupervised learning and why clusters need interpretation.
```

---

## Day 11 — PCA

Use Iris or Wine.

Goal:

```text
Reduce several columns to two columns and plot them.
```

---

## Day 12 — Digits and Image Tensors

Use Digits.

Goal:

```text
Understand image shape and flattening.
```

---

## Day 13 — Time-Series Forecasting

Use Air Quality or your own hotel occupancy data.

Goal:

```text
Understand time split and lag features.
```

---

## Day 14 — Review and Model Choice

Create a table:

```text
problem
data type
target
model tried
metric
result
lesson learned
```

Goal:

```text
Know which model to try first for common situations.
```

---

# Part 26 — Final Checklist Before Studying Models Seriously

You are ready to study machine-learning models if you can answer these.

## Problem checklist

```text
What is the unit of analysis?
What are the features?
What is the target?
Is this regression, classification, clustering, dimensionality reduction, or forecasting?
What action will the prediction support?
```

## Data checklist

```text
How many rows?
How many columns?
Which columns are numbers?
Which columns are categories?
Are there missing values?
Is there time order?
Could there be bias?
```

## Model checklist

```text
What is the baseline?
Which model is simplest?
Which model is more powerful?
What metric will I use?
Does the model beat the baseline?
Does it generalize to test data?
Where does it make mistakes?
```

## Ethics and judgment checklist

```text
Who may be affected by the model?
Can errors cause harm?
Is the data representative?
Can the model be challenged or audited?
Should the model advise, prioritize, or decide?
```

---

# Part 27 — One-Page Model Dictionary

Keep this near you while reading model chapters.

```text
Model                  Plain meaning                       Main use
Baseline               simple rule                          first comparison
Linear Regression      weighted sum -> number               regression
Logistic Regression    weighted sum -> probability          classification
KNN                    vote of similar examples             classification/regression
Naive Bayes            word/evidence probabilities          text classification
Decision Tree          if-then questions                    classification/regression
Random Forest          many trees voting/averaging          tabular prediction
Gradient Boosting      trees correcting earlier mistakes    strong tabular prediction
SVM                    boundary with wide margin            classification
K-Means                group by nearest center              clustering
PCA                    reduce many columns to few           visualization/compression
Neural Network         layered transformations              images/text/complex data
CNN                    neural network for grids/images      image recognition
RNN/LSTM/GRU           neural network for sequences         time/text sequences
Transformer            attention-based sequence model       modern text/multimodal AI
```

---

# Closing Note on Models

Do not try to learn all models at once.

For practical understanding, learn them in this order:

```text
1. Baseline
2. Linear Regression
3. Logistic Regression
4. KNN
5. Decision Tree
6. Random Forest
7. Gradient Boosting
8. Naive Bayes for text
9. K-Means
10. PCA
11. Neural Networks
```

Always return to the same questions:

```text
1. What is the input X?
2. What is the target y?
3. What shape is the data?
4. What model is being used?
5. What metric judges success?
6. Does the model beat a simple baseline?
7. What errors does it make?
8. What real-world decision would this support?
```

If you can answer these questions, you understand the methodology, not just the code.

---

# Part 28 — Algebra: Equations, Variables, and Graphs

This part fills a gap the earlier parts of this chapter did not cover. Parts 2 and 3 taught you what numbers and symbols mean. This part teaches you how to **move symbols around correctly** — the single most important hand-skill for reading the regression chapter that follows this one.

Do not move on to the Regression book until you can complete every exercise in this part without checking the solution first.

## 116. Order of Operations

When a line has more than one operation, you must do them in a fixed order. Otherwise two people could compute two different answers from the same line — which would make mathematics useless for communication.

The order, from first to last:

```text
1. Parentheses      ( )
2. Powers and roots  ^  sqrt
3. Multiply and divide   *  /
4. Add and subtract      +  -
```

Worked example:

```text
2 + 3 * 4
```

Multiplication happens before addition:

```text
3 * 4 = 12
2 + 12 = 14
```

The answer is 14, not 20.

Worked example with parentheses:

```text
(2 + 3) * 4
```

Parentheses happen first:

```text
2 + 3 = 5
5 * 4 = 20
```

The answer is 20. The parentheses changed the order, and changed the answer.

**Common mistake.** Reading left to right and ignoring the priority of operations. `2 + 3 * 4` is not `5 * 4`.

## 117. Percentages, Ratios, and Powers

**Percentage** means "out of 100." 25% means 25 out of every 100.

```text
25% of 200 = 0.25 * 200 = 50
```

To convert a percentage to a decimal, divide by 100. To convert a decimal to a percentage, multiply by 100.

```text
0.4  =  40%
0.03 =  3%
1.5  =  150%
```

**Percentage change** compares a new value to an old value:

```text
percentage change = (new - old) / old * 100
```

Worked example: a price rises from 40 to 50.

```text
(50 - 40) / 40 * 100 = 10 / 40 * 100 = 25%
```

**Ratio** compares two quantities directly, without converting to "out of 100."

```text
3 : 2
```

means "for every 3 of the first thing, there are 2 of the second thing."

**Powers** mean repeated multiplication of the same number.

```text
2^3 = 2 * 2 * 2 = 8
```

The small raised number (3) is called the **exponent**. It tells you how many times to multiply.

**Roots** undo powers. The square root of 9 asks "what number, multiplied by itself, gives 9?"

```text
sqrt(9) = 3, because 3 * 3 = 9
```

**Common mistake.** Writing `2^3` and calculating `2 * 3 = 6`. A power is repeated *multiplication* of the base by itself, not multiplication of the base by the exponent.

## 118. Variables, Expressions, and Equations

A **variable** is a letter that stands for a number you do not yet know, or a number that can change. Common letters: `x`, `y`, `a`, `b`.

An **expression** is a combination of numbers, variables, and operations, with no equals sign:

```text
3x + 5
```

An **equation** is a statement that two expressions are equal:

```text
3x + 5 = 20
```

**Solving an equation** means finding the value of the variable that makes the statement true.

Worked example:

```text
3x + 5 = 20
```

Step 1: subtract 5 from both sides (whatever you do to one side, you must do to the other, or the two sides stop being equal):

```text
3x + 5 - 5 = 20 - 5
3x = 15
```

Step 2: divide both sides by 3:

```text
3x / 3 = 15 / 3
x = 5
```

Check by substituting `x = 5` back into the original equation:

```text
3(5) + 5 = 15 + 5 = 20   ✓ matches the right-hand side
```

**Substitution** means replacing a variable with a specific number. You just did this in the check above.

An **inequality** is like an equation but uses `<`, `>`, `<=`, or `>=` instead of `=`. It is solved the same way, with one exception: multiplying or dividing both sides by a *negative* number flips the direction of the inequality sign.

```text
-2x < 8
x > -4        (divided by -2, so the < flipped to >)
```

**Common mistake.** Forgetting to apply the same operation to *every* term on a side, not just one of them. In `3x + 5 = 20`, subtracting 5 must be applied to the whole left side, and the whole right side.

## 119. Rearranging a Formula

Regression is full of formulas that must be rearranged to answer a specific question. This is the exact skill from Section 118, applied to formulas with more than one letter.

Worked example. Suppose:

```text
y = a + bx
```

You are given `y`, `a`, and `b`, and asked to find `x`.

Step 1: subtract `a` from both sides:

```text
y - a = bx
```

Step 2: divide both sides by `b` (assuming `b` is not zero):

```text
(y - a) / b = x
```

So:

```text
x = (y - a) / b
```

This is exactly the formula used later to convert a fitted regression line back into a predicted input.

**Rule to remember.** To isolate a variable, undo the operations applied to it, in reverse order, applying the same undoing operation to both sides of the equation every time.

## 120. Coordinate Axes, Slope, and Intercept

A **coordinate plane** has two number lines crossing at a right angle: the horizontal **x-axis** and the vertical **y-axis**. A point is written `(x, y)` — how far right, then how far up.

```text
(0, 0)   the origin, where the axes cross
(3, 2)   3 units right, 2 units up
(-1, 4)  1 unit left, 4 units up
```

A **linear equation** `y = a + bx` draws a straight line on this plane.

- `a` is the **intercept**: the value of `y` when `x = 0`. It is where the line crosses the y-axis.
- `b` is the **slope**: how much `y` changes for every 1-unit increase in `x`.

Worked example: `y = 2 + 3x`.

```text
x = 0:  y = 2 + 3(0) = 2    point (0, 2)
x = 1:  y = 2 + 3(1) = 5    point (1, 5)
x = 2:  y = 2 + 3(2) = 8    point (2, 8)
```

Each time `x` increases by 1, `y` increases by 3. That 3 is the slope.

The slope between any two points on the same line can be recovered from:

```text
slope = (y2 - y1) / (x2 - x1)
```

Using the points `(0, 2)` and `(2, 8)`:

```text
slope = (8 - 2) / (2 - 0) = 6 / 2 = 3
```

Matches `b = 3` from the equation, as it must.

**Common mistake.** Confusing the intercept with the slope — they answer different questions. The intercept answers "where does the line start?" The slope answers "how fast does it rise or fall?"

## 121. Systems of Two Equations

Sometimes you have two unknowns and two equations, and need both values at once.

```text
x + y = 10
x - y = 2
```

**Method: elimination.** Add the two equations together. Notice that `y` and `-y` cancel:

```text
(x + y) + (x - y) = 10 + 2
2x = 12
x = 6
```

Substitute `x = 6` into the first equation:

```text
6 + y = 10
y = 4
```

Check both equations:

```text
6 + 4 = 10   ✓
6 - 4 = 2    ✓
```

This is exactly the tool used later to derive the two coefficients of a simple regression line — regression's "normal equations" are a system of two equations in two unknowns, solved by the same elimination logic.

---

## Algebra Practice

### Exercise 1

Solve for `x`:

```text
5x - 7 = 18
```

### Solution

```text
5x - 7 = 18
5x = 18 + 7
5x = 25
x = 5
```

Check: `5(5) - 7 = 25 - 7 = 18` ✓

### Exercise 2

Rearrange `y = a + bx` to solve for `b`.

### Solution

```text
y = a + bx
y - a = bx
(y - a) / x = b
```

So `b = (y - a) / x` (assuming `x` is not zero).

### Exercise 3

A line passes through `(1, 4)` and `(3, 10)`. Find its slope and intercept.

### Solution

Slope:

```text
slope = (10 - 4) / (3 - 1) = 6 / 2 = 3
```

Intercept: use `y = a + bx` with the point `(1, 4)` and `b = 3`:

```text
4 = a + 3(1)
4 = a + 3
a = 1
```

The line is `y = 1 + 3x`. Check with the second point: `1 + 3(3) = 1 + 9 = 10` ✓

### Exercise 4

Solve the system:

```text
2x + y = 11
x - y = 1
```

### Solution

Add the two equations (`y` and `-y` cancel):

```text
3x = 12
x = 4
```

Substitute into the second equation: `4 - y = 1`, so `y = 3`.

Check: `2(4) + 3 = 11` ✓ and `4 - 3 = 1` ✓

### Exercise 5

A price increases from 80 to 92. What is the percentage change?

### Solution

```text
(92 - 80) / 80 * 100 = 12 / 80 * 100 = 15%
```

---

# Part 29 — Functions, Growth, and Logarithms

Regression models a straight-line relationship. Many real relationships are not straight lines — they grow multiplicatively (money in a bank account), or they represent a probability squeezed between 0 and 1 (logistic regression). This part builds the two tools needed for both: **exponentials** and **logarithms**.

Do not move on to the regression chapter's "log transformation" or "logistic regression" material until you can complete every exercise in this part.

## 122. What a Function Really Is

Section 13 already introduced a function as a rule `f: A -> B` mapping every input to exactly one output. Here is the everyday version of the same idea.

Think of a function as a machine. You put a number in. Exactly one number comes out. The same input always produces the same output.

```text
f(x) = x + 10
```

```text
f(5)  = 15
f(0)  = 10
f(-3) = 7
```

The **domain** is the set of allowed inputs. The **range** is the set of outputs that can actually occur.

A function can be drawn as a graph, exactly as in Section 120: put `x` on the horizontal axis, `f(x)` on the vertical axis, and plot the points.

## 123. Exponentials: Repeated Multiplication

Section 117 introduced powers: `2^3 = 2 * 2 * 2 = 8`. An **exponential function** puts the variable in the exponent instead of the base:

```text
f(x) = 2^x
```

```text
f(1) = 2
f(2) = 4
f(3) = 8
f(4) = 16
```

Notice the pattern: each step to the right *multiplies* the previous value, rather than adding a fixed amount. This is the difference between **linear growth** (Part 28: adding a fixed slope each step) and **exponential growth** (multiplying by a fixed factor each step).

```text
Linear:      2, 4, 6, 8, 10        (add 2 each time)
Exponential: 2, 4, 8, 16, 32       (multiply by 2 each time)
```

Exponential growth eventually outpaces linear growth by an enormous margin, no matter how large the linear step is. This is why compound interest, population growth, and viral spread are all modeled exponentially.

A special exponential base, called `e` (approximately 2.71828), appears constantly in probability and statistics because of how it behaves under calculus (Part 0.5 of the Regression guide covers why). The function `f(x) = e^x` is written `exp(x)` in Python and in most formulas in this handbook.

## 124. Logarithms: Undoing an Exponential

A **logarithm** answers the question: "2 raised to what power gives 8?"

```text
2^3 = 8       so       log base 2 of 8 = 3
```

In this handbook, "log" without a stated base almost always means the **natural logarithm**, the log that undoes `exp(x)`:

```text
exp(x) = e^x
log(exp(x)) = x
exp(log(x)) = x    (for x > 0)
```

Worked example: what power of `e` gives 1?

```text
e^0 = 1, so log(1) = 0
```

The logarithm of 1 is always 0, no matter the base — because any number raised to the power 0 equals 1.

**Logarithms are only defined for positive numbers.** You cannot take `log(0)` or `log(-5)`. This single fact is the reason later chapters spend real effort avoiding a probability of exactly 0 or exactly 1 before taking its logarithm (a "numerical safety" clamp, exactly like the one in the autograd mini-project's loss function).

## 125. Log Rules You Will Use Constantly

Three rules turn products and powers, inside a logarithm, into additions and multiplications:

```text
log(a * b) = log(a) + log(b)      (log of a product is a sum of logs)
log(a / b) = log(a) - log(b)      (log of a ratio is a difference of logs)
log(a^c)   = c * log(a)           (log of a power pulls the exponent out front)
```

Worked example using the first rule:

```text
log(2 * 3) = log(6)
log(2) + log(3) should equal log(6)
```

These rules are why maximum-likelihood estimation always works with the *log*-likelihood rather than the likelihood itself: a likelihood is a *product* of many small probabilities (one per data point), and the first rule turns that unwieldy product into a *sum* — sums are far easier to differentiate and far less prone to numerical underflow than a product of many numbers near zero.

## 126. Odds, Log-Odds, and the Logistic Function

A **probability** `p` is a number between 0 and 1. **Odds** compare the chance of an event happening to the chance of it not happening:

```text
odds = p / (1 - p)
```

Worked example: if `p = 0.75` (a 75% chance),

```text
odds = 0.75 / 0.25 = 3
```

This means "3 to 1 in favor" — the event is three times as likely to happen as not.

Odds range from 0 (impossible) to infinity (certain), while probability is capped at 1. This is inconvenient for building a straight-line model. The fix, used throughout logistic regression: take the logarithm of the odds, called the **log-odds** or **logit**:

```text
logit(p) = log(p / (1 - p))
```

The logit can be any real number — negative, zero, or positive — which makes it suitable to model as a straight line `a + bx`, exactly like ordinary regression.

To go back from a logit value to a probability, you undo each step in reverse: exponentiate, then convert odds back to a probability. This reverse function is called the **logistic function** (or **sigmoid**):

```text
p = 1 / (1 + exp(-(a + bx)))
```

You do not need to derive this yet — the Regression guide's logistic regression sections do that in full. This section's job is only to make sure "log," "odds," and "exponential" are no longer new words when you get there.

---

## Functions and Logarithms Practice

### Exercise 1

Compute `2^4` and `sqrt(16)`.

### Solution

```text
2^4 = 2*2*2*2 = 16
sqrt(16) = 4, because 4*4 = 16
```

### Exercise 2

If `p = 0.2`, compute the odds.

### Solution

```text
odds = 0.2 / (1 - 0.2) = 0.2 / 0.8 = 0.25
```

The event is one-quarter as likely to happen as not — i.e. it is much more likely *not* to happen.

### Exercise 3

Using the log rules from Section 125, simplify `log(8) - log(2)` without a calculator, given that `8 = 2 * 2 * 2`.

### Solution

```text
log(8) - log(2) = log(8/2) = log(4)
```

Since `4 = 2^2`, `log(4) = 2 * log(2)`.

### Exercise 4

A quantity doubles every year, starting at 100. Is this linear or exponential growth? Write a function `f(x)` for the quantity after `x` years.

### Solution

This is exponential growth (multiplying by a fixed factor of 2 each year, not adding a fixed amount).

```text
f(x) = 100 * 2^x
```

Check: `f(0) = 100`, `f(1) = 200`, `f(2) = 400` — doubling each year.

### Exercise 5

Why is `log(0)` undefined? Why does this matter for a probability-based loss function?

### Solution

A logarithm answers "what power of the base gives this number?" No power of `e` ever produces exactly 0 — `e^x` gets arbitrarily close to 0 as `x` goes to negative infinity, but never reaches it. So there is no answer to "what power gives 0," and the logarithm is undefined there.

This matters because loss functions like binary cross-entropy compute `log(p)` and `log(1-p)`. If a model ever predicts a probability of *exactly* 0 or 1, one of those terms becomes `log(0)`, which is undefined (in code, it raises an error or produces `-infinity`). This is why implementations clamp predicted probabilities to a tiny distance away from 0 and 1 before taking the log.

---

# Part 30 — Chance and Uncertainty: Probability from Zero

Every model in the Regression and GLM guides makes a probabilistic assumption about how the data was generated. This part builds the vocabulary of chance from nothing, so those assumptions stop being intimidating sentences and start being concrete, checkable claims.

## 127. Events and Sample Spaces

An **experiment** is any process with an uncertain outcome: a coin flip, a customer's decision to buy or not, tomorrow's rainfall.

The **sample space** is the set of every possible outcome. For a coin flip:

```text
sample space = {heads, tails}
```

An **event** is a subset of the sample space — a specific outcome or group of outcomes you care about. "The coin lands heads" is the event `{heads}`.

## 128. The Rules of Probability

A probability is a number between 0 and 1 assigned to an event, where 0 means "never happens" and 1 means "always happens."

Two rules govern how probabilities combine:

**Complement rule.** The probability that an event does *not* happen is 1 minus the probability that it does:

```text
P(not A) = 1 - P(A)
```

**Addition rule (for events that cannot both happen at once).** If a die roll gives you a 1 or a 2, and it cannot show both at once:

```text
P(1 or 2) = P(1) + P(2) = 1/6 + 1/6 = 2/6 = 1/3
```

**Multiplication rule (for independent events — see Section 129).** If two coin flips are independent, the probability that both land heads is:

```text
P(heads and heads) = P(heads) * P(heads) = 0.5 * 0.5 = 0.25
```

## 129. Conditional Probability and Independence

**Conditional probability** asks: "given that we already know one thing happened, what is the probability of another thing?" Written `P(A | B)`, read "probability of A given B."

Worked example. Among 100 emails, 30 are spam. Among the 30 spam emails, 25 contain the word "free." Among the 70 non-spam emails, 5 contain the word "free."

```text
P(spam | contains "free") = ?
```

This asks: out of every email that contains "free" (25 + 5 = 30 of them), how many are spam (25)?

```text
P(spam | "free") = 25 / 30 ≈ 0.83
```

Notice this is very different from `P(spam) = 30/100 = 0.30`. Seeing the word "free" changed our belief substantially — that is exactly what conditional probability measures.

Two events are **independent** if knowing one happened tells you nothing about the other: `P(A | B) = P(A)`. Successive fair coin flips are independent — the coin has no memory of the last flip.

## 130. Bayes' Rule

Bayes' rule lets you flip a conditional probability around — going from `P(B | A)` to `P(A | B)` — which is exactly what is needed when you know how a *cause* produces *evidence*, but want to reason backward from the evidence to the cause.

```text
P(A | B) = P(B | A) * P(A) / P(B)
```

Worked example: a medical test for a rare disease (1% of the population has it) is 99% accurate for people who have the disease, and 95% accurate for people who do not (a 5% false-positive rate). A random person tests positive. What is the chance they actually have the disease?

Most people's intuition says "about 99%." The real answer, using Bayes' rule, is much lower — because the disease is rare, most positive tests come from the huge pool of healthy people who got an unlucky false positive, not from the small pool of actually sick people. This surprising gap between intuition and the correct answer is exactly why Bayes' rule is worth learning carefully rather than approximating in your head — the full worked calculation (with actual counts, not just formulas) appears in this handbook's probability exercises.

## 131. Random Variables, Expectation, and Variance

A **random variable** is a number whose value depends on the outcome of an uncertain experiment. Rolling a die and recording the number shown is a random variable.

The **expectation** (or expected value, or mean) is the long-run average value you would see if you repeated the experiment many times, weighting each possible outcome by its probability:

```text
E[X] = sum over all outcomes of (outcome * its probability)
```

Worked example: a fair six-sided die.

```text
E[X] = 1*(1/6) + 2*(1/6) + 3*(1/6) + 4*(1/6) + 5*(1/6) + 6*(1/6)
     = (1+2+3+4+5+6) / 6
     = 21 / 6 = 3.5
```

Notice the expectation (3.5) is not even a value the die can show — expectation is a long-run average, not a prediction of any single roll.

**Variance** measures how spread out the outcomes are around the expectation:

```text
Var(X) = E[(X - E[X])^2]
```

In plain language: for each possible outcome, measure how far it is from the average, square that distance (so negative and positive distances do not cancel out), and average those squared distances, weighted by probability. A random variable that is almost always close to its average has low variance; one that swings wildly has high variance.

---

## Probability Practice

### Exercise 1

A bag has 4 red balls and 6 blue balls. What is the probability of drawing a red ball?

### Solution

```text
P(red) = 4 / (4 + 6) = 4/10 = 0.4
```

### Exercise 2

Using the complement rule, what is the probability of *not* drawing red in Exercise 1?

### Solution

```text
P(not red) = 1 - 0.4 = 0.6
```

Check directly: `P(blue) = 6/10 = 0.6` ✓

### Exercise 3

Two fair coins are flipped. What is the probability both land tails?

### Solution

The flips are independent, so multiply:

```text
P(tails and tails) = 0.5 * 0.5 = 0.25
```

### Exercise 4

Out of 50 students, 20 study French. Out of those 20, 12 also study Spanish. What is `P(Spanish | French)`?

### Solution

```text
P(Spanish | French) = 12 / 20 = 0.6
```

### Exercise 5

A fair coin is flipped 3 times. Let `X` be the number of heads. List every possible value of `X` and compute `E[X]` using the fact that each of the 8 equally likely outcomes (HHH, HHT, ..., TTT) has probability 1/8.

### Solution

Counting heads in each of the 8 outcomes: HHH=3, HHT=2, HTH=2, THH=2, HTT=1, THT=1, TTH=1, TTT=0.

```text
Values of X: 0 (1 outcome), 1 (3 outcomes), 2 (3 outcomes), 3 (1 outcome)
E[X] = 0*(1/8) + 1*(3/8) + 2*(3/8) + 3*(1/8)
     = (0 + 3 + 6 + 3) / 8 = 12/8 = 1.5
```

This matches the shortcut formula for a fair coin flipped `n` times: `E[X] = n * 0.5 = 3 * 0.5 = 1.5`.

---

# Part 31 — Data and Decisions: Statistical Reasoning from Zero

Probability (Part 30) describes chance going forward: given a known process, what outcomes are likely? Statistics runs the reasoning backward: given the outcomes we actually observed, what can we say about the unknown process that produced them? This is the exact question every regression, every confidence interval, and every p-value in the chapters ahead is trying to answer.

## 132. Population versus Sample

The **population** is the entire group you ultimately want to learn about — every customer, every patient, every possible measurement. The **sample** is the smaller group you actually observed and measured.

```text
Population: all voters in a country
Sample:     the 1,000 voters who were surveyed
```

Almost nothing in applied statistics has access to the whole population. Every number computed from a sample — an average, a proportion, a regression coefficient — is an **estimate** of the corresponding, unknown population quantity, not the population quantity itself.

## 133. Why Repeated Samples Differ: Sampling Variation

Imagine measuring the average height of 30 randomly chosen people from a city, then throwing those 30 back and drawing a fresh random 30. The two averages will almost never be exactly equal, even though both samples came from the exact same population. This is **sampling variation**: any quantity computed from a random sample is itself uncertain, because a different random sample would have given a (slightly) different number.

This is not a flaw in the method — it is an unavoidable consequence of only observing part of the population. The entire machinery of statistical inference (standard errors, confidence intervals, hypothesis tests) exists to *measure and report* this unavoidable uncertainty honestly, rather than to eliminate it.

## 134. The Standard Error and the Central Limit Theorem

The **standard error** of a sample average measures how much that average would typically vary from one random sample to the next, if you could repeat the sampling many times. It is not the spread of the raw data (that is the standard deviation, Section 131's variance, square-rooted) — it is the spread of the *average itself*, across hypothetical repeated samples.

A remarkable fact, called the **Central Limit Theorem**, says: no matter what shape the underlying population data has (skewed, lumpy, anything), the average of a large enough random sample from it tends to follow a symmetric, bell-shaped pattern (the **normal distribution**), centered on the true population average. This is *why* so many statistical formulas — confidence intervals, t-tests — can safely assume a bell-shaped sampling pattern for the average, even when the raw data itself is nothing like bell-shaped.

The standard error shrinks as the sample size grows — specifically, it shrinks in proportion to the square root of the sample size. Quadrupling your sample size only halves the standard error; this is why very precise estimates require disproportionately large samples.

## 135. Confidence Intervals in Plain Language

A single sample average is a single number — a **point estimate**. A **confidence interval** reports a range of plausible values for the true, unknown population quantity, built around that point estimate using the standard error from Section 134.

```text
confidence interval ≈ estimate ± (a multiplier) * (standard error)
```

A "95% confidence interval" does **not** mean "there is a 95% chance the true value is in this specific interval" (the true value is a fixed, unknown number — it either is or is not in any given interval, with no probability about it). It means: if you repeated the entire sampling-and-estimating procedure many times, about 95% of the intervals constructed this way would contain the true value. The 95% is a statement about the *procedure's* long-run reliability, not about this one interval.

## 136. Hypothesis Testing and p-values Without the Jargon

A hypothesis test starts by assuming a specific, "boring" claim is true (called the **null hypothesis** — often "there is no effect" or "no difference"). It then asks: if that boring claim really were true, how surprising would data as extreme as what we actually observed be?

The **p-value** is the answer to that question, expressed as a probability: it is the chance of seeing data at least as extreme as what was observed, *if the null hypothesis were exactly true*. A small p-value means the observed data would be quite surprising under the "no effect" assumption — which is evidence (not proof) against that assumption.

**A p-value is not the probability that the null hypothesis is true.** That is one of the most common misreadings in applied statistics, and the Regression guide's own chapter on hypothesis testing returns to this point in more depth.

## 137. Common Misconceptions, Corrected

```text
Misconception: "A 95% confidence interval has a 95% chance of containing the truth."
Correction:    95% describes the reliability of the *procedure* across many repeats,
               not the probability for this one interval (Section 135).

Misconception: "A small p-value proves the effect is real and important."
Correction:    A small p-value only means the data would be surprising under
               "no effect." It says nothing about how large or practically
               important the effect is — a tiny, unimportant effect can still
               produce a small p-value if the sample is large enough.

Misconception: "A large sample average is the true population average."
Correction:    Every sample average carries sampling variation (Section 133).
               A larger sample shrinks that variation but never eliminates it.

Misconception: "If two things are correlated, one must cause the other."
Correction:    Correlation only says two quantities move together. A third,
               unmeasured factor causing both (a confounder) can produce a
               strong correlation with no causal link at all — see the
               Regression guide's causal-inference sections.
```

---

## Statistics Practice

### Exercise 1

Explain, in one sentence, why a sample average is not the same thing as the population average.

### Solution

The sample average is computed from only part of the population, so it is subject to sampling variation (Section 133) — a different random sample would give a different average, while the true population average is one fixed, unknown number.

### Exercise 2

A poll of 100 people finds 55% support a proposal, with a standard error of 5 percentage points. Roughly, what range of true support values is plausible?

### Solution

Using the approximate rule from Section 135 (estimate plus or minus about 2 standard errors for a 95% interval):

```text
55% ± 2*(5%) = 55% ± 10% = a range of about 45% to 65%
```

### Exercise 3

A study reports "p = 0.001, so the effect is definitely real and important." What is wrong with this statement?

### Solution

Two errors. First, a p-value of 0.001 means the data would be very surprising under "no effect" — that is evidence against the null hypothesis, not a guarantee ("definitely") that the alternative is true. Second, statistical significance (a small p-value) says nothing about practical importance (effect size) — a very large sample can produce a tiny p-value for an effect too small to matter in practice (Section 137).

### Exercise 4

If a sample size quadruples (multiplies by 4), what happens to the standard error, according to Section 134?

### Solution

The standard error shrinks in proportion to the square root of the sample size. `sqrt(4) = 2`, so the standard error is cut in half, not to one-quarter.

### Exercise 5

Ice cream sales and drowning deaths are strongly correlated across months of the year. Using Section 137's language, explain why this does not mean ice cream causes drowning.

### Solution

Both ice cream sales and drowning rates rise and fall with a third, unmeasured factor: hot weather. Hot weather increases both ice cream purchases and swimming (and therefore drowning risk), producing a strong correlation between ice cream and drowning with no causal link between the two directly. This is a textbook confounder.
