# Programming Languages - Interview Q&A

**How to use**: Questions are formatted as collapsible headings. Try to answer before expanding.

---


## Table of Contents

- [[#Part 1: Python vs C++]]
  - [[#What's the difference between Python and C++?]]
  - [[#What are the practical implications of these differences?]]
  - [[#Python variables - types and implications?]]
  - [[#Memory leaks - Python vs C++?]]
  - [[#Private variables - Python vs C++?]]
- [[#Part 2: Object-Oriented Programming]]
  - [[#What's object-oriented programming?]]
  - [[#Advantages of OOP?]]
  - [[#When is OOP good? Bad?]]
  - [[#What's inheritance?]]
  - [[#Composition vs inheritance?]]
  - [[#What's polymorphism?]]
  - [[#What's encapsulation?]]
- [[#Part 3: Type Systems]]
  - [[#Static vs dynamic typing?]]
  - [[#Strong vs weak typing?]]
  - [[#What are type hints in Python?]]
  - [[#What's duck typing?]]
- [[#Part 4: Advanced Concepts]]
  - [[#What's a closure?]]
  - [[#What's a decorator?]]
  - [[#What's a generator?]]
  - [[#What's metaprogramming?]]

---

## Part 1: Python vs C++

### What's the difference between Python and C++?

**Python**:
- **Interpreted**: Executed line-by-line
- **Dynamically typed**: Type checked at runtime
- **Automatic memory**: Garbage collector
- **Higher level**: Abstract, simpler syntax
- **Slower execution**: ~10-100x slower than C++
- **Faster development**: Less code, easier to write

**C++**:
- **Compiled**: Translated to machine code
- **Statically typed**: Type checked at compile time
- **Manual memory**: new/delete, RAII
- **Lower level**: Control over hardware
- **Faster execution**: Direct machine code
- **Slower development**: More code, complex syntax

**Example** (same program):
```python
# Python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # Concise recursive call

print(factorial(5))
```

```cpp
// C++
#include <iostream>

int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);  # Explicit type declarations required
}

int main() {
    std::cout << factorial(5) << std::endl;
    return 0;  # Entry point required
}
```

**When to use**:
- **Python**: Rapid prototyping, scripting, data science, web backends
- **C++**: Performance-critical, systems programming, game engines, embedded

### What are the practical implications of these differences?

**Development speed**:
```python
# Python - 5 lines (Concise and readable)
data = [x**2 for x in range(100)]  # List comprehension
result = sum(x for x in data if x % 2 == 0)  # Generator expression for filtering and summing
```

```cpp
// C++ - 15+ lines (Verbose but strict)
#include <vector>
#include <numeric>

std::vector<int> data;
for (int x = 0; x < 100; ++x) {
    data.push_back(x * x);  # Manual loop and push_back
}

// STL algorithm with lambda for summation and filtering
int result = std::accumulate(data.begin(), data.end(), 0,
    [](int sum, int x) { return (x % 2 == 0) ? sum + x : sum; });
```

**Memory management**:
```python
# Python - automatic
data = [1, 2, 3]
# Garbage collector handles memory
```

```cpp
// C++ - manual
int* data = new int[3]{1, 2, 3};
delete[] data;  // Must free!

// Or RAII
std::vector<int> data = {1, 2, 3};  // Automatic cleanup
```

**Performance**:
```python
# Python - slow for loops
total = 0
for i in range(10**7):
    total += i
# ~0.5 seconds
```

```cpp
// C++ - fast loops
long long total = 0;
for (int i = 0; i < 10'000'000; ++i) {
    total += i;
}
// ~0.01 seconds (50x faster!)
```

**Debugging**:
```python
# Python - runtime errors
x = "10"
y = x + 5  # TypeError at runtime
```

```cpp
// C++ - compile-time errors
std::string x = "10";
int y = x + 5;  // Compile error!
```

### Python variables - types and implications?

**Python**: Dynamic typing, variables are references
```python
x = 10          # x refers to int object (name bound to object)
type(x)         # <class 'int'>

x = "hello"     # x now refers to str object (dynamic re-binding)
type(x)         # <class 'str'>

# Everything is an object
y = x           # y refers to same string object as x
y += " world"   # Strings are immutable; creates NEW object "hello world"
# x unchanged: "hello" (y now points to new object)
```

**C++**: Static typing, variables have fixed type
```cpp
int x = 10;         // x is int, cannot change
x = "hello";        // Compile error!

std::string s = "hello";
std::string& ref = s;  // Reference to same object
ref += " world";    // Modifies original
// s is now "hello world"
```

**Implications**:

**1. Type flexibility**:
```python
def process(x):
    return x + 1  # Works with int, float, etc.
```

```cpp
template <typename T>
T process(T x) {
    return x + 1;  // Must use template for generic
}
```

**2. Performance**:
```python
# Python - type checking at runtime (slow)
x = 10
y = x + 5  # Check if + is valid for int
```

```cpp
// C++ - type known at compile time (fast)
int x = 10;
int y = x + 5;  // Direct add instruction
```

**3. Memory**:
```python
# Python - everything on heap, reference counting
x = 10  # Creates int object on heap
```

```cpp
// C++ - can be stack or heap
int x = 10;           // Stack
int* y = new int(10); // Heap
```

### Memory leaks - Python vs C++?

**C++**: Memory leaks possible
```cpp
void leak() {
    int* ptr = new int(42);
    // Forgot to delete - leak! Memory remains allocated
}

void correct() {
    int* ptr = new int(42);
    delete ptr;  // Must manually free heap memory
}

// Better: RAII
void raii() {
    std::unique_ptr<int> ptr = std::make_unique<int>(42);
    // Automatically deleted when ptr goes out of scope
}
```

**Python**: No memory leaks (with garbage collector)
```python
def no_leak():
    data = [1, 2, 3]
    # Garbage collector frees data when function returns (ref count -> 0)

# But can have memory issues:
class Node:
    def __init__(self):
        self.ref = None

a = Node()
b = Node()
a.ref = b
b.ref = a  # Circular reference: a -> b -> a
# Ref count never hits 0, relies on GC cycle detector to clean up
```

**C++ memory issues**:
1. **Leak**: Allocated but never freed
2. **Dangling pointer**: Use after free
3. **Double free**: Free twice

**Python memory issues**:
1. **High memory usage**: Everything is object (overhead)
2. **Circular references**: Delay collection
3. **C extension leaks**: If C code leaks

**Detection**:
```bash
# C++
valgrind ./program

# Python
import tracemalloc
tracemalloc.start()
# ... code ...
snapshot = tracemalloc.take_snapshot()
```

### Private variables - Python vs C++?

**C++**: True privacy with access specifiers
```cpp
class BankAccount {
private:
    double balance;  // Truly private

public:
    BankAccount(double initial) : balance(initial) {}

    void deposit(double amount) {
        balance += amount;
    }

    double getBalance() const {
        return balance;
    }
};

BankAccount account(1000);
// account.balance = 9999;  // Compile error!
account.deposit(100);  // Must use public interface
```

**Python**: No true privacy (convention-based)
```python
class BankAccount:
    def __init__(self, initial):
        self._balance = initial  # Single underscore: convention for "internal use only"

    def deposit(self, amount):
        self._balance += amount

    def get_balance(self):
        return self._balance

account = BankAccount(1000)
account._balance = 9999  # Technically allowed (no strict enforcement), but violates convention
```

**Name mangling** (Python):
```python
class BankAccount:
    def __init__(self, initial):
        self.__balance = initial  # Double underscore triggers name mangling

account = BankAccount(1000)
# account.__balance  # AttributeError: seems private
account._BankAccount__balance = 9999  # Can still access via mangled name (ClassName__variable)
```

**Can you write secure software in Python?**

**Answer**: Yes, but security doesn't come from private variables.

**Security is NOT**:
- Hiding variable names
- Access control in code

**Security IS**:
- Input validation
- Authentication/authorization
- Encryption
- Network security
- Defense in depth

**Example** (secure regardless of language):
```python
# Secure Python
import hashlib

class User:
    def __init__(self, password):
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()

    def check_password(self, password):
        return self.password_hash == hashlib.sha256(password.encode()).hexdigest()

# Even if someone accesses password_hash, password is safe (hashed)
```

**Conclusion**: Language features don't determine security; design and practices do.

---

## Part 2: Object-Oriented Programming

### What's object-oriented programming?

**OOP**: Programming paradigm organizing code into objects (data + methods).

**Four pillars**:

**1. Encapsulation**: Bundle data and methods
```python
class Car:
    def __init__(self, make, model):
        self.make = make
        self.model = model

    def start(self):
        print(f"{self.make} {self.model} started")
```

**2. Abstraction**: Hide complexity, expose interface
```python
# User doesn't need to know engine details
car.start()  # Simple interface
```

**3. Inheritance**: Reuse code from parent class
```python
class ElectricCar(Car):
    def __init__(self, make, model, battery):
        super().__init__(make, model)  # Initialize parent attributes
        self.battery = battery

    def charge(self):
        print("Charging battery")  # New method specific to ElectricCar
```

**4. Polymorphism**: Same interface, different implementations
```python
def start_vehicle(vehicle):
    vehicle.start()  # Works for Car, ElectricCar, Motorcycle, etc.
```

### Advantages of OOP?

**1. Modularity**: Code organized into classes
- Easy to understand and maintain
- Each class handles one responsibility

**2. Reusability**: Inheritance and composition
```python
class Vehicle:
    def move(self):
        pass

class Car(Vehicle):
    def move(self):
        print("Driving")

class Boat(Vehicle):
    def move(self):
        print("Sailing")
```

**3. Flexibility**: Polymorphism
```python
vehicles = [Car(), Boat(), Airplane()]
for v in vehicles:
    v.move()  # Different behavior for each
```

**4. Maintainability**: Changes localized
```python
# Change Car without affecting other code
class Car:
    def move(self):
        print("Driving (now electric!)")
```

**5. Real-world modeling**: Maps to problem domain
```python
class Student:
    def __init__(self, name, grades):
        self.name = name
        self.grades = grades

    def average_grade(self):
        return sum(self.grades) / len(self.grades)
```

### When is OOP good? Bad?

**OOP is good when**:
- ✅ Complex system with many entities
- ✅ Need code reuse (inheritance)
- ✅ Multiple developers (clear interfaces)
- ✅ Long-term maintenance
- ✅ Domain naturally maps to objects

**Examples**:
- GUI applications (Button, Window, Menu)
- Games (Player, Enemy, Item)
- Business applications (Customer, Order, Product)

**OOP is bad when**:
- ❌ Simple scripts (overkill)
- ❌ Performance-critical (overhead)
- ❌ Data transformation pipelines (functional better)
- ❌ Concurrent systems (shared mutable state)

**Examples where functional is better**:
```python
# Data pipeline - functional
data = (
    load_data()
    .filter(lambda x: x > 0)
    .map(lambda x: x * 2)
    .reduce(lambda acc, x: acc + x, 0)
)

# vs OOP (more verbose)
class DataProcessor:
    def __init__(self, data):
        self.data = data

    def filter_positive(self):
        self.data = [x for x in self.data if x > 0]
        return self

    def double(self):
        self.data = [x * 2 for x in self.data]
        return self

    def sum(self):
        return sum(self.data)
```

**Alternatives**:
- **Functional**: Immutable data, pure functions
- **Procedural**: Step-by-step instructions
- **Data-oriented**: Focus on data layout and transformations

### What's inheritance?

**Inheritance**: Derive class from parent, inherit attributes and methods.

**Syntax**:
```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

dog = Dog("Buddy")
dog.speak()  # "Buddy says Woof!"
```

**Types**:

**1. Single inheritance**: One parent
```python
class Child(Parent):
    pass
```

**2. Multiple inheritance**: Multiple parents
```python
class Child(Parent1, Parent2):
    pass
```

**3. Multilevel inheritance**: Chain
```python
class Grandparent:
    pass

class Parent(Grandparent):
    pass

class Child(Parent):
    pass
```

**super()**: Call parent method
```python
class Parent:
    def __init__(self, name):
        self.name = name

class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)  # Call parent init
        self.age = age
```

### Composition vs inheritance?

**Inheritance** (is-a relationship):
```python
class Animal:
    def eat(self):
        print("Eating")

class Dog(Animal):
    def bark(self):
        print("Woof")

# Dog IS-A Animal
```

**Composition** (has-a relationship):
```python
class Engine:
    def start(self):
        print("Engine started")

class Car:
    def __init__(self):
        self.engine = Engine()  # Car HAS-A Engine

    def start(self):
        self.engine.start()
        print("Car started")
```

**When to use**:

**Inheritance**:
- ✅ True is-a relationship
- ✅ Liskov substitution: Child can replace parent
- ✅ Want to reuse code

**Composition**:
- ✅ Has-a relationship
- ✅ More flexible (change components)
- ✅ Avoid deep inheritance hierarchies

**Prefer composition** (general guideline):
```python
# Bad: Inheritance for code reuse
class Stack(list):
    def push(self, item):
        self.append(item)
# Problem: Inherits all list methods (not all make sense)

# Good: Composition
class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        return self._items.pop()
# Only expose Stack interface
```

### What's polymorphism?

**Polymorphism**: Same interface, different implementations.

**Types**:

**1. Subtype polymorphism** (most common):
```python
class Shape:
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

def print_area(shape: Shape):
    print(f"Area: {shape.area()}")

print_area(Circle(5))      # Works
print_area(Rectangle(4, 6)) # Works
```

**2. Duck typing** (Python):
```python
# No inheritance needed!
class Square:
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2

print_area(Square(4))  # Works! "If it walks like a duck..."
```

**3. Operator overloading**:
```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __str__(self):
        return f"({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2  # Calls __add__
print(v3)     # Calls __str__
```

**Benefits**:
- Write generic code
- Extend without modifying existing code
- Open/Closed Principle: Open for extension, closed for modification

### What's encapsulation?

**Encapsulation**: Bundle data and methods, hide internal details.

**Example**:
```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Private

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return True
        return False

    def get_balance(self):
        return self.__balance

account = BankAccount(1000)
account.deposit(500)  # Use public methods
# Can't directly access __balance (name mangled)
```

**Benefits**:
1. **Control access**: Validate inputs
2. **Hide complexity**: Implementation details hidden
3. **Change implementation**: Without breaking code

**Example** (change implementation):
```python
# Version 1
class Account:
    def __init__(self):
        self.balance = 0

# Version 2 - add transaction log
class Account:
    def __init__(self):
        self.__balance = 0
        self.__transactions = []

    def deposit(self, amount):
        self.__balance += amount
        self.__transactions.append(("deposit", amount))

    @property
    def balance(self):
        return self.__balance
# External code unchanged!
```

---

## Part 3: Type Systems

### Static vs dynamic typing?

**Static typing** (C++, Java, Rust):
- Type checked at **compile time**
- Variables have fixed type
- Type annotations required

```cpp
int x = 10;
x = "hello";  // Compile error!
```

**Dynamic typing** (Python, JavaScript, Ruby):
- Type checked at **runtime**
- Variables can change type
- No type annotations (optional in Python 3.5+)

```python
x = 10
x = "hello"  # OK
```

**Comparison**:

| Aspect | Static | Dynamic |
|--------|--------|---------|
| Errors | Compile time | Runtime |
| Speed | Faster (no runtime checks) | Slower |
| Flexibility | Less | More |
| Refactoring | Easier (compiler helps) | Harder |
| Development | Slower (more code) | Faster |

**Gradual typing** (Python with type hints):
```python
def greet(name: str) -> str:
    return f"Hello, {name}"

# Type checker (mypy) catches errors at static analysis time, not runtime
greet(123)  # Error: Argument 1 has incompatible type "int"
```

### Strong vs weak typing?

**Strong typing**: No implicit type coercion (or limited)

**Python** (strongly typed):
```python
"1" + 2  # TypeError: cannot concatenate str and int
"1" + str(2)  # OK: "12"
```

**Weak typing**: Implicit type coercion

**JavaScript** (weakly typed):
```javascript
"1" + 2  // "12" (number coerced to string)
"1" - 2  // -1 (string coerced to number)
```

**C** (weakly typed):
```c
int x = 5;
float y = x;  // Implicit conversion
```

**Confusion**: Static/dynamic ≠ Strong/weak

| Language | Static/Dynamic | Strong/Weak |
|----------|----------------|-------------|
| Python | Dynamic | Strong |
| JavaScript | Dynamic | Weak |
| Java | Static | Strong |
| C | Static | Weak |

### What are type hints in Python?

**Type hints** (Python 3.5+): Optional annotations for type checking.

**Basic types**:
```python
def greet(name: str) -> str:
    return f"Hello, {name}"

age: int = 25
names: list[str] = ["Alice", "Bob"]  # List of strings
scores: dict[str, int] = {"Alice": 90, "Bob": 85}  # Dict mapping string keys to int values
```

**Generics**:
```python
from typing import List, Dict, Tuple, Optional

def process(data: List[int]) -> Dict[str, int]:
    return {"sum": sum(data), "count": len(data)}

def find(arr: List[int], target: int) -> Optional[int]:
    # Returns int if found, otherwise None (Optional[int] is Union[int, None])
    try:
        return arr.index(target)
    except ValueError:
        return None
```

**Union types**:
```python
from typing import Union

def parse(value: Union[int, str]) -> int:
    if isinstance(value, str):
        return int(value)
    return value
```

**Callable**:
```python
from typing import Callable

def apply(func: Callable[[int], int], x: int) -> int:
    return func(x)

apply(lambda x: x * 2, 5)  # 10
```

**Type checking**:
```bash
# Install mypy
pip install mypy

# Check types
mypy script.py
```

**Runtime**: Type hints are **not enforced** at runtime
```python
def add(x: int, y: int) -> int:
    return x + y

add("hello", "world")  # Works at runtime! Returns "helloworld"
```

### What's duck typing?

**Duck typing**: "If it walks like a duck and quacks like a duck, it's a duck."

**Philosophy**: Don't check type, check behavior.

**Example**:
```python
class Duck:
    def quack(self):
        print("Quack!")

class Person:
    def quack(self):
        print("I'm imitating a duck!")

def make_it_quack(thing):
    thing.quack()  # Don't care about type, just needs quack() method

make_it_quack(Duck())    # Works (Duck has quack)
make_it_quack(Person())  # Also works! (Person has quack)
```

**Protocols** (Python 3.8+):
```python
from typing import Protocol

class Quackable(Protocol):
    def quack(self) -> None:
        ...

def make_it_quack(thing: Quackable):
    thing.quack()
```

**Built-in protocols**:
```python
# Iterable: has __iter__
def process(items):
    for item in items:  # Works with list, tuple, set, custom class
        print(item)

# File-like: has read()
def read_data(file):
    return file.read()  # Works with file, StringIO, BytesIO
```

**Advantage**: Flexible, works with any compatible type
**Disadvantage**: Runtime errors if method missing

---

## Part 4: Advanced Concepts

### What's a closure?

**Closure**: Function that captures variables from enclosing scope.

**Example** (Python):
```python
def outer(x):
    def inner(y):
        return x + y  # Captures x from outer scope (closure)
    return inner

add_five = outer(5) # Returns inner function with x=5 trapped
print(add_five(3))  # 8 (uses trapped x=5)
print(add_five(7))  # 12
```

**Use cases**:

**1. Function factories**:
```python
def multiplier(factor):
    def multiply(x):
        return x * factor
    return multiply

times_two = multiplier(2)
times_three = multiplier(3)

print(times_two(5))    # 10
print(times_three(5))  # 15
```

**2. Callbacks**:
```python
def make_button_handler(button_id):
    def handler():
        print(f"Button {button_id} clicked")
    return handler

button1_handler = make_button_handler(1)
button2_handler = make_button_handler(2)
```

**3. Decorators**:
```python
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Done")
        return result
    return wrapper

@logger
def greet(name):
    print(f"Hello, {name}")
```

**C++** (lambda capture):
```cpp
int x = 5;
auto add_x = [x](int y) { return x + y; };  // Capture x by value
auto add_x_ref = [&x](int y) { return x + y; };  // Capture x by reference
```

### What's a decorator?

**Decorator** (Python): Function that modifies another function.

**Syntax**:
```python
def decorator(func):
    def wrapper(*args, **kwargs):
        # Before: code to run before original function
        result = func(*args, **kwargs)
        # After: code to run after original function
        return result
    return wrapper

@decorator
def my_function():
    pass

# Equivalent to:
my_function = decorator(my_function)
```

**Examples**:

**1. Timing**:
```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
```

**2. Authentication**:
```python
def requires_auth(func):
    def wrapper(*args, **kwargs):
        if not is_authenticated():
            raise PermissionError("Not authenticated")
        return func(*args, **kwargs)
    return wrapper

@requires_auth
def sensitive_operation():
    pass
```

**3. Caching**:
```python
def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)  # Compute and store if not cached
        return cache[args]  # Return cached result
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)  # Recursive calls hit cache, making it O(n)
```

**With arguments**:
```python
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}")

greet("Alice")  # Prints 3 times
```

### What's a generator?

**Generator** (Python): Function that yields values lazily (one at a time).

**Syntax**:
```python
def count_up_to(n):
    i = 0
    while i < n:
        yield i
        i += 1

for num in count_up_to(5):
    print(num)  # 0, 1, 2, 3, 4
```

**Difference from function**:
- **Function**: Returns once, exits
- **Generator**: Yields multiple times, pauses and resumes

**Benefits**:

**1. Memory efficient**:
```python
# List (memory inefficient)
def get_numbers(n):
    return [i for i in range(n)]  # Creates entire list in memory

# Generator (memory efficient)
def get_numbers_gen(n):
    for i in range(n):
        yield i  # Yields one value at a time, doesn't store all in memory

# Example
numbers = get_numbers(10**9)  # Crashes: tries to create massive list
numbers_gen = get_numbers_gen(10**9)  # OK: Creates generator object, values generated on-demand
```

**2. Infinite sequences**:
```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b  # Infinite sequence generation

# Take first 10
import itertools
fib = fibonacci()
first_10 = list(itertools.islice(fib, 10))  # Consume only what's needed
```

**3. Pipeline processing**:
```python
def read_file(filename):
    with open(filename) as f:
        for line in f:
            yield line

def filter_lines(lines):
    for line in lines:
        if "error" in line.lower():
            yield line

def process_logs(filename):
    lines = read_file(filename)
    errors = filter_lines(lines)
    for error in errors:
        print(error)
```

**Generator expression**:
```python
# List comprehension
squares_list = [x**2 for x in range(10)]

# Generator expression
squares_gen = (x**2 for x in range(10))
```

### What's metaprogramming?

**Metaprogramming**: Code that manipulates code.

**Python examples**:

**1. Dynamic attribute access**:
```python
class DynamicClass:
    def __getattr__(self, name):
        return f"Accessed {name}"

obj = DynamicClass()
print(obj.anything)  # "Accessed anything"
print(obj.foo)       # "Accessed foo"
```

**2. Class decorators**:
```python
def add_methods(cls):
    cls.new_method = lambda self: "Added dynamically"
    return cls

@add_methods
class MyClass:
    pass

obj = MyClass()
obj.new_method()  # Works!
```

**3. Metaclasses**:
```python
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    pass

db1 = Database()
db2 = Database()
print(db1 is db2)  # True (same instance)
```

**4. Code generation**:
```python
# Generate class dynamically
MyClass = type('MyClass', (object,), {
    'x': 10,
    'method': lambda self: self.x * 2
})

obj = MyClass()
obj.method()  # 20
```

**C++ metaprogramming** (templates):
```cpp
// Compile-time factorial
template<int N>
struct Factorial {
    static const int value = N * Factorial<N-1>::value;
};

template<>
struct Factorial<0> {
    static const int value = 1;
};

int x = Factorial<5>::value;  // Computed at compile time: 120
```

**Use cases**:
- ORMs (database mapping)
- Testing frameworks
- API clients
- DSLs (Domain Specific Languages)
