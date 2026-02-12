# Systems Programming - Interview Q&A

**How to use**: Questions are formatted as collapsible headings. Try to answer before expanding.

---

## Part 1: Memory Management

### What's the difference between stack and heap?

**Stack**:
- **Automatic memory**: Managed by compiler
- **LIFO structure**: Function calls, local variables
- **Fast**: Simple pointer increment
- **Limited size**: Typically 1-8 MB
- **Scope**: Variables deallocated when function returns

**Heap**:
- **Dynamic memory**: Managed by programmer (malloc/new)
- **Unstructured**: Can allocate/free in any order
- **Slower**: More complex allocation
- **Large size**: Limited by available RAM
- **Scope**: Persists until explicitly freed

**Example** (C++):
```cpp
void example() {
    int stack_var = 10;        // Stack
    int* heap_var = new int(20); // Heap

    delete heap_var;  // Must free
}  // stack_var automatically freed
```

**When to use**:
- **Stack**: Small, short-lived data, known size at compile time
- **Heap**: Large data, dynamic size, needs to persist beyond function

### What's a memory leak?

**Memory leak**: Allocated memory not freed, becomes inaccessible.

**Causes**:
1. **Forgot to free**: Allocated but never called delete/free
2. **Lost pointer**: No reference to allocated memory
3. **Exception before free**: Exception thrown before delete
4. **Circular references**: Objects reference each other (smart pointers)

**Example** (C++):
```cpp
void leak_example() {
    int* data = new int[1000];
    // ... do work ...
    // Oops! Forgot to delete[] data
}  // Memory leaked
```

**Consequences**:
- Available memory decreases over time
- Eventually out of memory
- Program/system crash

**Prevention**:
- **C++**: Use RAII, smart pointers (unique_ptr, shared_ptr)
- **Python**: Garbage collector handles it
- **Tools**: Valgrind, AddressSanitizer

**Detection**:
```bash
valgrind --leak-check=full ./program
```

### What's a pointer?

**Pointer**: Variable storing memory address of another variable.

**Declaration** (C/C++):
```cpp
int x = 10;
int* ptr = &x;  // ptr stores address of x

cout << x;      // 10 (value)
cout << &x;     // 0x7fff... (address)
cout << ptr;    // 0x7fff... (same address)
cout << *ptr;   // 10 (dereference - get value at address)
```

**Operations**:
- `&`: Address-of operator
- `*`: Dereference operator (get value)
- `->`: Access member through pointer

**Pointer arithmetic**:
```cpp
int arr[5] = {1, 2, 3, 4, 5};
int* ptr = arr;

ptr++;      // Move to next int (4 bytes forward)
*(ptr+2);   // Access arr[2]
```

**NULL/nullptr**: Pointer to nothing
```cpp
int* ptr = nullptr;  // C++11
if (ptr) { ... }     // Check if not null
```

### What's a reference (C++)?

**Reference**: Alias for existing variable (C++ concept).

**Declaration**:
```cpp
int x = 10;
int& ref = x;  // ref is alias for x

ref = 20;      // Changes x
cout << x;     // 20
```

**Pointer vs Reference**:

| Feature | Pointer | Reference |
|---------|---------|-----------|
| Nullable | Yes (nullptr) | No |
| Reassignable | Yes | No (fixed at initialization) |
| Syntax | `*ptr`, `ptr->` | Direct variable access |
| Memory | Has own address | Just alias |
| Arithmetic | Yes | No |

**Use reference**:
- Function parameters (avoid copy)
- Return from function
- Operator overloading

```cpp
void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}
```

### What's a dangling pointer?

**Dangling pointer**: Pointer to freed/deleted memory.

**Causes**:
1. **Delete then use**:
```cpp
int* ptr = new int(10);
delete ptr;
*ptr = 20;  // Dangling! Undefined behavior
```

2. **Return address of local**:
```cpp
int* bad_function() {
    int local = 42;
    return &local;  // Dangling! local destroyed when function returns
}
```

3. **Pointer to deleted object**:
```cpp
int* ptr1 = new int(10);
int* ptr2 = ptr1;
delete ptr1;
*ptr2 = 20;  // Dangling! Memory freed
```

**Prevention**:
- Set pointer to nullptr after delete
- Use smart pointers
- Don't return addresses of local variables

**Smart pointer solution**:
```cpp
std::unique_ptr<int> ptr = std::make_unique<int>(10);
// Automatically deleted, can't create dangling
```

### What's RAII?

**RAII** (Resource Acquisition Is Initialization): C++ idiom where resource lifetime tied to object lifetime.

**Principle**:
- Acquire resource in constructor
- Release resource in destructor

**Example** (file handling):
```cpp
class FileHandler {
    FILE* file;
public:
    FileHandler(const char* filename) {
        file = fopen(filename, "r");
    }

    ~FileHandler() {
        if (file) fclose(file);
    }

    // Prevent copying
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
};

// Usage
{
    FileHandler fh("data.txt");
    // ... use file ...
}  // Automatically closed when fh destroyed
```

**Benefits**:
- **Exception safe**: Resource cleaned up even with exception
- **No manual cleanup**: Destructor guarantees cleanup
- **Prevents leaks**: Can't forget to free

**Standard library examples**:
- `std::unique_ptr`, `std::shared_ptr` (memory)
- `std::lock_guard` (mutex)
- `std::fstream` (file)

### What are smart pointers?

**Smart pointers**: RAII wrapper for raw pointers with automatic memory management.

**Types**:

**1. unique_ptr**: Exclusive ownership
```cpp
std::unique_ptr<int> ptr = std::make_unique<int>(42);
// Automatically deleted when ptr goes out of scope

// Cannot copy
auto ptr2 = ptr;  // Error!

// Can move
auto ptr2 = std::move(ptr);  // OK, ptr now nullptr
```

**Use case**: Single owner, most common

**2. shared_ptr**: Shared ownership (reference counting)
```cpp
std::shared_ptr<int> ptr1 = std::make_shared<int>(42);
std::shared_ptr<int> ptr2 = ptr1;  // OK, refcount = 2

ptr1.use_count();  // 2
ptr1.reset();      // refcount = 1
// Deleted when last shared_ptr destroyed
```

**Use case**: Multiple owners

**3. weak_ptr**: Non-owning reference (breaks cycles)
```cpp
std::shared_ptr<int> shared = std::make_shared<int>(42);
std::weak_ptr<int> weak = shared;

// Must lock to access
if (auto locked = weak.lock()) {
    // Use locked (shared_ptr)
}
```

**Use case**: Prevent circular references

**Circular reference problem**:
```cpp
struct Node {
    std::shared_ptr<Node> next;  // Problem! Cycle never freed
};

// Solution: Use weak_ptr
struct Node {
    std::weak_ptr<Node> next;
};
```

---

## Part 2: Operating Systems

### What's a process vs thread?

**Process**:
- **Independent program**: Own memory space
- **Isolated**: Processes don't share memory
- **Heavy**: Expensive to create/switch
- **Communication**: IPC (pipes, sockets, shared memory)

**Thread**:
- **Lightweight process**: Shares memory with parent process
- **Shared memory**: All threads see same data
- **Fast**: Quick to create/switch
- **Communication**: Direct memory access (need synchronization)

**Comparison**:
```
Process:
┌─────────────┐  ┌─────────────┐
│ Process A   │  │ Process B   │
│ ┌─────────┐ │  │ ┌─────────┐ │
│ │ Memory  │ │  │ │ Memory  │ │  Separate memory
│ └─────────┘ │  │ └─────────┘ │
└─────────────┘  └─────────────┘

Thread:
┌─────────────────────┐
│ Process             │
│ ┌─────────────────┐ │
│ │ Shared Memory   │ │  Shared memory
│ └─────────────────┘ │
│ Thread1  Thread2    │
└─────────────────────┘
```

**When to use**:
- **Process**: Isolation needed, different programs
- **Thread**: Parallelism within program, shared data

### What's a race condition?

**Race condition**: Multiple threads access shared data, at least one writes, outcome depends on timing.

**Example**:
```cpp
int counter = 0;

// Thread 1                 // Thread 2
counter++;                  counter++;

// What's counter?
// Could be 1 or 2 (race!)
```

**Why**: `counter++` is three operations:
1. Read counter (e.g., 0)
2. Increment (e.g., 0 + 1 = 1)
3. Write back (e.g., counter = 1)

**Interleaving**:
```
Thread 1: Read (0)
Thread 2: Read (0)
Thread 1: Increment (1)
Thread 2: Increment (1)
Thread 1: Write (1)
Thread 2: Write (1)
Result: 1 (should be 2!)
```

**Fix with mutex**:
```cpp
std::mutex mtx;
int counter = 0;

// Thread function
void increment() {
    std::lock_guard<std::mutex> lock(mtx);
    counter++;
}
```

### What's a mutex?

**Mutex** (Mutual Exclusion): Lock ensuring only one thread accesses resource.

**Operations**:
- **lock()**: Acquire lock (block if locked)
- **unlock()**: Release lock
- **try_lock()**: Try to acquire (non-blocking)

**Example** (C++):
```cpp
#include <mutex>

std::mutex mtx;
int shared_data = 0;

void critical_section() {
    mtx.lock();
    // Critical section - only one thread at a time
    shared_data++;
    mtx.unlock();
}

// Better: RAII with lock_guard
void critical_section_safe() {
    std::lock_guard<std::mutex> lock(mtx);
    shared_data++;
}  // Automatically unlocked
```

**Example** (Python):
```python
from threading import Lock

lock = Lock()
shared_data = 0

def critical_section():
    with lock:
        shared_data += 1
```

**Deadlock**: Two threads each hold lock, waiting for other
```cpp
Thread 1: lock(A) ... waiting for lock(B)
Thread 2: lock(B) ... waiting for lock(A)
// Deadlock!
```

**Prevention**: Always acquire locks in same order.

### What's a semaphore?

**Semaphore**: Synchronization primitive with counter.

**Types**:
1. **Binary semaphore**: Counter 0 or 1 (like mutex)
2. **Counting semaphore**: Counter can be > 1

**Operations**:
- **wait() / P()**: Decrement counter (block if 0)
- **signal() / V()**: Increment counter

**Example** (limited resources):
```cpp
#include <semaphore>

std::counting_semaphore<5> pool(5);  // 5 connections

void use_connection() {
    pool.acquire();  // Wait for available connection
    // Use connection
    pool.release();  // Return to pool
}
```

**Producer-Consumer**:
```cpp
std::counting_semaphore<0> items(0);   // Count items
std::counting_semaphore<N> spaces(N);  // Count spaces

void producer() {
    spaces.acquire();     // Wait for space
    // Produce item
    items.release();      // Signal item available
}

void consumer() {
    items.acquire();      // Wait for item
    // Consume item
    spaces.release();     // Signal space available
}
```

**Mutex vs Semaphore**:
- Mutex: Binary, ownership (locker must unlock)
- Semaphore: Counting, no ownership

### What's deadlock?

**Deadlock**: Two or more processes waiting for each other, none can proceed.

**Four necessary conditions** (Coffman conditions):
1. **Mutual exclusion**: Resource held by at most one process
2. **Hold and wait**: Process holds resource while waiting for another
3. **No preemption**: Resource can't be forcibly taken
4. **Circular wait**: Cycle in resource dependency graph

**Example**:
```cpp
Thread 1:
    lock(A);
    lock(B);  // Waiting...

Thread 2:
    lock(B);
    lock(A);  // Waiting...
// Deadlock!
```

**Prevention**: Break one of four conditions

**1. Avoid hold and wait**: Acquire all locks at once
```cpp
// Atomic lock acquisition
std::lock(mtx1, mtx2);
std::lock_guard<std::mutex> lock1(mtx1, std::adopt_lock);
std::lock_guard<std::mutex> lock2(mtx2, std::adopt_lock);
```

**2. Lock ordering**: Always acquire in same order
```cpp
void transfer(Account& from, Account& to) {
    // Always lock lower ID first
    std::mutex& first = (from.id < to.id) ? from.mtx : to.mtx;
    std::mutex& second = (from.id < to.id) ? to.mtx : from.mtx;

    std::lock_guard<std::mutex> lock1(first);
    std::lock_guard<std::mutex> lock2(second);
    // Transfer
}
```

**3. Timeout**: Try lock with timeout, give up and retry
```cpp
while (!acquired_all_locks) {
    if (mtx1.try_lock_for(100ms)) {
        if (mtx2.try_lock_for(100ms)) {
            // Got both
            break;
        }
        mtx1.unlock();
    }
    std::this_thread::sleep_for(10ms);
}
```

### What's virtual memory?

**Virtual memory**: Abstraction giving each process its own address space.

**Key concepts**:

**1. Virtual address space**: Each process sees 0 to MAX
- Isolated from other processes
- Can be larger than physical RAM

**2. Pages**: Fixed-size blocks (typically 4KB)
- Virtual pages mapped to physical frames

**3. Page table**: Mapping from virtual to physical addresses
- Maintained by OS
- Consulted by MMU (Memory Management Unit)

**Translation**:
```
Virtual Address → [Page Table] → Physical Address
```

**Benefits**:
- **Isolation**: Process can't access others' memory
- **Illusion of large memory**: Use disk as swap
- **Efficient memory use**: Share read-only pages (code)

**Page fault**: Access unmapped page
1. OS loads page from disk
2. Updates page table
3. Resumes instruction

**Example** (Linux process memory layout):
```
High addresses
┌──────────────┐
│ Stack        │ ↓ Grows down
├──────────────┤
│ (unmapped)   │
├──────────────┤
│ Heap         │ ↑ Grows up
├──────────────┤
│ BSS (uninit) │
│ Data (init)  │
│ Text (code)  │
└──────────────┘
Low addresses
```

---

## Part 3: Compilation & Linking

### What's the difference between compiled and interpreted?

**Compiled** (C, C++, Rust):
- **Source → Machine code**: Compiler translates entire program
- **Ahead-of-time**: Compilation before execution
- **Fast execution**: Direct machine code
- **Platform-specific**: Different binary per OS/architecture

**Interpreted** (Python, JavaScript):
- **Source → Bytecode → Execution**: Interpreter executes line-by-line
- **Runtime**: Translation during execution
- **Slower execution**: Overhead of interpretation
- **Platform-independent**: Same source runs anywhere

**JIT Compilation** (Java, C#, modern JavaScript):
- **Hybrid**: Compile to bytecode, then JIT compile hot code to machine code
- **Dynamic optimization**: Optimize based on runtime behavior

**Comparison**:
```
C/C++:
source.cpp → [compiler] → binary → [CPU] → execution

Python:
source.py → [interpreter] → [bytecode] → execution

Java:
source.java → [compiler] → bytecode → [JVM + JIT] → execution
```

### What are the stages of compilation?

**Compilation pipeline**:

**1. Preprocessing**:
- Handle directives (#include, #define)
- Macro expansion
- Conditional compilation (#ifdef)

```cpp
#include <iostream>  // Include header
#define MAX 100      // Define constant

// After preprocessing:
// (iostream contents inserted)
// MAX replaced with 100
```

**2. Compilation**:
- Parse source code
- Generate assembly code
- Check syntax, types

```cpp
int main() {
    int x = 10;
    return 0;
}
```
→ Assembly

**3. Assembly**:
- Convert assembly to machine code
- Generate object file (.o, .obj)

**4. Linking**:
- Combine object files
- Resolve external references
- Generate executable

```
main.o + library.o → [linker] → executable
```

**Commands**:
```bash
# Separate steps
g++ -E source.cpp -o source.i     # Preprocess
g++ -S source.i -o source.s       # Compile to assembly
g++ -c source.s -o source.o       # Assemble
g++ source.o -o program           # Link

# Or all at once
g++ source.cpp -o program
```

### What's static vs dynamic linking?

**Static linking**:
- **Library code copied into executable**
- Large executable
- No runtime dependencies
- Faster startup (no loading)
- Updates require recompilation

```bash
g++ main.cpp -static -o program
```

**Dynamic linking** (shared libraries):
- **Library loaded at runtime**
- Small executable
- Shared among programs (saves memory)
- Can update library without recompiling
- Slower startup (load libraries)

```bash
g++ main.cpp -o program -lmylib
# Links with libmylib.so (Linux) or mylib.dll (Windows)
```

**Check dependencies**:
```bash
ldd program    # Linux
otool -L program  # macOS
```

**Example**:
```
Static:
program (10 MB) = code + library

Dynamic:
program (1 MB) = code
libmylib.so (9 MB) = library (shared)
```

### What's a virtual function?

**Virtual function** (C++): Function in base class overridden by derived class, resolved at runtime (polymorphism).

**Without virtual** (compile-time binding):
```cpp
class Base {
public:
    void foo() { cout << "Base"; }
};

class Derived : public Base {
public:
    void foo() { cout << "Derived"; }
};

Base* ptr = new Derived();
ptr->foo();  // "Base" - calls Base::foo (static type)
```

**With virtual** (runtime binding):
```cpp
class Base {
public:
    virtual void foo() { cout << "Base"; }
};

class Derived : public Base {
public:
    void foo() override { cout << "Derived"; }
};

Base* ptr = new Derived();
ptr->foo();  // "Derived" - calls Derived::foo (dynamic type)
```

**How it works**: vtable (virtual table)
- Each object with virtual functions has vtable pointer
- vtable contains function pointers
- Runtime lookup to call correct function

**Pure virtual** (abstract class):
```cpp
class Abstract {
public:
    virtual void foo() = 0;  // Must override
};
```

**Destructor**: Always make virtual in base class
```cpp
class Base {
public:
    virtual ~Base() {}  // Important!
};
```
Otherwise derived destructor not called → leak!

---

## Part 4: Computer Architecture

### How does computer store floating point numbers?

**IEEE 754 standard**: Represents floating-point numbers in binary.

**Format** (32-bit float):
```
| Sign | Exponent | Mantissa/Fraction |
|  1   |    8     |        23         |
```

**Components**:
1. **Sign bit**: 0 = positive, 1 = negative
2. **Exponent**: 8 bits, biased by 127
3. **Mantissa**: 23 bits, implicit leading 1

**Value calculation**:
$$(-1)^{sign} \times 1.mantissa \times 2^{exponent - 127}$$

**Example** (12.5):
```
12.5 = 1100.1 (binary) = 1.1001 × 2^3

Sign: 0 (positive)
Exponent: 3 + 127 = 130 = 10000010 (binary)
Mantissa: 1001000...0 (23 bits)

Representation: 0 10000010 10010000000000000000000
```

**Special values**:
- **Zero**: Exponent = 0, Mantissa = 0
- **Infinity**: Exponent = 255, Mantissa = 0
- **NaN**: Exponent = 255, Mantissa ≠ 0

**Precision issues**:
```cpp
float a = 0.1 + 0.2;
// a might not exactly equal 0.3
if (abs(a - 0.3) < 1e-7) {  // Use epsilon comparison
    // Close enough
}
```

**Double precision** (64-bit):
- 1 sign, 11 exponent, 52 mantissa
- More precision, larger range

### What's cache and why does it matter?

**Cache**: Fast memory between CPU and RAM.

**Hierarchy**:
```
CPU Registers (fastest, smallest)
    ↓
L1 Cache (~64 KB, ~1 cycle)
    ↓
L2 Cache (~256 KB, ~10 cycles)
    ↓
L3 Cache (~8 MB, ~40 cycles)
    ↓
RAM (~16 GB, ~100 cycles)
    ↓
Disk (~1 TB, ~10,000,000 cycles)
```

**Cache line**: Typically 64 bytes
- Fetch entire line, not single byte
- Spatial locality: Nearby data accessed together

**Why cache matters**:
```cpp
// Cache-friendly (sequential)
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        sum += matrix[i][j];  // Row-major access
    }
}

// Cache-unfriendly (strided)
for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
        sum += matrix[i][j];  // Column-major access (cache misses!)
    }
}
```

**Cache performance** (row-major vs column-major):
- Row-major: One cache miss per line (8 elements)
- Column-major: One cache miss per element
- **10-100x performance difference!**

**Optimizations**:
1. **Spatial locality**: Access nearby memory
2. **Temporal locality**: Reuse recently accessed data
3. **Loop tiling**: Block algorithms for cache
4. **Structure of arrays**: Better than array of structures

### What's pipelining?

**Pipelining**: Overlap execution of multiple instructions.

**Without pipeline**:
```
Instruction 1: Fetch → Decode → Execute → Memory → WriteBack
Instruction 2:                                      Fetch → ...
(5 cycles per instruction)
```

**With pipeline**:
```
Cycle 1:   I1-Fetch
Cycle 2:   I1-Decode    I2-Fetch
Cycle 3:   I1-Execute   I2-Decode    I3-Fetch
Cycle 4:   I1-Memory    I2-Execute   I3-Decode    I4-Fetch
Cycle 5:   I1-WriteBack I2-Memory    I3-Execute   I4-Decode   I5-Fetch
...
(1 instruction completes per cycle after initial latency)
```

**Speedup**: Near N-fold speedup for N-stage pipeline.

**Hazards** (pipeline stalls):

**1. Data hazard**: Instruction depends on previous result
```assembly
ADD R1, R2, R3   # R1 = R2 + R3
SUB R4, R1, R5   # Needs R1 (not ready yet!)
```
**Solution**: Forwarding, stalling

**2. Control hazard**: Branch changes PC
```assembly
BEQ R1, R2, LABEL  # Don't know where to fetch next
```
**Solution**: Branch prediction

**3. Structural hazard**: Resource conflict
**Solution**: Duplicate resources

### What's branch prediction?

**Branch prediction**: Guess which way branch will go to keep pipeline full.

**Problem**: Branch instruction takes 3-4 cycles to resolve
```assembly
if (x > 0) {
    ...
}
```
→ Don't know which instructions to fetch!

**Strategies**:

**1. Static prediction**:
- Always predict not taken
- Or predict backward branches taken (loops)

**2. Dynamic prediction**:
- **1-bit predictor**: Remember last outcome
- **2-bit predictor**: Change prediction after 2 misses
- **Branch history table**: Table of predictions

**Modern CPUs**: Complex predictors (99%+ accuracy)

**Cost of misprediction**:
- Flush pipeline (10-20 cycles wasted)
- Performance impact if unpredictable branches

**Optimization**:
```cpp
// Unpredictable branch (50/50)
if (random() > 0.5) {
    ...
}

// Better: Branchless
int result = (condition) ? value1 : value2;
// Or bitwise tricks
int result = (-condition & value1) | (~-condition & value2);
```

**Profile-guided optimization**: Use runtime data to optimize branches.

---

## Part 5: Concurrency

### What's the difference between concurrency and parallelism?

**Concurrency**: Multiple tasks making progress (interleaved)
- Can be on single core
- About structure/design
- Example: Single-core handling multiple requests

**Parallelism**: Multiple tasks executing simultaneously
- Requires multiple cores
- About execution
- Example: Multi-core processing array in parallel

**Analogy**:
- **Concurrency**: One chef switching between 3 dishes
- **Parallelism**: Three chefs each cooking one dish

**Illustration**:
```
Concurrency (single core):
Task A: █░░█░░█░░
Task B: ░█░░█░░█░
Task C: ░░█░░█░░█
Time → → → → → →

Parallelism (multi-core):
Core 1: █████████
Core 2: █████████
Core 3: █████████
```

**In code**:
```python
# Concurrent (asyncio) - single thread
async def task1():
    await asyncio.sleep(1)

async def task2():
    await asyncio.sleep(1)

# Parallel (multiprocessing) - multiple processes
from multiprocessing import Process

def task1():
    # CPU-bound work

def task2():
    # CPU-bound work

p1 = Process(target=task1)
p2 = Process(target=task2)
```

### What's a thread pool?

**Thread pool**: Reusable set of threads for executing tasks.

**Problem**: Creating thread is expensive
```cpp
for (int i = 0; i < 1000; i++) {
    std::thread t(task, i);  // Create 1000 threads - expensive!
    t.join();
}
```

**Solution**: Create pool once, reuse threads
```cpp
ThreadPool pool(4);  // 4 worker threads

for (int i = 0; i < 1000; i++) {
    pool.enqueue(task, i);  // Add task to queue
}
// 4 threads process tasks from queue
```

**Architecture**:
```
         Task Queue
         ┌────────┐
Tasks →  │ Task 1 │
         │ Task 2 │  → Worker Thread 1 →
         │ Task 3 │  → Worker Thread 2 →  Results
         │  ...   │  → Worker Thread 3 →
         └────────┘  → Worker Thread 4 →
```

**Benefits**:
- Avoid thread creation overhead
- Limit concurrency (prevent resource exhaustion)
- Better CPU utilization

**Example** (C++17):
```cpp
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <condition_variable>

class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;

public:
    ThreadPool(size_t threads) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });

                        if (stop && tasks.empty())
                            return;

                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers)
            worker.join();
    }
};
```

### What's atomic operation?

**Atomic operation**: Indivisible operation that appears instantaneous to other threads.

**Non-atomic** (race condition):
```cpp
int counter = 0;

// Thread 1 and Thread 2 both execute:
counter++;  // Three operations: read, increment, write
// Race condition!
```

**Atomic** (thread-safe):
```cpp
std::atomic<int> counter(0);

// Both threads execute:
counter++;  // Atomic - no race
// Or
counter.fetch_add(1);
```

**How it works**: CPU instructions (e.g., x86 `LOCK INC`)
- Hardware ensures atomicity
- No need for locks (faster)

**Compare-and-swap (CAS)**:
```cpp
std::atomic<int> value(0);

int expected = 0;
int desired = 1;

// Atomic: if (value == expected) value = desired; return success
bool success = value.compare_exchange_strong(expected, desired);
```

**Lock-free data structures**: Use atomics instead of locks
- Faster (no contention)
- No deadlocks
- Complex to implement correctly

**Memory ordering**: Control how operations are reordered
```cpp
counter.store(1, std::memory_order_release);
int x = counter.load(std::memory_order_acquire);
```

### What's the GIL (Python)?

**GIL** (Global Interpreter Lock): Mutex in CPython preventing multiple threads from executing Python bytecode simultaneously.

**Consequence**: Multi-threading doesn't give parallelism for CPU-bound tasks
```python
# CPU-bound - NO speedup from threads
import threading

def cpu_bound():
    total = 0
    for i in range(10**7):
        total += i

threads = [threading.Thread(target=cpu_bound) for _ in range(4)]
# Runs sequentially due to GIL!
```

**Why GIL exists**:
- Simplifies CPython implementation
- Protects shared data structures
- Makes C extensions easier

**Workarounds**:

**1. Multiprocessing** (separate processes):
```python
from multiprocessing import Process

processes = [Process(target=cpu_bound) for _ in range(4)]
# True parallelism!
```

**2. NumPy/C extensions** (release GIL):
```python
import numpy as np
# NumPy releases GIL for operations
arr = np.array([...])
result = arr.sum()  # Can parallelize
```

**3. Asyncio** (I/O-bound):
```python
import asyncio

async def io_bound():
    await asyncio.sleep(1)  # I/O operation
# Good for I/O, not CPU
```

**Threading still useful for**:
- I/O-bound tasks (network, disk)
- GIL released during I/O operations

**Alternative interpreters**: PyPy, Jython (no GIL)
