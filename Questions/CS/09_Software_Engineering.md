# Software Engineering for ML - Interview Q&A

Essential software engineering practices for ML roles: Git, Testing, CI/CD, Code Quality, and Performance Profiling.

---

## Table of Contents

- [[#Part 1: Git & Version Control]]
  - [[#Git Merge vs Git Rebase - When to use which?]]
  - [[#How do you debug with Git Bisect?]]
  - [[#What is Git Cherry-Pick and when is it useful?]]
  - [[#Explain Git Workflows (Feature Branch vs Trunk-Based)]]
- [[#Part 2: Testing & CI/CD]]
  - [[#Unit Tests vs Integration Tests - What's the difference?]]
  - [[#What is Mocking in testing? (unittest.mock)]]
  - [[#Explain the concept of CI/CD pipeline for ML]]
  - [[#What is Property-Based Testing (Hypothesis)?]]
- [[#Part 3: Code Quality & Design]]
  - [[#Duck Typing vs Static Typing (mypy) in Python]]
  - [[#What are Design Patterns relevant to ML? (Factory, Strategy)]]
  - [[#Why use linters (Ruff, Black, Pylint)?]]
- [[#Part 4: Performance Profiling]]
  - [[#How do you profile Python code? (cProfile, line_profiler)]]
  - [[#How to identify Memory Leaks in Python?]]
  - [[#GIL (Global Interpreter Lock) - Impact on ML code?]]
  - [[#Multiprocessing vs Multithreading in Python]]

---

## Part 1: Git & Version Control

### Git Merge vs Git Rebase - When to use which?

*   **Merge**: Combines histories. Preserves the exact history of commits. Creates a "merge commit".
    *   **Use when**: You want to preserve the context of a feature branch merge. Safe for shared branches.
*   **Rebase**: Moves commits to a new base. Rewrites history to be linear.
    *   **Use when**: You want a clean, linear history. Updating a local feature branch with latest `main` changes.
    *   **Danger**: Do not rebase pushed branches shared with others (rewrites history they rely on).

### How do you debug with Git Bisect?

Binary search for the commit that introduced a bug.
1.  `git bisect start`
2.  `git bisect bad` (Current commit is broken)
3.  `git bisect good <commit-hash>` (Last known working commit)
4.  Git checks out a middle commit. You test it.
5.  Run `git bisect good` or `git bisect bad`. Repeat until culprit found.

### What is Git Cherry-Pick and when is it useful?

Applying the changes from a specific commit (hash) onto your current branch.
*   **Use case**: Hot-fixing `main` with a fix from a feature branch without merging the entire unfinished feature.

### Explain Git Workflows (Feature Branch vs Trunk-Based)

*   **Gitflow (Feature Branch)**: `develop` branch + `feature/xyz` branches + `release` branches. Good for releases.
*   **Trunk-Based**: Everyone commits to `main` (trunk) frequently (at least daily). Feature flags hide incomplete features. Good for CI/CD and rapid iteration.

---

## Part 2: Testing & CI/CD

### Unit Tests vs Integration Tests - What's the difference?

*   **Unit Tests**: Test a small, isolated piece of code (function/class) *independently*. Fast, mocked dependencies (DB, API).
    *   *Example*: Test `calculate_loss(y_true, y_pred)`.
*   **Integration Tests**: Test how modules work *together*. Slower, uses real/test DB.
    *   *Example*: Test `train_step()` which calls model, optimizer, and data loader.

### What is Mocking in testing? (unittest.mock)

Replacing real objects (like a Database connection, API call, or S3 download) with fake objects to ensure tests are fast, deterministic, and isolated.
*   **Side Effect**: Configuring a mock to return a planned value or raise an exception to test error handling.

### Explain the concept of CI/CD pipeline for ML

**Continuous Integration (CI)**:
*   Linting (Ruff), Type Checking (mypy).
*   Unit Tests (Pytest).
*   Build Docker Image.

**Continuous Deployment/Delivery (CD)**:
*   Deploy model to Staging (Kubernetes namespace).
*   Run Integration Tests / Smoke Tests.
*   Traffic shifting (Canary deployment) to Production.

**For ML**: Often includes **CT (Continuous Training)** trigger if data drift is detected.

### What is Property-Based Testing (Hypothesis)?

Instead of writing specific examples (`assert add(2, 2) == 4`), you write properties that should hold *for all* valid inputs. The framework generates random inputs to try and break your code.
*   *Example*: `assert decode(encode(x)) == x` for all strings `x`.

---

## Part 3: Code Quality & Design

### Duck Typing vs Static Typing (mypy) in Python

*   **Duck Typing**: "If it walks like a duck and quacks like a duck, it's a duck." Dynamic. Python doesn't check types at compile time. Flexible but error-prone in large codebases.
*   **Static Typing (Type Hints + Mypy)**: Annotating variables (`x: int`). Tools check for type errors *before* running code. Critical for large, robust ML pipelines.

### What are Design Patterns relevant to ML? (Factory, Strategy)

*   **Strategy Pattern**: Swapping algorithms at runtime.
    *   *ML*: Choosing between `Adam`, `SGD`, `RMSprop` optimizers dynamically via config.
*   **Factory Pattern**: Creating objects without specifying exact class.
    *   *ML*: `ModelFactory.get_model("resnet50")` returns a ResNet object.
*   **Observer Pattern**: object notifies others of state changes.
    *   *ML*: `EarlyStopping` callback observing `val_loss`.

### Why use linters (Ruff, Black, Pylint)?

*   **Consistency**: `Black` enforces one formatting style automatically. No bike-shedding in code reviews.
*   **Bug Prevention**: `Ruff`/`Pylint` catch unused variables, mutable default arguments, undefined names.
*   **Maintainability**: Clean code is easier to read and debug.

---

## Part 4: Performance Profiling

### How do you profile Python code? (cProfile, line_profiler)

*   **cProfile**: Built-in, deterministic profiler. Shows total time per function call. Good for high-level bottlenecks.
    *   `python -m cProfile -o output.prof myscript.py`
    *   Visualize with `snakeviz`.
*   **line_profiler**: Shows time spent on *each line* of code. Critical for optimizing inner loops in training (e.g., custom data augmentation).

### How to identify Memory Leaks in Python?

*   **Common Causes**: Storing tensors in a global list (e.g., `losses.append(loss)` instead of `loss.item()`), circular references, C-extensions.
*   **Tools**:
    *   `memory_profiler`: Line-by-line memory usage.
    *   `tracemalloc`: Track allocations.
    *   `objgraph`: Visualize object references (find what's holding onto memory).

### GIL (Global Interpreter Lock) - Impact on ML code?

A mutex that allows only one thread to hold the control of the Python interpreter.
*   **Impact**: CPU-bound Python threads (e.g., complex math in pure Python) *cannot* run in parallel.
*   **ML Workaround**: Most ML libraries (NumPy, PyTorch, TF) release the GIL when doing heavy C++/CUDA operations. So `torch.matmul` runs in parallel.
*   **Data Loading**: Use `multiprocessing` (e.g., `num_workers > 0` in DataLoader) to bypass GIL.

### Multiprocessing vs Multithreading in Python

*   **Multithreading**: Defines threads in same process. Shared memory.
    *   *Good for*: I/O bound tasks (API requests, downloading files).
    *   *Bad for*: CPU bound tasks (due to GIL).
*   **Multiprocessing**: Spawns separate processes. Separate memory space.
    *   *Good for*: CPU bound tasks (Data augmentation, preprocessing).
    *   *Cost*: High overhead to create processes and serialize data (pickle) between them.
