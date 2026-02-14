# Algorithms - Interview Q&A

**How to use**: Questions are formatted as collapsible headings. Try to answer before expanding.

---


## Table of Contents

- [[#Part 1: Sorting Algorithms]]
  - [[#What's the time complexity of common sorting algorithms?]]
  - [[#Explain merge sort]]
  - [[#Explain quick sort]]
  - [[#When to use which sorting algorithm?]]
  - [[#What's a stable sort and why does it matter?]]
- [[#Part 2: Searching Algorithms]]
  - [[#What's binary search?]]
  - [[#How to find first/last occurrence?]]
  - [[#What's the two-pointer technique?]]
- [[#Part 3: Dynamic Programming]]
  - [[#What's dynamic programming?]]
  - [[#Fibonacci - DP example]]
  - [[#What's the knapsack problem?]]
  - [[#What's the longest common subsequence (LCS)?]]
  - [[#How to identify DP problems?]]
- [[#Part 4: Greedy Algorithms]]
  - [[#What's a greedy algorithm?]]
  - [[#Example: Activity selection problem]]
  - [[#Example: Huffman coding]]
- [[#Part 5: Backtracking]]
  - [[#What's backtracking?]]
  - [[#Example: Generate all subsets]]
  - [[#Example: Generate all permutations]]
  - [[#Example: N-Queens]]
- [[#Part 6: Complexity Analysis]]
  - [[#What's Big O notation?]]
  - [[#What's the difference between O, Ω, and Θ?]]
  - [[#How to analyze recursive algorithms?]]
  - [[#What's amortized analysis?]]
  - [[#Space complexity - what counts?]]

---

## Part 1: Sorting Algorithms

### What's the time complexity of common sorting algorithms?

| Algorithm      | Best       | Average    | Worst      | Space  | Stable |
|---------------|------------|------------|------------|--------|--------|
| Bubble Sort   | O(n)       | O(n²)      | O(n²)      | O(1)   | Yes    |
| Selection Sort| O(n²)      | O(n²)      | O(n²)      | O(1)   | No     |
| Insertion Sort| O(n)       | O(n²)      | O(n²)      | O(1)   | Yes    |
| Merge Sort    | O(n log n) | O(n log n) | O(n log n) | O(n)   | Yes    |
| Quick Sort    | O(n log n) | O(n log n) | O(n²)      | O(log n)| No    |
| Heap Sort     | O(n log n) | O(n log n) | O(n log n) | O(1)   | No     |
| Counting Sort | O(n + k)   | O(n + k)   | O(n + k)   | O(k)   | Yes    |
| Radix Sort    | O(nk)      | O(nk)      | O(nk)      | O(n+k) | Yes    |

**Stable**: Maintains relative order of equal elements

### Explain merge sort

**Merge sort**: Divide-and-conquer sorting algorithm.

**Algorithm**:
1. **Divide**: Split array in half recursively
2. **Conquer**: Sort each half
3. **Combine**: Merge sorted halves

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

**Time**: O(n log n) - always
**Space**: O(n) - temporary arrays

**Advantages**:
- Stable
- Predictable (always O(n log n))
- Good for linked lists (no random access needed)

**Disadvantages**:
- O(n) extra space
- Slower than quicksort in practice (overhead)

### Explain quick sort

**Quick sort**: Divide-and-conquer using pivot partitioning.

**Algorithm**:
1. **Choose pivot**: Pick element as pivot
2. **Partition**: Rearrange so smaller left, larger right
3. **Recursively sort** left and right

```python
def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

**Time**:
- Average: O(n log n)
- Worst: O(n²) if always pick min/max as pivot

**Space**: O(log n) call stack

**Advantages**:
- Fast in practice (cache-friendly)
- In-place (O(log n) space)

**Disadvantages**:
- Not stable
- Worst case O(n²)

**Improvements**:
- Random pivot selection
- Median-of-three pivot
- Introsort (switch to heapsort if too deep)

### When to use which sorting algorithm?

**Small arrays** (n < 10-50):
- **Insertion sort**: Simple, fast for small n

**General purpose**:
- **Quick sort**: Default choice (fast average case)
- **Merge sort**: Need stability or consistent O(n log n)
- **Heap sort**: In-place + guaranteed O(n log n)

**Nearly sorted**:
- **Insertion sort**: O(n) for nearly sorted

**Integer keys in small range**:
- **Counting sort**: O(n + k) if k not too large
- **Radix sort**: For multi-digit integers

**External sorting** (data doesn't fit in memory):
- **Merge sort**: Sequential access pattern

**Python/Java default**:
- **Timsort**: Hybrid of merge sort + insertion sort
- Stable, O(n log n), optimized for real-world data

### What's a stable sort and why does it matter?

**Stable sort**: Maintains relative order of equal elements.

**Example**:
```
Input:  [(3, "a"), (1, "b"), (3, "c"), (2, "d")]
Stable: [(1, "b"), (2, "d"), (3, "a"), (3, "c")]  # "a" before "c"
Unstable: [(1, "b"), (2, "d"), (3, "c"), (3, "a")]  # order changed
```

**Why it matters**:
- **Multi-key sorting**: Sort by secondary key, then primary
  - Example: Sort students by name, then by grade
- **Preserving order**: When equal elements have meaningful order

**Stable algorithms**: Merge sort, insertion sort, bubble sort, counting sort

**Unstable**: Quick sort (standard), heap sort, selection sort

**Making unstable stable**: Store original index with each element.

---

## Part 2: Searching Algorithms

### What's binary search?

**Binary search**: Search sorted array by repeatedly halving search space.

**Algorithm**:
1. Compare target with middle element
2. If equal, return
3. If target < middle, search left half
4. If target > middle, search right half

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

**Time**: O(log n)
**Space**: O(1) iterative, O(log n) recursive

**Requirements**: Array must be sorted

**Variants**:
- Find first occurrence
- Find last occurrence
- Find insertion position
- Search in rotated sorted array

### How to find first/last occurrence?

**First occurrence**:
```python
def find_first(arr, target):
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result
```

**Last occurrence**:
```python
def find_last(arr, target):
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result
```

**Key difference**: After finding target, continue searching in desired direction.

### What's the two-pointer technique?

**Two-pointer**: Use two pointers to traverse array/list.

**Patterns**:

**1. Opposite ends** (converging):
```python
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]

        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return []
```

**2. Same direction** (slow/fast):
```python
def remove_duplicates(arr):
    if not arr:
        return 0

    slow = 0

    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]

    return slow + 1
```

**3. Sliding window**:
```python
def max_sum_subarray(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum
```

**Use cases**:
- Two sum (sorted array)
- Remove duplicates
- Container with most water
- Trapping rain water
- Linked list cycle detection

---

## Part 3: Dynamic Programming

### What's dynamic programming?

**Dynamic programming**: Solve complex problems by breaking into simpler subproblems.

**Key properties**:
1. **Optimal substructure**: Optimal solution contains optimal solutions to subproblems
2. **Overlapping subproblems**: Same subproblems solved multiple times

**Approaches**:

**1. Top-down (Memoization)**:
- Recursive with caching
- Solve problem by solving subproblems
- Cache results to avoid recomputation

**2. Bottom-up (Tabulation)**:
- Iterative with table
- Solve smaller subproblems first
- Build up to full solution

### Fibonacci - DP example

**Problem**: Compute nth Fibonacci number.

**Naive recursive** (O(2ⁿ)):
```python
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)
```

**Top-down with memoization** (O(n)):
```python
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]
```

**Bottom-up** (O(n) time, O(n) space):
```python
def fib_dp(n):
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]
```

**Optimized** (O(n) time, O(1) space):
```python
def fib_optimized(n):
    if n <= 1:
        return n

    prev2, prev1 = 0, 1

    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current

    return prev1
```

### What's the knapsack problem?

**0/1 Knapsack**: Given items with weights and values, maximize value without exceeding capacity.

**Problem**:
- n items, each with weight w[i] and value v[i]
- Capacity W
- Maximize total value

**DP solution**:
```python
def knapsack(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(W + 1):
            # Don't take item i
            dp[i][w] = dp[i-1][w]

            # Take item i if fits
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                              dp[i-1][w - weights[i-1]] + values[i-1])

    return dp[n][W]
```

**Time**: O(nW)
**Space**: O(nW) or O(W) with space optimization

**State**: `dp[i][w]` = max value using first i items with capacity w

**Recurrence**:
```
dp[i][w] = max(
    dp[i-1][w],                           # Don't take item i
    dp[i-1][w - weights[i]] + values[i]   # Take item i
)
```

### What's the longest common subsequence (LCS)?

**LCS**: Longest subsequence common to two sequences (not necessarily contiguous).

**Example**:
- "ABCDGH" and "AEDFHR" → LCS = "ADH" (length 3)

**DP solution**:
```python
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]
```

**Time**: O(mn)
**Space**: O(mn) or O(n) with optimization

**State**: `dp[i][j]` = LCS length of text1[0..i-1] and text2[0..j-1]

**Recurrence**:
```
if text1[i] == text2[j]:
    dp[i][j] = dp[i-1][j-1] + 1
else:
    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
```

### How to identify DP problems?

**Signs of DP**:
1. **Optimization**: Maximize/minimize something
2. **Counting**: Count number of ways
3. **Decision**: Yes/no for possibility
4. **Overlapping subproblems**: Same calculation repeated
5. **Optimal substructure**: Optimal solution uses optimal solutions to subproblems

**Pattern recognition**:

**1. Linear DP**: 1D array
- Fibonacci, climbing stairs
- House robber
- Decode ways

**2. 2D DP**: Grid/matrix
- Knapsack
- LCS, edit distance
- Unique paths

**3. State machine DP**: Multiple states
- Stock buy/sell with cooldown
- Paint house

**4. Interval DP**: Subproblems on intervals
- Matrix chain multiplication
- Burst balloons

**Steps to solve**:
1. Define state (dp array meaning)
2. Find recurrence relation
3. Determine base cases
4. Determine iteration order
5. Optimize space if possible

---

## Part 4: Greedy Algorithms

### What's a greedy algorithm?

**Greedy**: Make locally optimal choice at each step, hoping to find global optimum.

**Key property**: **Greedy choice property**
- Locally optimal choice leads to globally optimal solution

**Difference from DP**:
- DP: Consider all choices, build from subproblems
- Greedy: Make one choice, never reconsider

**When greedy works**:
- Problem has greedy choice property
- Problem has optimal substructure

**When greedy fails**:
- Locally optimal ≠ globally optimal
- Example: Coin change with [1, 3, 4] - greedy fails for 6

### Example: Activity selection problem

**Problem**: Select maximum number of non-overlapping activities.

**Greedy strategy**: Always pick activity with earliest end time.

```python
def activity_selection(start, end):
    n = len(start)

    # Sort by end time
    activities = sorted(zip(start, end), key=lambda x: x[1])

    count = 1
    last_end = activities[0][1]

    for i in range(1, n):
        if activities[i][0] >= last_end:
            count += 1
            last_end = activities[i][1]

    return count
```

**Why greedy works**: Picking earliest end time leaves most room for future activities.

**Time**: O(n log n) for sorting
**Space**: O(1) or O(n) for sorting

### Example: Huffman coding

**Huffman coding**: Variable-length encoding based on frequency.

**Algorithm**:
1. Build frequency table
2. Create min-heap of nodes
3. Repeatedly merge two minimum nodes
4. Assign 0/1 to left/right

```python
import heapq

class Node:
    def __init__(self, freq, char=None, left=None, right=None):
        self.freq = freq
        self.char = char
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(text):
    # Build frequency table
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1

    # Create heap
    heap = [Node(f, c) for c, f in freq.items()]
    heapq.heapify(heap)

    # Build tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    # Generate codes
    root = heap[0]
    codes = {}

    def generate_codes(node, code):
        if node.char:
            codes[node.char] = code
            return
        generate_codes(node.left, code + "0")
        generate_codes(node.right, code + "1")

    generate_codes(root, "")
    return codes
```

**Time**: O(n log n)
**Space**: O(n)

**Optimal**: Huffman generates optimal prefix-free code.

---

## Part 5: Backtracking

### What's backtracking?

**Backtracking**: Explore all possible solutions by incrementally building candidates and abandoning ("backtracking") when candidate cannot lead to solution.

**Template**:
```python
def backtrack(candidate):
    if is_solution(candidate):
        output(candidate)
        return

    for next_candidate in generate_candidates(candidate):
        if is_valid(next_candidate):
            make_move(next_candidate)
            backtrack(next_candidate)
            undo_move(next_candidate)
```

**Key steps**:
1. **Choose**: Select a possibility
2. **Explore**: Recursively explore
3. **Unchoose**: Backtrack (undo choice)

**Use cases**:
- N-Queens
- Sudoku solver
- Subset generation
- Permutations/combinations
- Path finding

### Example: Generate all subsets

**Problem**: Generate all subsets of a set.

```python
def subsets(nums):
    result = []

    def backtrack(start, path):
        result.append(path[:])

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

**Time**: O(2ⁿ × n) - 2ⁿ subsets, O(n) to copy each
**Space**: O(n) recursion depth

### Example: Generate all permutations

**Problem**: Generate all permutations of array.

```python
def permute(nums):
    result = []

    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return

        for num in nums:
            if num in path:
                continue
            path.append(num)
            backtrack(path)
            path.pop()

    backtrack([])
    return result
```

**Optimized** (with swapping):
```python
def permute_swap(nums):
    result = []

    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return

        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]

    backtrack(0)
    return result
```

**Time**: O(n! × n)
**Space**: O(n) recursion

### Example: N-Queens

**Problem**: Place N queens on N×N chessboard so none attack each other.

```python
def solve_n_queens(n):
    result = []
    board = [['.'] * n for _ in range(n)]

    def is_valid(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        # Check diagonal
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1

        # Check anti-diagonal
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1

        return True

    def backtrack(row):
        if row == n:
            result.append([''.join(row) for row in board])
            return

        for col in range(n):
            if is_valid(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'

    backtrack(0)
    return result
```

**Time**: O(n!)
**Space**: O(n²)

---

## Part 6: Complexity Analysis

### What's Big O notation?

**Big O**: Upper bound on growth rate of algorithm.

**Formal definition**: f(n) = O(g(n)) if ∃ constants c, n₀ such that:
$$f(n) ≤ c \cdot g(n) \text{ for all } n ≥ n₀$$

**Common complexities** (from best to worst):
1. **O(1)**: Constant - array access, hash table lookup
2. **O(log n)**: Logarithmic - binary search, balanced tree
3. **O(n)**: Linear - linear search, array traversal
4. **O(n log n)**: Linearithmic - merge sort, heap sort
5. **O(n²)**: Quadratic - nested loops, bubble sort
6. **O(n³)**: Cubic - matrix multiplication
7. **O(2ⁿ)**: Exponential - fibonacci (naive), subset generation
8. **O(n!)**: Factorial - permutations, TSP brute force

**Ignore**:
- Constants: O(2n) = O(n)
- Lower order terms: O(n² + n) = O(n²)

### What's the difference between O, Ω, and Θ?

**Big O (O)**: Upper bound (worst case)
- f(n) ≤ c·g(n)
- "At most"

**Big Omega (Ω)**: Lower bound (best case)
- f(n) ≥ c·g(n)
- "At least"

**Big Theta (Θ)**: Tight bound (average case)
- c₁·g(n) ≤ f(n) ≤ c₂·g(n)
- "Exactly"

**Example** (binary search):
- Best case: O(1) - find immediately
- Worst case: O(log n) - search entire tree
- Average case: Θ(log n)

**In practice**: Usually discuss Big O (worst case).

### How to analyze recursive algorithms?

**Method 1: Recurrence relations**

**Example** (merge sort):
```
T(n) = 2T(n/2) + O(n)
```
- 2T(n/2): Two recursive calls on half
- O(n): Merge step

**Solve using Master Theorem**:
```
T(n) = aT(n/b) + f(n)
```

**Example** (binary search):
```
T(n) = T(n/2) + O(1)
→ T(n) = O(log n)
```

**Method 2: Recursion tree**

Visualize recursive calls as tree:
- Nodes: Work at each call
- Height: Recursion depth
- Sum all nodes: Total work

**Example** (fibonacci):
```
         fib(n)
        /      \
    fib(n-1)   fib(n-2)
   /    \      /    \
  ...  ...    ...  ...
```
- Height: n
- Work per level: doubles each level
- Total: O(2ⁿ)

### What's amortized analysis?

**Amortized analysis**: Average time per operation over sequence of operations.

**Example** (dynamic array append):
- **Single append**: O(1) if no resize, O(n) if resize
- **n appends**: Total O(n), so **amortized O(1)** per append

**Why**: Resize at sizes 1, 2, 4, 8, 16, ...
- Copy costs: 1 + 2 + 4 + 8 + ... + n = 2n
- Total for n appends: n + 2n = O(n)
- Amortized per append: O(1)

**Methods**:
1. **Aggregate**: Total cost / # operations
2. **Accounting**: Assign credits, some ops prepay for future
3. **Potential**: Define potential function

**Other examples**:
- Stack with multi-pop
- Union-find with path compression
- Splay trees

### Space complexity - what counts?

**Space complexity**: Extra memory used by algorithm (beyond input).

**Count**:
1. **Variables**: Local variables, pointers
2. **Data structures**: Arrays, hash tables, etc.
3. **Recursion**: Call stack depth
4. **Output**: Sometimes (depends on problem)

**Don't count**: Input size (usually)

**Examples**:

**O(1)** space:
```python
def sum_array(arr):
    total = 0
    for x in arr:
        total += x
    return total
```
Only variable `total`.

**O(n)** space:
```python
def reverse(arr):
    return arr[::-1]
```
Creates new array.

**O(log n)** space:
```python
def binary_search_recursive(arr, target, left, right):
    if left > right:
        return -1
    mid = (left + right) // 2
    ...
    return binary_search_recursive(arr, target, ...)
```
Recursion depth = O(log n).

**O(n)** space:
```python
def fibonacci_dp(n):
    dp = [0] * (n + 1)
    ...
```
Array of size n.
