# Data Structures - Interview Q&A

**How to use**: Questions are formatted as collapsible headings. Try to answer before expanding.

---


## Table of Contents

- [[#Part 1: Arrays & Strings]]
  - [[#What's the time complexity of array access?]]
  - [[#What's the difference between array and ArrayList/vector?]]
  - [[#How do you reverse a string in-place?]]
  - [[#What's a sliding window algorithm?]]
- [[#Part 2: Linked Lists]]
  - [[#What's a linked list?]]
  - [[#Array vs Linked List - when to use each?]]
  - [[#How to detect a cycle in a linked list?]]
  - [[#How to find the middle of a linked list?]]
  - [[#How to reverse a linked list?]]
  - [[#What's a dummy node and when to use it?]]
- [[#Part 3: Stacks & Queues]]
  - [[#What's a stack? Applications?]]
  - [[#What's a queue? Applications?]]
  - [[#What's a monotonic stack?]]
  - [[#Implement a min-stack (getMin in O(1))]]
  - [[#Implement a queue using stacks]]
- [[#Part 4: Trees]]
  - [[#What's a binary tree?]]
  - [[#What's a BST (Binary Search Tree)?]]
  - [[#DFS vs BFS for trees?]]
  - [[#What's tree traversal complexity?]]
  - [[#What's a balanced tree?]]
  - [[#What's the lowest common ancestor (LCA)?]]
- [[#Part 5: Heaps & Priority Queues]]
  - [[#What's a heap?]]
  - [[#When to use a heap?]]
  - [[#What's heapify? Time complexity?]]
  - [[#Implement heap sort]]
- [[#Part 6: Hash Tables]]
  - [[#What's a hash table?]]
  - [[#What's a hash function?]]
  - [[#What are collisions? Resolution strategies?]]
  - [[#When does hash table degrade to O(n)?]]
  - [[#What's rehashing?]]
- [[#Part 7: Graphs]]
  - [[#What's a graph?]]
  - [[#What's graph traversal (DFS vs BFS)?]]
  - [[#How to detect cycle in graph?]]
  - [[#What's topological sort?]]
  - [[#What's Dijkstra's algorithm?]]
  - [[#What's union-find (disjoint set)?]]
  - [[#What's minimum spanning tree (MST)?]]
- [[#Part 8: Tries]]
  - [[#What's a trie?]]
  - [[#Implement a trie]]
  - [[#When to use trie vs hash table?]]

---

## Part 1: Arrays & Strings

### What's the time complexity of array access?

**Access by index**: O(1) constant time
- Direct memory address calculation: `base_address + (index × element_size)`
- No iteration needed

**Why O(1)**:
- Arrays stored in contiguous memory
- Can compute any element's address directly

**Example**:
```python
arr = [10, 20, 30, 40]
x = arr[2]  # O(1) - directly access index 2
```

### What's the difference between array and ArrayList/vector?

**Fixed array**:
- Static size (defined at creation)
- No resize operation
- Lower memory overhead

**Dynamic array** (ArrayList/vector):
- Dynamic size (grows automatically)
- Amortized O(1) append
- Resize: allocate new array, copy elements

**Resize strategy**: Double size when full
- If capacity = n, resize to 2n
- Amortized O(1) append (expensive occasionally, cheap usually)

### How do you reverse a string in-place?

**Two-pointer approach**:
```python
def reverse_string(s):
    left, right = 0, len(s) - 1
    s = list(s)  # strings immutable in Python

    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1

    return ''.join(s)
```

**Time**: O(n)
**Space**: O(1) if array is mutable, O(n) for string (immutable)

### What's a sliding window algorithm?

**Technique**: Maintain window of elements, slide through array.

**Use cases**:
- Subarray sum problems
- Longest substring without repeating characters
- Maximum in sliding window

**Example** (max sum subarray of size k):
```python
def max_sum_subarray(arr, k):
    n = len(arr)
    if n < k:
        return -1

    # Compute sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # Slide window
    for i in range(k, n):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum
```

**Time**: O(n) - single pass
**Space**: O(1)

---

## Part 2: Linked Lists

### What's a linked list?

**Linked list**: Sequence of nodes, each containing data and pointer to next node.

**Node structure**:
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None  # Pointer to next node
```

**Types**:
1. **Singly linked**: One pointer (next)
2. **Doubly linked**: Two pointers (next, prev)
3. **Circular**: Last node points to first

### Array vs Linked List - when to use each?

**Array**:
- ✅ Random access: O(1)
- ✅ Cache-friendly (contiguous memory)
- ✅ Less memory overhead
- ❌ Fixed size (or expensive resize)
- ❌ Insert/delete middle: O(n)

**Linked List**:
- ✅ Dynamic size
- ✅ Insert/delete at known position: O(1)
- ❌ No random access: O(n)
- ❌ More memory (pointers)
- ❌ Cache-unfriendly

**Use array**: Random access frequent, size known
**Use linked list**: Many insertions/deletions, size varies

### How to detect a cycle in a linked list?

**Floyd's cycle detection** (tortoise and hare):
```python
def has_cycle(head):
    if not head:
        return False

    slow = fast = head

    # Floyd's Cycle-Finding Algorithm (Tortoise and Hare)
    while fast and fast.next:
        slow = slow.next          # Move 1 step
        fast = fast.next.next     # Move 2 steps

        if slow == fast:          # If they meet, there's a cycle
            return True

    return False  # Fast reached end, no cycle
```

**How it works**:
- Slow moves 1 step, fast moves 2 steps
- If cycle exists, they'll meet
- If no cycle, fast reaches end

**Time**: O(n)
**Space**: O(1)

### How to find the middle of a linked list?

**Two-pointer approach**:
```python
def find_middle(head):
    slow = fast = head

    # Fast pointer moves 2x speed, slow moves 1x speed
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow  # When fast reaches end, slow is at middle
```

**Why it works**:
- Fast moves twice as fast as slow
- When fast reaches end, slow is at middle

**Time**: O(n)
**Space**: O(1)

### How to reverse a linked list?

**Iterative approach**:
```python
def reverse_list(head):
    prev = None
    curr = head

    while curr:
        next_temp = curr.next  # Save next node
        curr.next = prev       # Reverse pointer
        prev = curr            # Move prev forward
        curr = next_temp       # Move curr forward

    return prev  # New head of reversed list
```

**Steps**:
1. Save next node
2. Reverse current pointer
3. Move prev and curr forward

**Time**: O(n)
**Space**: O(1)

**Recursive** (O(n) space due to call stack):
```python
def reverse_recursive(head):
    # Base case: empty list or single node
    if not head or not head.next:
        return head

    # Recursive step: reverse rest of list
    new_head = reverse_recursive(head.next)
    
    # Reverse current node's pointer
    head.next.next = head
    head.next = None

    return new_head
```

### What's a dummy node and when to use it?

**Dummy node**: Placeholder node before head to simplify edge cases.

**Use when**:
- Head might change (insertions/deletions at start)
- Simplifies code (no special case for head)

**Example** (remove elements):
```python
def remove_elements(head, val):
    dummy = Node(0)      # Create dummy node
    dummy.next = head    # Point dummy next to head

    curr = dummy
    while curr.next:
        if curr.next.val == val:
            # Skip the node with value val
            curr.next = curr.next.next
        else:
            # Move to next node
            curr = curr.next

    return dummy.next    # Return actual head (skipping dummy)
```

**Advantage**: No special handling for removing head.

---

## Part 3: Stacks & Queues

### What's a stack? Applications?

**Stack**: LIFO (Last In, First Out) data structure.

**Operations**:
- `push(x)`: Add to top - O(1)
- `pop()`: Remove from top - O(1)
- `peek()`: View top - O(1)

**Applications**:
1. **Function call stack**: Recursion, activation records
2. **Undo mechanism**: Text editors
3. **Expression evaluation**: Infix to postfix
4. **Backtracking**: DFS, maze solving
5. **Parentheses matching**: Compiler parsing

**Implementation**:
```python
# Using list
stack = []
stack.append(1)  # push operation
x = stack.pop()  # pop operation (removes last element)

# Or using deque (more efficient, O(1) from ends)
from collections import deque
stack = deque()
```

### What's a queue? Applications?

**Queue**: FIFO (First In, First Out) data structure.

**Operations**:
- `enqueue(x)`: Add to rear - O(1)
- `dequeue()`: Remove from front - O(1)
- `front()`: View front - O(1)

**Applications**:
1. **BFS traversal**: Level-order tree traversal
2. **Task scheduling**: OS process scheduling
3. **Buffering**: IO buffers, printer queue
4. **Cache**: LRU cache with queue
5. **Async processing**: Message queues

**Implementation**:
```python
from collections import deque
queue = deque()
queue.append(1)      # enqueue (add to right)
x = queue.popleft()  # dequeue (remove from left)
```

**Don't use list**: `list.pop(0)` is O(n)

### What's a monotonic stack?

**Monotonic stack**: Stack maintaining elements in monotonic order (increasing or decreasing).

**Use cases**:
- Next greater element
- Stock span problem
- Largest rectangle in histogram

**Example** (next greater element):
```python
def next_greater(arr):
    n = len(arr)
    result = [-1] * n  # Initialize result with -1
    stack = []         # Monotonic decreasing stack (stores indices)

    for i in range(n):
        # While stack is not empty and current element is greater than element at stack top
        while stack and arr[stack[-1]] < arr[i]:
            idx = stack.pop()   # Current element is next greater for this popped index
            result[idx] = arr[i]
        stack.append(i)         # Push current index onto stack

    return result
```

**Core Idea**: Maintain a specific order (invariant) in the stack.
- **Increasing Stack**: Keep elements in increasing order. Good for finding "next smaller element".
- **Decreasing Stack**: Keep elements in decreasing order. Good for finding "next greater element".

**Logic**:
1. When a new element `x` arrives, compare it with stack top.
2. If `x` violates the invariant (e.g., `x > top` for decreasing stack), **pop** the top.
3. The element `x` is the "next greater element" for the popped item.
4. Repeat pop until invariant is restored, then push `x`.

**Complexity Analysis**:
- **Time**: O(N). Although there's a while loop, each element is pushed exactly once and popped at most once. Amortized O(1) per step.
- **Space**: O(N) worst case.

### Implement a min-stack (getMin in O(1))

**Problem**: Stack with O(1) minimum retrieval.

**Solution 1** (two stacks):
```python
class MinStack:
    def __init__(self):
        self.stack = []      # Main stack stores all elements
        self.min_stack = []  # Auxiliary stack stores minimums

    def push(self, x):
        self.stack.append(x)
        # Only push to min_stack if x is new minimum (or equal to current min)
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)

    def pop(self):
        # If popping minimum element, remove from min_stack too
        if self.stack[-1] == self.min_stack[-1]:
            self.min_stack.pop()
        self.stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]  # Top of min_stack is current minimum
```

**Key Idea (Auxiliary Stack)**:
- We need a history of minimums. If we pop the current minimum, we must revert to the previous one.
- **Invariant**: `min_stack` top is always the minimum of all elements currently in `stack`.
- **Optimization**: Only push to `min_stack` if the new value is `<= current_min`. When popping `stack`, only pop `min_stack` if values match.

**Complexity**:
- **Time**: O(1) for all operations (push, pop, top, getMin).
- **Space**: O(N) worst case (e.g., sorting: 5, 4, 3, 2, 1 -> duplicates main stack). Best case O(1) (e.g., 1, 2, 3, 4, 5 -> min stack only has '1').

**Solution 2** (store differences):
- More space-efficient but complex
- Store (x - min) in stack

### Implement a queue using stacks

**Two-stack approach**:
```python
class QueueUsingStacks:
    def __init__(self):
        self.input_stack = []   # Push new elements here
        self.output_stack = []  # Pop/peek from here

    def enqueue(self, x):
        self.input_stack.append(x)  # Always push to input stack

    def dequeue(self):
        # Move elements only if output stack is empty
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())
        return self.output_stack.pop()

    def front(self):
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())
        return self.output_stack[-1]
```

**Logic (Two Stacks)**:
- **Input Stack (`s1`)**: Acts as an inbox. Always push here.
- **Output Stack (`s2`)**: Acts as an outbox. Always pop/peek from here.
- **Transfer**: When `s2` is empty and we need to pop/peek, move **all** elements from `s1` to `s2`. This reverses the order (LIFO + LIFO = FIFO).

**Complexity Analysis**:
- **Enqueue**: O(1).
- **Dequeue**:
  - **Worst case**: O(N) (when `s2` is empty, move N items).
  - **Amortized**: O(1).
  - **Proof**: Each element is pushed to `s1` (1 op), popped from `s1` (1 op), pushed to `s2` (1 op), and popped from `s2` (1 op). Total ~4 operations per element over its lifetime. Thus, average is constant.

---

## Part 4: Trees

### What's a binary tree?

**Binary tree**: Tree where each node has at most 2 children (left, right).

**Node structure**:
```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
```

**Properties**:
- **Height**: Longest path from root to leaf
- **Depth**: Distance from root to node
- **Level**: Depth + 1

**Types**:
- **Full**: Every node has 0 or 2 children
- **Complete**: All levels filled except possibly last (left-filled)
- **Perfect**: All internal nodes have 2 children, all leaves same level

### What's a BST (Binary Search Tree)?

**BST property**: For each node:
- Left subtree: all values < node.val
- Right subtree: all values > node.val

**Operations** (average case):
- Search: O(log n)
- Insert: O(log n)
- Delete: O(log n)

**Worst case**: O(n) for skewed tree (linked list)

**Search**:
```python
def search(root, val):
    # Base case: root is None or value found
    if not root or root.val == val:
        return root

    # Recursive step: search left or right subtree
    if val < root.val:
        return search(root.left, val)
    else:
        return search(root.right, val)
```

### DFS vs BFS for trees?

**DFS** (Depth-First Search):
- Traverse deep before wide
- Uses stack (or recursion)
- **Three orders**:
  - Inorder: Left → Root → Right (BST: sorted)
  - Preorder: Root → Left → Right
  - Postorder: Left → Right → Root

**BFS** (Breadth-First Search):
- Level-order traversal
- Uses queue
- Process all nodes at level k before k+1

**DFS implementation**:
```python
def inorder(root):
    if not root:
        return
    inorder(root.left)   # Visit left subtree
    print(root.val)      # Process current node
    inorder(root.right)  # Visit right subtree
```

**BFS implementation**:
```python
from collections import deque

def level_order(root):
    if not root:
        return

    queue = deque([root])  # Initialize queue with root
    while queue:
        node = queue.popleft()  # Dequeue front node
        print(node.val)         # Process node

        # Enqueue children if exist
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
```

### What's tree traversal complexity?

**All traversals**: O(n) time, visit each node once

**Space complexity**:
- **Recursive DFS**: O(h) call stack (h = height)
  - Best: O(log n) for balanced tree
  - Worst: O(n) for skewed tree
- **Iterative DFS**: O(h) explicit stack
- **BFS**: O(w) queue (w = max width)
  - Complete tree: O(n) at last level

### What's a balanced tree?

**Balanced tree**: Height is O(log n).

**Definition**: For each node, |height(left) - height(right)| ≤ 1

**Examples**:
- AVL tree: Strictly balanced (balance factor ≤ 1)
- Red-black tree: Relaxed balance (black-height balanced)

**Why important**: Guarantees O(log n) operations.

**Check if balanced**:
```python
def is_balanced(root):
    def height(node):
        if not node:
            return 0  # Height of None is 0

        # Recursively get height of left subtree
        left_h = height(node.left)
        if left_h == -1: return -1  # Propagate imbalance

        # Recursively get height of right subtree
        right_h = height(node.right)
        if right_h == -1: return -1

        # Check balance condition for current node
        if abs(left_h - right_h) > 1:
            return -1  # Imbalanced

        # Return height of this subtree
        return max(left_h, right_h) + 1

    return height(root) != -1
```

### What's the lowest common ancestor (LCA)?

**LCA**: Lowest node that has both p and q as descendants.

**For BST** (simpler):
```python
def lca_bst(root, p, q):
    while root:
        # If both values are smaller, LCA is in left subtree
        if p.val < root.val and q.val < root.val:
            root = root.left
        # If both values are larger, LCA is in right subtree
        elif p.val > root.val and q.val > root.val:
            root = root.right
        # Otherwise, current node is the split point (LCA)
        else:
            return root
```

**For general binary tree**:
```python
def lca_binary_tree(root, p, q):
    # Base case: If root is None, or root is one of the target nodes, it's the LCA
    if not root or root == p or root == q:
        return root

    # Search in left and right subtrees
    left = lca_binary_tree(root.left, p, q)
    right = lca_binary_tree(root.right, p, q)

    # If both return non-None, p and q are in different branches, so root is LCA
    if left and right:
        return root
    
    # Otherwise return the non-None branch (or None if neither found)
    return left if left else right
```

**Time**: O(n)
**Space**: O(h) recursion

---

## Part 5: Heaps & Priority Queues

### What's a heap?

**Heap**: Complete binary tree with heap property.

**Types**:
1. **Min-heap**: Parent ≤ children (min at root)
2. **Max-heap**: Parent ≥ children (max at root)

**Properties**:
- Complete binary tree (filled left to right)
- Height: O(log n)
- Array representation: `left_child = 2i+1, right_child = 2i+2, parent = (i-1)//2`

**Operations**:
- Insert: O(log n) - bubble up
- Extract min/max: O(log n) - bubble down
- Peek min/max: O(1)
- Heapify array: O(n)

### When to use a heap?

**Use cases**:
1. **Priority queue**: Tasks with priorities
2. **Top K elements**: K largest/smallest elements
3. **Median maintenance**: Running median
4. **Merge K sorted arrays**: K-way merge
5. **Dijkstra's algorithm**: Shortest path
6. **Huffman coding**: Data compression

**Example** (K largest elements):
```python
import heapq

# Min-heap (default in Python)
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 4)

min_val = heapq.heappop(heap)  # Returns 1 (smallest)

# Max-heap workaround (negate values)
max_heap = []
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -1)
heapq.heappush(max_heap, -4)

max_val = -heapq.heappop(max_heap) # Returns 4 (largest)
```

**Time**: O(n log k)
**Space**: O(k)

### What's heapify? Time complexity?

**Heapify**: Convert array into heap.

**Bottom-up heapify**:
```python
def heapify(arr):
    n = len(arr)

    # Start from last non-leaf node
    for i in range(n // 2 - 1, -1, -1):
        bubble_down(arr, n, i)

def bubble_down(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        bubble_down(arr, n, largest)
```

**Time complexity**: **O(n)** (not O(n log n)!)

**Why O(n)**:
- Height h nodes: at most n/(2^(h+1))
- Work per node: O(h)
- Total: Σ(n/(2^(h+1)) × h) = O(n)

### Implement heap sort

**Heap sort**: Sort using heap.

```python
def heap_sort(arr):
    n = len(arr)

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)
```

**Time**: O(n log n)
**Space**: O(1) (in-place)

**Advantage**: In-place
**Disadvantage**: Not stable

---

## Part 6: Hash Tables

### What's a hash table?

**Hash table**: Data structure mapping keys to values using hash function.

**Components**:
1. **Hash function**: key → index
2. **Array**: Store values
3. **Collision resolution**: Handle collisions

**Operations** (average):
- Insert: O(1)
- Search: O(1)
- Delete: O(1)

**Worst case**: O(n) if all keys collide

### What's a hash function?

**Hash function**: Maps key to array index.

**Properties of good hash function**:
1. **Deterministic**: Same key → same hash
2. **Uniform distribution**: Spread keys evenly
3. **Fast**: O(1) computation
4. **Minimize collisions**: Different keys → different hashes (ideally)

**Example**:
```python
def hash_function(key, size):
    return hash(key) % size
```

**Common techniques**:
- Division method: `h(k) = k mod m`
- Multiplication method: `h(k) = floor(m * (k*A mod 1))`
- Universal hashing: Random hash function from family

### What are collisions? Resolution strategies?

**Collision**: Two keys hash to same index.

**Resolution strategies**:

**1. Chaining** (separate chaining):
- Each bucket → linked list
- Insert: Add to list at index
- Search: Traverse list
- Load factor α = n/m (n keys, m buckets)
- Average time: O(1 + α)

```python
class HashTableChaining:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]  # List of lists for chaining

    def insert(self, key, value):
        index = hash(key) % self.size
        # Check if key exists in bucket to update value
        for pair in self.table[index]:
            if pair[0] == key:
                pair[1] = value
                return
        # Key not found, append new pair to bucket
        self.table[index].append([key, value])
```

**2. Open addressing** (probing):
- Store in array itself
- On collision, probe for empty slot

**Probing methods**:
- **Linear**: Try i+1, i+2, i+3, ... (clustering problem)
- **Quadratic**: Try i+1², i+2², i+3², ...
- **Double hashing**: Use second hash function

```python
class HashTableOpenAddressing:
    def __init__(self, size=10):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size

    def insert(self, key, value):
        index = hash(key) % self.size

        # Linear probing: find empty slot or matching key
        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.values[index] = value  # Update existing key
                return
            index = (index + 1) % self.size  # Move to next slot (wrap around)

        # Place new key-value pair in empty slot
        self.keys[index] = key
        self.values[index] = value
```

### When does hash table degrade to O(n)?

**Degrades when**:
1. **All keys collide**: Hash to same bucket
2. **Poor hash function**: Non-uniform distribution
3. **High load factor**: α > 1 (chaining) or α close to 1 (open addressing)

**Example**: Collision on every insert
- Chaining: Single bucket has n elements → O(n) search
- Open addressing: Many probes → O(n) worst case

**Prevention**:
- Good hash function
- Resize when load factor too high (rehashing)
- Typical: Resize when α > 0.7

### What's rehashing?

**Rehashing**: Resize hash table when too full.

**When**: Load factor α exceeds threshold (e.g., 0.75)

**Process**:
1. Create new array (typically 2x size)
2. Recompute hash for all keys (new size)
3. Insert into new array

**Time**: O(n) for rehashing, but **amortized O(1)** per insert

**Example**:
```python
def resize(self):
    old_table = self.table
    self.size = self.size * 2  # Double the size
    self.table = [[] for _ in range(self.size)]

    # Rehash all existing items into new table
    for bucket in old_table:
        for key, value in bucket:
            self.insert(key, value)  # Uses new size for hash calculation
```

**Why 2x**: Ensures amortized O(1)
- Resize at sizes: 1, 2, 4, 8, 16, ...
- Total work to reach n: n + n/2 + n/4 + ... = O(n)
- Amortized per insert: O(1)

---

## Part 7: Graphs

### What's a graph?

**Graph** G = (V, E): Set of vertices V and edges E.

**Types**:
1. **Directed**: Edges have direction (u → v)
2. **Undirected**: Edges bidirectional (u — v)
3. **Weighted**: Edges have weights
4. **Unweighted**: All edges equal weight

**Representations**:

**1. Adjacency matrix**: 2D array
```python
# graph[i][j] = 1 if edge from i to j
graph = [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]
```
- Space: O(V²)
- Check edge: O(1)
- Iterate neighbors: O(V)

**2. Adjacency list**: Array of lists
```python
# graph[i] = list of neighbors of i
graph = {
    0: [1],
    1: [0, 2],
    2: [1]
}
```
- Space: O(V + E)
- Check edge: O(degree)
- Iterate neighbors: O(degree)

**When to use**:
- Adjacency matrix: Dense graphs (E ≈ V²)
- Adjacency list: Sparse graphs (E << V²) - most common

### What's graph traversal (DFS vs BFS)?

**DFS** (Depth-First Search):
- Explore as deep as possible before backtracking
- Uses stack (or recursion)
- Applications: Cycle detection, topological sort, connected components

**Recursive DFS**:
```python
def dfs_recursive(graph, node, visited):
    # Mark current node as visited and print it
    visited.add(node)
    print(node)

    # Recur for all adjacent vertices
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

# Example usage (assuming 'graph' is defined, e.g., as an adjacency list)
# graph = {'A': ['B', 'C'], 'B': ['D'], 'C': ['E'], 'D': [], 'E': []}
# visited = set()
# dfs_recursive(graph, 'A', visited)
```

**Iterative DFS**:
```python
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]  # Use stack for DFS

    while stack:
        node = stack.pop()
        if node not in visited:
            print(node)
            visited.add(node)
            # Add neighbors to stack. Order might differ from recursive depending on implementation.
            # To mimic recursive DFS (processing left-most child first), add neighbors in reverse order.
            for neighbor in reversed(graph[node]): # Add neighbors to stack
                if neighbor not in visited:
                    stack.append(neighbor)
```

**BFS** (Breadth-First Search):
- Explore neighbors before going deeper
- Uses queue
- Applications: Shortest path (unweighted), level-order

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])  # Use queue for BFS
    visited.add(start)      # Mark start as visited

    while queue:
        node = queue.popleft() # Dequeue front node
        print(node)

        # Enqueue unvisited neighbors
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

**Time**: O(V + E) for both
**Space**: O(V) for visited set

### How to detect cycle in graph?

**Undirected graph** (DFS):
```python
def has_cycle_undirected(graph):
    visited = set()

    def dfs(node, parent):
        visited.add(node) # Mark current node as visited

        for neighbor in graph[node]:
            if neighbor not in visited:
                # If neighbor is not visited, recur for it
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                # If neighbor is visited and not parent of current node, then a cycle exists
                return True

        return False

    # Iterate over all nodes to handle disconnected components
    for node in graph:
        if node not in visited:
            if dfs(node, -1): # -1 indicates no parent for the starting node
                return True

    return False
```

**Directed graph** (DFS + recursion stack):
```python
def has_cycle_directed(graph):
    visited = set()      # Keeps track of all visited nodes
    rec_stack = set()    # Keeps track of nodes in the current recursion path

    def find_cycle(node):
        visited.add(node)
        rec_stack.add(node)  # Add node to recursion stack

        for neighbor in graph[node]:
            if neighbor not in visited:
                # If neighbor is not visited, recur for it
                if find_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
    for node in graph:
        if node not in visited:
            if dfs(node):
                return True

    return False
```

### What's topological sort?

**Topological sort**: Linear ordering of vertices in DAG (Directed Acyclic Graph) such that for every edge (u → v), u comes before v.

**Applications**:
- Task scheduling (dependencies)
- Course prerequisites
- Build systems

**DFS approach** (reverse postorder):
```python
def topological_sort(graph):
    visited = set()
    stack = []

    def dfs(node):
        visited.add(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

        stack.append(node)

    for node in graph:
        if node not in visited:
            dfs(node)

    return stack[::-1]
```

**Kahn's algorithm** (BFS + indegree):
```python
from collections import deque

def topological_sort_kahn(graph):
    # Compute indegree
    indegree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            indegree[neighbor] += 1

    # Queue nodes with indegree 0
    queue = deque([node for node in indegree if indegree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    # If result has all nodes, valid topological sort
    return result if len(result) == len(graph) else []
```

**Time**: O(V + E)
**Space**: O(V)

### What's Dijkstra's algorithm?

**Dijkstra's**: Shortest path from source to all vertices (non-negative weights).

**Algorithm**:
1. Initialize distances: dist[source] = 0, others = ∞
2. Use min-heap: (distance, node)
3. Extract min, relax neighbors
4. Repeat until heap empty

```python
import heapq

def dijkstra(graph, start):
    # graph[u] = [(v, weight), ...]
    dist = {node: float('inf') for node in graph}
    dist[start] = 0

    heap = [(0, start)]
    visited = set()

    while heap:
        d, node = heapq.heappop(heap)

        if node in visited:
            continue

        visited.add(node)

        for neighbor, weight in graph[node]:
            new_dist = d + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))

    return dist
```

**Time**: O((V + E) log V) with heap
**Space**: O(V)

**Note**: Doesn't work with negative weights (use Bellman-Ford).

### What's union-find (disjoint set)?

**Union-Find**: Data structure for disjoint sets with two operations:
- **Find**: Which set does element belong to?
- **Union**: Merge two sets

**Applications**:
- Detect cycles in undirected graph
- Kruskal's MST algorithm
- Network connectivity

**Implementation** (with path compression + union by rank):
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True
```

**Time**: O(α(n)) per operation (α is inverse Ackermann, practically constant)

### What's minimum spanning tree (MST)?

**MST**: Subset of edges connecting all vertices with minimum total weight (for connected, weighted, undirected graph).

**Applications**:
- Network design (minimize cable)
- Clustering
- Approximation algorithms

**Kruskal's algorithm**:
1. Sort edges by weight
2. Use union-find to avoid cycles
3. Add edge if doesn't create cycle

```python
def kruskal(n, edges):
    # edges = [(weight, u, v), ...]
    edges.sort()  # Sort edges by weight (greedy approach)
    uf = UnionFind(n)
    mst = []
    total_weight = 0

    for weight, u, v in edges:
        # If adding edge doesn't create a cycle (vertices in different sets)
        if uf.union(u, v):
            mst.append((u, v, weight))
            total_weight += weight

    return mst, total_weight
```

**Time**: O(E log E) for sorting
**Space**: O(V) for union-find

**Prim's algorithm**: Similar to Dijkstra's, grow MST from source using heap.

---

## Part 8: Tries

### What's a trie?

**Trie** (prefix tree): Tree for storing strings where each path represents a string.

**Properties**:
- Each node represents a character
- Root is empty
- Path from root to node = prefix
- Special marker for end of word

**Applications**:
- Autocomplete
- Spell checker
- IP routing
- Dictionary

**Node structure**:
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
```

### Implement a trie

```python
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()  # Create new node if path doesn't exist
            node = node.children[char]  # Move to next node
        node.is_end = True  # Mark end of word

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False  # Path doesn't exist
            node = node.children[char]
        return node.is_end  # True if it's a complete word, not just a prefix

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True  # Prefix exists
```

**Time**:
- Insert: O(m) where m = word length
- Search: O(m)
- Prefix: O(m)

**Space**: O(ALPHABET_SIZE × N × M) where N = number of words, M = avg length

### When to use trie vs hash table?

**Trie**:
- ✅ Prefix queries (autocomplete)
- ✅ Sorted order iteration
- ✅ Space-efficient for common prefixes
- ❌ More complex implementation
- ❌ More memory per node (pointers)

**Hash Table**:
- ✅ Exact match queries
- ✅ Simpler implementation
- ✅ O(1) average lookup
- ❌ No prefix queries
- ❌ No sorted order

**Use trie**: Prefix operations, autocomplete, dictionaries
**Use hash table**: Simple key-value lookups
