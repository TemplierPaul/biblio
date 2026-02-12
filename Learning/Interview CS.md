# Questions
- [ ] Python vs C++:
	- [ ] Difference between python and C++?
	- [ ] Practical implications of the differences?
	- [ ] When you use variables in C++ vs Python, what are the types? (Hinted at python variables not having a fixed type.)
	- [ ] Memory leaks, memory management. Python vs C++.
	- [ ] Private variables in C++ and Python. So does this mean that you can't use Python to write secure software?
- [ ] Object oriented programming. What are the advantages? When can it be good?
- [ ] How would you implement Python dictionary if Python interpreter didn't exist?
- [ ] What is a hashtable? When should one be used? What is meant by collisions? What are some strategies for resolving collisions?
- [ ] Geometric algorithm for calculating pi
- [ ] Write efficient code for matrix multiplication.
- [ ] Design a distributed pipeline for training a neural network on terabytes of data.
	- [ ] **Data Ingestion:** Use Apache Kafka for streaming data.
	- [ ] **Distributed Training:** Implement Horovod for multi-GPU training.
	- [ ] **Storage:** Use cloud storage like AWS S3 for intermediate results.
- [ ] Describe a linked list
- [ ] how to invert a matrix efficiently
- [ ] What is big O notation
	- [ ] Landau notation, f = O(g) if f/g is basically a constant
- [ ] what is a memory leak?
- [ ] what is a virtual function?
- [ ] How does computer store floating point number?
- [ ] What’s encapsulation?

```python
class HashMap:

	def __init__(self):
		self.size = 100
		self.map = [[] for _ in range(self.size)]

	def _hash(self, key):
		return hash(key) % self.size

	def insert(self, key, value):
		hash_key = self._hash(key)
		for pair in self.map[hash_key]:
			if pair[0] == key:
			pair[1] = value
			return self.map[hash_key].append([key, value])

	def get(self, key):
		hash_key = self._hash(key)
		for pair in self.map[hash_key]:
			if pair[0] == key:
			return pair[1]
		return None
```


```python
class HashNode:
    """Node to store key-value pairs in a linked list for collision resolution."""
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None


class HashTable:
    """Hash Table implementation using an array of linked lists."""
    def __init__(self, capacity=10):
        self.capacity = capacity  # Number of buckets
        self.size = 0  # Number of key-value pairs stored
        self.buckets = [None] * capacity  # Array of linked lists

    def _hash(self, key):
        """Generate a hash for the given key."""
        return hash(key) % self.capacity

    def insert(self, key, value):
        """Insert a key-value pair into the hash table."""
        index = self._hash(key)
        head = self.buckets[index]

        # Check if key already exists in the chain
        while head:
            if head.key == key:
                head.value = value  # Update value if key exists
                return
            head = head.next

        # Insert new node at the beginning of the chain
        new_node = HashNode(key, value)
        new_node.next = self.buckets[index]
        self.buckets[index] = new_node
        self.size += 1

    def get(self, key):
        """Retrieve the value associated with the given key."""
        index = self._hash(key)
        head = self.buckets[index]

        # Search for the key in the chain
        while head:
            if head.key == key:
                return head.value
            head = head.next

        return None  # Key not found

    def delete(self, key):
        """Remove a key-value pair from the hash table."""
        index = self._hash(key)
        head = self.buckets[index]
        prev = None

        # Search for the key in the chain
        while head:
            if head.key == key:
                if prev:
                    prev.next = head.next  # Remove node from chain
                else:
                    self.buckets[index] = head.next  # Update head of chain
                self.size -= 1
                return
            prev = head
            head = head.next

    def display(self):
        """Display the contents of the hash table."""
        for i in range(self.capacity):
            print(f"Bucket {i}:", end=" ")
            head = self.buckets[i]
            while head:
                print(f"({head.key}: {head.value})", end=" -> ")
                head = head.next
            print("None")


# Example usage
hash_table = HashTable()

# Insert key-value pairs
hash_table.insert("apple", 10)
hash_table.insert("banana", 20)
hash_table.insert("grape", 30)
hash_table.insert("orange", 40)

# Retrieve a value
print("Value for 'banana':", hash_table.get("banana"))

# Delete a key
hash_table.delete("banana")
print("After deleting 'banana':")
hash_table.display()

# Insert a new key-value pair
hash_table.insert("kiwi", 50)
print("After inserting 'kiwi':")
hash_table.display()
```

# Cheatsheet
## Monotonic increasing stack
```python
def fn(arr):
    stack = []
    ans = 0

    for num in arr:
        # for monotonic decreasing, just flip the > to <
        while stack and stack[-1] > num:
            # do logic
            stack.pop()
        stack.append(num)
    
    return ans
```

## Binary tree: DFS (recursive)
```python
def dfs(root):
    if not root:
        return
    
    ans = 0

    # do logic
    dfs(root.left)
    dfs(root.right)
    return ans
```

## Binary tree: DFS (iterative)
```python
def dfs(root):
    stack = [root]
    ans = 0

    while stack:
        node = stack.pop()
        # do logic
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

    return ans
```

## Binary tree: BFS
```python
from collections import deque

def fn(root):
    queue = deque([root])
    ans = 0

    while queue:
        current_length = len(queue)
        # do logic for current level

        for _ in range(current_length):
            node = queue.popleft()
            # do logic
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return ans
```

## Find top k elements with heap
```python
import heapq

def fn(arr, k):
    heap = []
    for num in arr:
        # do some logic to push onto heap according to problem's criteria
        heapq.heappush(heap, (CRITERIA, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for num in heap]
```

## Backtracking
```python
def backtrack(curr, OTHER_ARGUMENTS...):
    if (BASE_CASE):
        # modify the answer
        return
    
    ans = 0
    for (ITERATE_OVER_INPUT):
        # modify the current state
        ans += backtrack(curr, OTHER_ARGUMENTS...)
        # undo the modification of the current state
    
    return ans
```

## Sorting
![[sorting.png]]