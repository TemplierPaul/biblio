# Computer Science Interview Q&A

Comprehensive CS interview questions and answers covering data structures, algorithms, systems programming, and language concepts.

## 📁 File Organization

### [[01_Data_Structures]]
Fundamental and advanced data structures:
- **Part 1: Arrays & Strings** - Access patterns, sliding window, string manipulation
- **Part 2: Linked Lists** - Cycle detection, reversal, dummy nodes
- **Part 3: Stacks & Queues** - Applications, monotonic stack, min-stack
- **Part 4: Trees** - BST, traversals (DFS/BFS), balanced trees, LCA
- **Part 5: Heaps & Priority Queues** - Heapify, heap sort, applications
- **Part 6: Hash Tables** - Hash functions, collisions, load factor, rehashing
- **Part 7: Graphs** - Traversal, cycle detection, topological sort, Dijkstra, union-find
- **Part 8: Tries** - Prefix trees, autocomplete, comparison with hash tables

### [[02_Algorithms]]
Algorithm design and analysis:
- **Part 1: Sorting** - Merge sort, quick sort, heap sort, complexity comparison
- **Part 2: Searching** - Binary search, two-pointer technique, variants
- **Part 3: Dynamic Programming** - Memoization vs tabulation, knapsack, LCS
- **Part 4: Greedy Algorithms** - Activity selection, Huffman coding
- **Part 5: Backtracking** - Template, subsets, permutations, N-Queens
- **Part 6: Complexity Analysis** - Big O/Ω/Θ, recurrence relations, amortized analysis

### [[03_Systems_Programming]]
Low-level programming and operating systems:
- **Part 1: Memory Management** - Stack vs heap, memory leaks, pointers, RAII, smart pointers
- **Part 2: Operating Systems** - Processes vs threads, race conditions, mutex, semaphore, deadlock, virtual memory
- **Part 3: Compilation & Linking** - Compilation stages, static vs dynamic linking, virtual functions
- **Part 4: Computer Architecture** - Floating point, cache, pipelining, branch prediction
- **Part 5: Concurrency** - Concurrency vs parallelism, thread pools, atomics, GIL

### [[04_Programming_Languages]]
Language concepts and paradigms:
- **Part 1: Python vs C++** - Differences, practical implications, memory management
- **Part 2: OOP** - Encapsulation, inheritance, polymorphism, composition vs inheritance
- **Part 3: Type Systems** - Static vs dynamic, strong vs weak, type hints, duck typing
- **Part 4: Advanced Concepts** - Closures, decorators, generators, metaprogramming

### [[05_Distributed_Systems]]
Distributed computing for ML at scale:
- **Part 1: Fundamentals** - CAP theorem, Consistency models
- **Part 2: MapReduce & Spark** - RDDs, DAGs, Spark vs MapReduce
- **Part 3: Distributed Training** - Data/Model/Pipeline parallelism, DDP, All-reduce, Ray, Horovod
- **Part 4: Fault Tolerance** - Checkpointing, Replication, Two-phase commit

### [[06_System_Design]]
Designing scalable ML systems:
- **Part 1: Framework** - 5-step design process, requirements, constraints
- **Part 2: Serving** - Batch vs Real-time, Feature Stores, Deployment patterns
- **Part 3: Monitoring** - Training-serving skew, Data drift, Concept drift
- **Part 4: Case Studies** - RecSys, Search Ranking, Ad Click Prediction

### [[07_Databases_SQL]]
Data storage and retrieval:
- **Part 1: SQL** - Joins, Window functions, CTEs, Execution order
- **Part 2: Design** - Normalization, Star vs Snowflake schema
- **Part 3: NoSQL** - Redis, MongoDB, Cassandra, Neo4j
- **Part 4: Optimization** - Indexes, Explain plans, Columnar storage

### [[08_ML_Infrastructure]]
Hardware and operational infrastructure:
- **Part 1: GPU Programming** - CUDA hierarchy, Kernel launches, Tensor cores
- **Part 2: Optimization** - Quantization, Pruning, Distillation
- **Part 3: MLOps** - Experiment tracking, Model registry, Data versioning
- **Part 4: Serving** - TensorRT, ONNX, Triton, Batching

### [[09_Software_Engineering]]
Best practices for ML engineering:
- **Part 1: Git** - Rebase vs Merge, Bisect, Cherry-pick
- **Part 2: Testing** - Unit vs Integration, Mocking, Property-based testing
- **Part 3: Code Quality** - Typing (mypy), Linting, Design patterns
- **Part 4: Profiling** - cProfile, Memory leaks, GIL, Multiprocessing

## 🎯 How to Use

### For Interview Prep
1. **Master fundamentals first** - Start with data structures and algorithms
2. **Understand, don't memorize** - Know why algorithms work
3. **Practice implementations** - Code key algorithms from scratch
4. **Analyze complexity** - Always consider time and space complexity
5. **Draw diagrams** - Visualize data structures and algorithm steps

### Study Paths

**SWE Interview Prep** (2-3 weeks):
01 (Data Structures) → 02 (Algorithms) → Practice coding problems

**Systems Engineer**:
03 (Systems Programming) → 01 (Data Structures) → 02 (Algorithms: Part 6)

**ML Infrastructure / Research Engineer**:
05 (Distributed) → 08 (Infrastructure) → 06 (System Design) → 09 (Software Engineering)

**Full Stack Developer**:
04 (Programming Languages) → 01 (Hash Tables, Tries) → 02 (Dynamic Programming) → 07 (Databases)

**Deep Dive** (comprehensive):
All files in order (01 → 02 → 03 → 04)

## 📊 Content Statistics

- **Total Questions**: ~200+
- **Code Examples**: ~300+
- **Implementations**: Python & C++
- **Difficulty Range**: Fundamentals → Advanced

## 🔑 Key Concepts to Master

### Must-Know Data Structures
- **Array/String**: Two-pointer, sliding window
- **Linked List**: Cycle detection, reversal
- **Stack/Queue**: Monotonic stack, BFS/DFS
- **Tree**: In/pre/postorder, level-order
- **Hash Table**: O(1) operations, collision handling
- **Heap**: Priority queue operations
- **Graph**: DFS/BFS, topological sort

### Must-Know Algorithms
- **Sorting**: Merge sort, quick sort (understand trade-offs)
- **Binary Search**: Template and variants
- **Two-Pointer**: For sorted arrays
- **DFS/BFS**: Recursion and iteration
- **Dynamic Programming**: Pattern recognition
- **Greedy**: When it works vs when it doesn't
- **Backtracking**: Template

### Must-Know Complexity
- **Time**: O(1), O(log n), O(n), O(n log n), O(n²), O(2ⁿ)
- **Space**: Stack vs heap, auxiliary space
- **Amortized**: Dynamic arrays, union-find

### Must-Know Systems
- **Memory**: Stack vs heap, pointers, memory leaks
- **Concurrency**: Race conditions, deadlock
- **OS**: Processes vs threads
- **Compilation**: Stages, linking

## 💡 Interview Tips

### Before Interview
- ✅ Review time/space complexity of common algorithms
- ✅ Practice implementing on whiteboard/paper
- ✅ Prepare questions about edge cases
- ✅ Review your own code from past projects
- ✅ Understand trade-offs (when to use what)

### During Interview
- 🎯 **Clarify problem**: Ask questions, confirm understanding
- 🎯 **Think aloud**: Explain your thought process
- 🎯 **Start simple**: Brute force first, optimize later
- 🎯 **Test your code**: Walk through with examples
- 🎯 **Optimize**: Discuss time/space trade-offs
- 🎯 **Handle edge cases**: Empty input, null, duplicates

### Common Pitfalls to Avoid
- ❌ Jumping to code without planning
- ❌ Not testing with examples
- ❌ Ignoring edge cases
- ❌ Not analyzing complexity
- ❌ Being silent (explain your thinking!)

## 🧪 Practice Strategy

### Week 1-2: Foundations
- Arrays, strings, linked lists
- Basic sorting and searching
- Hash tables
- 20-30 easy problems

### Week 3-4: Core Algorithms
- Trees and graphs
- DFS/BFS
- Binary search
- Dynamic programming (intro)
- 30-40 medium problems

### Week 5-6: Advanced
- Advanced DP
- Heaps and greedy
- Backtracking
- Graph algorithms (Dijkstra, MST)
- 20-30 hard problems

### Week 7-8: Review & Mock Interviews
- Review all topics
- Mock interviews with peers
- Timed problem solving
- System design (if applicable)

## 📚 Recommended Resources

### Books
- **Cracking the Coding Interview** - Gayle Laakmann McDowell
- **Introduction to Algorithms** - CLRS
- **The C Programming Language** - K&R
- **Effective C++** - Scott Meyers
- **Fluent Python** - Luciano Ramalho

### Online Platforms
- **LeetCode** - Problem practice (start with Easy → Medium)
- **HackerRank** - Tutorials + problems
- **Codeforces/AtCoder** - Competitive programming
- **Project Euler** - Mathematical problems

### Video Resources
- **Abdul Bari** (YouTube) - Algorithms
- **mycodeschool** (YouTube) - Data structures
- **MIT OpenCourseWare** - 6.006, 6.046
- **CS Dojo** (YouTube) - Interview prep

## 🚀 Topic Checklist

### Data Structures ✓
- [ ] Array manipulation (two-pointer, sliding window)
- [ ] String algorithms (KMP, Rabin-Karp)
- [ ] Linked list operations
- [ ] Stack (monotonic, min-stack)
- [ ] Queue (BFS, priority queue)
- [ ] Binary trees (BST, AVL, traversals)
- [ ] Heaps (min-heap, max-heap, operations)
- [ ] Hash tables (implementation, collision handling)
- [ ] Graphs (adjacency list/matrix, traversals)
- [ ] Tries (prefix trees, autocomplete)
- [ ] Union-Find (disjoint sets)

### Algorithms ✓
- [ ] Sorting (merge, quick, heap, counting, radix)
- [ ] Searching (binary search, two-pointer)
- [ ] Dynamic programming (common patterns)
- [ ] Greedy algorithms (when they work)
- [ ] Backtracking (template, pruning)
- [ ] Graph algorithms (DFS, BFS, topological sort)
- [ ] Shortest path (Dijkstra, Bellman-Ford)
- [ ] MST (Kruskal, Prim)
- [ ] Divide and conquer
- [ ] Bit manipulation

### Complexity Analysis ✓
- [ ] Big O notation
- [ ] Time complexity analysis
- [ ] Space complexity analysis
- [ ] Amortized analysis
- [ ] Master theorem
- [ ] Recurrence relations

### Systems & Languages ✓
- [ ] Memory management (stack, heap, pointers)
- [ ] Processes vs threads
- [ ] Synchronization (mutex, semaphore)
- [ ] Deadlock (detection, prevention)
- [ ] Compilation process
- [ ] Static vs dynamic linking
- [ ] Virtual memory
- [ ] Cache performance
- [ ] OOP concepts
- [ ] Type systems

### ML System Design & Infra ✓
- [ ] Distributed training (DDP, Model Parallelism)
- [ ] System design framework (Training -> Serving)
- [ ] Feature Stores
- [ ] Model Monitoring (Drift)
- [ ] SQL Window Functions & Joins
- [ ] NoSQL types (Key-Value, Document, Column, Graph)
- [ ] GPU architecture (Global/Shared memory)
- [ ] Model Optimization (Quantization, Pruning)
- [ ] MLOps (Registry, Versioning)
- [ ] Docker & Kubernetes basics

## 🎓 Company-Specific Focus

### FAANG/Big Tech
- **Heavy focus**: Algorithms, data structures
- **Practice**: LeetCode Medium/Hard
- **Topics**: DP, graphs, trees
- **Time**: 45-60 min coding rounds

### Startups
- **Focus**: Practical coding, system design
- **Practice**: Build projects, real-world problems
- **Topics**: Web dev, databases, APIs
- **Time**: Varied, often take-home

### Systems Companies (C/C++ roles)
- **Focus**: Low-level programming, performance
- **Practice**: Memory management, concurrency
- **Topics**: Pointers, threading, caches
- **Time**: Deep technical discussions

### ML/AI Companies
- **Focus**: Algorithms, math, Python
- **Practice**: LeetCode + ML problems
- **Topics**: Trees, graphs, matrices
- **Time**: Often includes ML theory

## 💻 Sample Interview Structure

### Phone Screen (45-60 min)
- Intro (5 min)
- Coding problem(s) (30-40 min)
  - Usually 1-2 medium problems
  - Focus on correctness + clarity
- Your questions (5-10 min)

### Onsite (4-6 rounds)
- **2-3 Coding rounds**: Data structures + algorithms
- **1 System design**: Architecture discussion (senior roles)
- **1 Behavioral**: Team fit, past projects
- **Optional**: Domain-specific (ML, distributed systems, etc.)

### What Interviewers Look For
1. **Problem solving**: Can you break down problems?
2. **Coding**: Clean, correct, efficient code?
3. **Communication**: Explain thinking clearly?
4. **Debugging**: Find and fix errors?
5. **Optimization**: Improve initial solution?
6. **Edge cases**: Thorough testing?

## 📝 Quick Reference

### Time Complexity Cheat Sheet
```
O(1)      - Hash table lookup, array access
O(log n)  - Binary search, balanced tree operations
O(n)      - Linear search, array traversal
O(n log n)- Merge sort, heap sort, quicksort (average)
O(n²)     - Bubble sort, nested loops
O(2ⁿ)     - Subsets, recursive Fibonacci
O(n!)     - Permutations, traveling salesman (brute force)
```

### Space Complexity Cheat Sheet
```
O(1)      - Few variables
O(log n)  - Recursion (balanced tree height)
O(n)      - Array/hash table, recursion (linear)
O(n²)     - 2D array, adjacency matrix
```

### Common Patterns
```
Two Pointer:    Sorted array problems
Sliding Window: Subarray/substring problems
Fast/Slow:      Linked list cycle detection
BFS:            Shortest path, level-order
DFS:            All paths, backtracking
DP:             Optimization problems
Greedy:         Local optimum → global optimum
Binary Search:  Search in sorted space
```

## 🔗 Related Resources

- **@Questions/ML/**: Machine learning and AI interview questions
- **@Learning/**: Detailed learning notes on various CS topics
- **Practice platforms**: LeetCode, HackerRank, CodeSignal

---

**Remember**: Understanding > Memorization. Focus on **why** algorithms work, not just **what** they do.

**Good luck with your interviews!** 🚀
