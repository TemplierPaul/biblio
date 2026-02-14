# Distributed Systems - Interview Q&A

**How to use**: Questions are formatted as collapsible headings. Try to answer before expanding.

---

## Table of Contents

- [[#Part 1: Distributed Computing Fundamentals]]
  - [[#What's a distributed system?]]
  - [[#Why distribute computation?]]
  - [[#What's the CAP theorem?]]
  - [[#What's consistency in distributed systems?]]
  - [[#What's eventual consistency?]]
- [[#Part 2: MapReduce & Spark]]
  - [[#What's MapReduce?]]
  - [[#How does MapReduce work?]]
  - [[#What's Apache Spark?]]
  - [[#Spark vs MapReduce?]]
  - [[#What's a Spark RDD?]]
- [[#Part 3: Distributed Training]]
  - [[#What's data parallelism?]]
  - [[#What's model parallelism?]]
  - [[#What's pipeline parallelism?]]
  - [[#What's PyTorch DDP?]]
  - [[#What's Ray and how does it differ from Spark?]]
  - [[#What's Horovod?]]
  - [[#What's all-reduce?]]
  - [[#What's ring all-reduce?]]
- [[#Part 4: Fault Tolerance & Reliability]]
  - [[#What's fault tolerance?]]
  - [[#What's checkpointing?]]
  - [[#What's replication?]]
  - [[#What's the two-phase commit?]]
  - [[#How do you handle node failures?]]
- [[#Part 5: Communication Patterns]]
  - [[#What's a parameter server?]]
  - [[#What's peer-to-peer communication?]]
  - [[#What's collective communication?]]
  - [[#What's gradient accumulation?]]
  - [[#What's asynchronous vs synchronous training?]]

---

## Part 1: Distributed Computing Fundamentals

### What's a distributed system?

**Distributed system**: Collection of independent computers that appear to users as a single coherent system.

**Key characteristics**:
1. **Multiple nodes**: Independent machines with own CPU/memory
2. **Network communication**: Nodes communicate via messages
3. **Shared goal**: Coordinate to solve a problem
4. **No shared memory**: Each node has own local memory
5. **Partial failures**: Some nodes can fail while others continue

**Examples**:
- **Web services**: Load-balanced servers
- **Databases**: Sharded/replicated databases
- **ML training**: Multi-GPU/multi-node training
- **Storage**: HDFS, S3
- **Computation**: Spark, Ray

**Why distributed**:
- Scale beyond single machine
- Fault tolerance
- Geographic distribution
- Cost (commodity hardware)

### Why distribute computation?

**Reasons to distribute**:

**1. Scale beyond single machine**:
```
Single GPU: 80GB memory → Limited to models < 80GB
8 GPUs: 640GB memory → Train 100B+ parameter models
```

**2. Speed up training**:
```
1 GPU: 7 days to train
8 GPUs with data parallelism: ~1 day (7x speedup)
```

**3. Handle large datasets**:
```
Dataset: 10TB
Single machine RAM: 256GB → Cannot fit
Distributed: 100 machines × 256GB = 25.6TB → Fits
```

**4. Fault tolerance**:
- Single machine failure → entire job fails
- Distributed with replication → job continues

**5. Cost efficiency**:
- Preemptible/spot instances cheaper
- Can use commodity hardware

**Trade-offs**:
- **Communication overhead**: Network slower than local memory
- **Complexity**: Synchronization, failure handling
- **Diminishing returns**: Amdahl's law limits speedup

**When NOT to distribute**:
- Problem fits on single machine
- Communication cost > computation cost
- Debugging/prototyping (distributed harder to debug)

### What's the CAP theorem?

**CAP theorem** (Brewer's theorem): Distributed system can have at most 2 of 3:

**C** (Consistency): All nodes see same data at same time
**A** (Availability): Every request receives response (success/failure)
**P** (Partition tolerance): System works despite network partitions

**You must choose**: CA, CP, or AP (but P usually required)

**Partition**: Network split, some nodes can't communicate

**Trade-offs**:

**CP (Consistency + Partition tolerance)**: Sacrifice availability
- **Example**: Banking system
- Network partition → reject writes until healed
- Better to be unavailable than inconsistent (wrong balance!)

**AP (Availability + Partition tolerance)**: Sacrifice consistency
- **Example**: Social media feed
- Network partition → serve stale data
- Better to show old post than no posts

**CA (Consistency + Availability)**: Sacrifice partition tolerance
- **Example**: Single-node database
- No network partitions (single machine)
- Not truly distributed

**Real systems**:
```
MongoDB: CP (consistent but may be unavailable)
Cassandra: AP (available but eventually consistent)
PostgreSQL (single): CA (consistent + available, not distributed)
```

**ML training context**:
- **Synchronous SGD**: CP (consistent gradients, but stalls on failure)
- **Asynchronous SGD**: AP (always progressing, but stale gradients)

### What's consistency in distributed systems?

**Consistency**: Guarantees about what values reads return.

**Consistency models** (strongest → weakest):

**1. Linearizability (Strict Consistency)**:
- Reads return most recent write
- Operations appear atomic
- As if single copy of data

**Example**:
```
Time →
T1: Write(x=1) ------→ [committed]
T2:                         Read(x) → Must return 1
```

**Use**: Banking, inventory

**2. Sequential Consistency**:
- All nodes see operations in same order
- Order may differ from real time

**Example**:
```
Node A: Write(x=1), Write(x=2)
Node B: Must see x=1 before x=2 (not x=2 then x=1)
```

**3. Causal Consistency**:
- Causally related operations seen in order
- Concurrent operations can be seen in any order

**Example**:
```
Alice posts: "Hello"
Bob replies: "Hi Alice" (causally dependent)
All nodes must see "Hello" before "Hi Alice"

But concurrent posts can be seen in any order
```

**4. Eventual Consistency**:
- If no new writes, all nodes eventually converge
- No guarantees about when

**Example**:
```
Write(x=1) on Node A
Node B might read x=0 (stale) for a while
Eventually Node B will read x=1
```

**Use**: DNS, social media, caching

**ML training**:
- **Synchronous SGD**: Strong consistency (all workers use same weights)
- **Asynchronous SGD**: Eventual consistency (workers use stale weights)

### What's eventual consistency?

**Eventual consistency**: If no new updates, all replicas eventually converge to same value.

**Key properties**:
1. **Weak guarantee**: No bound on convergence time
2. **Allows stale reads**: May read old data
3. **High availability**: Always accept reads/writes
4. **Conflict resolution**: Handle concurrent writes

**Example** (social media):
```
You post: "Hello world"

Replica 1: Sees post immediately
Replica 2: Sees post after 100ms
Replica 3: Sees post after 500ms

Eventually all replicas have the post
```

**Conflict resolution**:
```
User edits profile from two devices:

Device A: name = "Alice"
Device B: name = "Alicia"

Which wins?
- Last-write-wins (timestamp)
- Vector clocks
- Application-specific (merge)
```

**Advantages**:
- High availability (always responsive)
- Low latency (no waiting for sync)
- Partition tolerant

**Disadvantages**:
- Stale reads possible
- Conflict resolution needed
- Application complexity

**When to use**:
- High availability > strict consistency
- Read-heavy workloads
- Geographic distribution
- Examples: DNS, CDNs, shopping carts, social feeds

**When NOT to use**:
- Financial transactions
- Inventory management
- Anything requiring immediate consistency

---

## Part 2: MapReduce & Spark

### What's MapReduce?

**MapReduce**: Programming model for processing large datasets in parallel across clusters.

**Key idea**: Express computation as two functions:
1. **Map**: Process each item independently → (key, value) pairs
2. **Reduce**: Combine values for each key

**Example** (word count):
```
Input: ["hello world", "hello spark"]

Map phase:
"hello world" → [("hello", 1), ("world", 1)]
"hello spark" → [("hello", 1), ("spark", 1)]

Shuffle: Group by key
("hello", [1, 1])
("world", [1])
("spark", [1])

Reduce phase:
("hello", [1, 1]) → ("hello", 2)
("world", [1])    → ("world", 1)
("spark", [1])    → ("spark", 1)

Output: [("hello", 2), ("world", 1), ("spark", 1)]
```

**Why MapReduce**:
- Abstracts parallelism (don't think about threads)
- Fault tolerance (framework handles failures)
- Scales to thousands of machines

**Code example**:
```python
# Map function
def mapper(line):
    for word in line.split():
        yield (word, 1)

# Reduce function
def reducer(key, values):
    yield (key, sum(values))
```

### How does MapReduce work?

**MapReduce execution**:

**Phases**:

**1. Input Split**:
- Divide input into chunks (e.g., 64MB HDFS blocks)
- Each chunk assigned to a mapper

**2. Map Phase**:
- Each mapper processes its chunk in parallel
- Emit (key, value) pairs
- Output written to local disk

**3. Shuffle & Sort**:
- Group all values by key
- Partition keys across reducers (e.g., hash(key) % num_reducers)
- Transfer data over network (expensive!)
- Sort by key

**4. Reduce Phase**:
- Each reducer processes all values for its keys
- Emit final output
- Write to distributed filesystem (HDFS)

**Architecture**:
```
Input Data (HDFS)
        ↓
    [Split 1] [Split 2] [Split 3]
        ↓          ↓         ↓
    [Mapper] [Mapper] [Mapper]  ← Map phase (parallel)
        ↓          ↓         ↓
    (key, value) pairs
        ↓
    [Shuffle & Sort]  ← Network transfer
        ↓
    Grouped by key
        ↓          ↓         ↓
   [Reducer] [Reducer] [Reducer]  ← Reduce phase (parallel)
        ↓
   Output (HDFS)
```

**Fault tolerance**:
- Task failure → re-run on different node
- Input data replicated (HDFS)
- Map output on local disk (can recompute)

**Example** (counting user page views):
```python
# Input: (user_id, page_url)
# Output: (user_id, total_views)

def mapper(record):
    user_id, page_url = record
    yield (user_id, 1)

def reducer(user_id, counts):
    yield (user_id, sum(counts))

# Execution:
# 1B records across 1000 mappers
# Shuffle groups by user_id
# 100 reducers compute totals
```

### What's Apache Spark?

**Apache Spark**: Fast, general-purpose distributed computing framework.

**Key improvements over MapReduce**:

**1. In-memory computing**:
- Cache data in RAM (not disk)
- 10-100x faster for iterative algorithms

**2. DAG execution**:
- Optimize entire workflow (not just map-reduce)
- Lazy evaluation

**3. Rich APIs**:
- Transformations: map, filter, join, groupBy
- Actions: collect, count, save
- SQL, streaming, ML libraries

**Example** (word count in Spark):
```python
from pyspark import SparkContext

sc = SparkContext()

# Read data
lines = sc.textFile("hdfs://data.txt")

# Transformation (lazy)
words = lines.flatMap(lambda line: line.split())
pairs = words.map(lambda word: (word, 1))
counts = pairs.reduceByKey(lambda a, b: a + b)

# Action (triggers execution)
results = counts.collect()
```

**Architecture**:
```
Driver Program
    ↓
SparkContext → Cluster Manager (YARN/Mesos/K8s)
    ↓
Executor (Worker 1)  Executor (Worker 2)  Executor (Worker 3)
  [Cache]              [Cache]              [Cache]
  [Tasks]              [Tasks]              [Tasks]
```

**When to use Spark**:
- Iterative algorithms (ML, graph processing)
- Interactive queries
- Stream processing
- Complex DAG workflows

### Spark vs MapReduce?

| Feature | MapReduce | Spark |
|---------|-----------|-------|
| **Speed** | Slow (disk I/O) | Fast (in-memory) |
| **Iteration** | Poor (write to disk each time) | Excellent (cache in RAM) |
| **Ease of use** | Verbose (Java) | Concise (Python, Scala) |
| **Fault tolerance** | Re-run tasks | Re-run from lineage |
| **Latency** | Batch only | Batch + streaming |
| **API** | Map + Reduce only | Rich (SQL, ML, graphs) |
| **Memory** | Disk-based | Memory-based |

**Performance comparison** (iterative algorithm):
```
Logistic Regression (10 iterations):

MapReduce:
- Each iteration: read from disk → compute → write to disk
- 10 × (read + compute + write) ≈ 100 minutes

Spark:
- First iteration: read from disk → compute → cache in RAM
- Next 9 iterations: read from RAM → compute
- 1 × disk + 10 × compute ≈ 10 minutes

10x speedup!
```

**When to use MapReduce**:
- Simple batch processing
- One-pass algorithms
- Already have Hadoop infrastructure
- Limited memory

**When to use Spark**:
- Iterative algorithms (ML)
- Interactive analytics
- Complex pipelines
- Stream processing
- Available memory

### What's a Spark RDD?

**RDD** (Resilient Distributed Dataset): Immutable, partitioned collection of records that can be operated on in parallel.

**Key properties**:

**1. Resilient**: Fault-tolerant via lineage
```
If partition lost → recompute from parent RDD
```

**2. Distributed**: Partitioned across cluster
```
1M records → 100 partitions → 100 machines
```

**3. Immutable**: Transformations create new RDDs
```
rdd1 = sc.parallelize([1, 2, 3])
rdd2 = rdd1.map(lambda x: x * 2)  # New RDD, rdd1 unchanged
```

**Operations**:

**Transformations** (lazy):
```python
rdd2 = rdd1.map(lambda x: x * 2)        # Apply function
rdd3 = rdd1.filter(lambda x: x > 5)     # Filter elements
rdd4 = rdd1.flatMap(lambda x: [x, x*2]) # Map then flatten
rdd5 = rdd1.union(rdd2)                 # Union
rdd6 = rdd1.groupByKey()                # Group by key
rdd7 = rdd1.reduceByKey(lambda a,b: a+b) # Reduce by key
```

**Actions** (eager):
```python
count = rdd.count()           # Count elements
first = rdd.first()           # Get first element
data = rdd.collect()          # Return all data to driver
total = rdd.reduce(lambda a,b: a+b)  # Reduce to single value
rdd.saveAsTextFile("hdfs://output")  # Save to file
```

**Lineage** (fault tolerance):
```python
# Lineage graph
rdd1 = sc.textFile("data.txt")           # Read from HDFS
rdd2 = rdd1.flatMap(lambda x: x.split()) # Split words
rdd3 = rdd2.map(lambda x: (x, 1))        # Create pairs
rdd4 = rdd3.reduceByKey(lambda a,b: a+b) # Count

# If rdd4 partition lost:
# Recompute: textFile → flatMap → map → reduceByKey
```

**Persistence** (caching):
```python
rdd = sc.textFile("large_file.txt")
rdd.cache()  # Keep in memory

# First action: computes and caches
count1 = rdd.count()

# Second action: reads from cache (fast!)
count2 = rdd.filter(lambda x: "error" in x).count()
```

**Partitioning**:
```python
# Default partitioning
rdd = sc.parallelize(range(100), numSlices=10)  # 10 partitions

# Custom partitioning (for skewed data)
rdd2 = rdd.partitionBy(20, lambda x: hash(x) % 20)
```

---

## Part 3: Distributed Training

### What's data parallelism?

**Data parallelism**: Split data across devices, each device has full model copy.

**How it works**:
1. Each device has complete model
2. Split batch across devices
3. Each device computes gradients on its batch
4. Synchronize gradients (all-reduce)
5. All devices update model with averaged gradients

**Example** (4 GPUs):
```
Batch size: 128
Split: 32 per GPU

GPU 0: Model copy + Data[0:32]   → Gradients_0
GPU 1: Model copy + Data[32:64]  → Gradients_1
GPU 2: Model copy + Data[64:96]  → Gradients_2
GPU 3: Model copy + Data[96:128] → Gradients_3

All-reduce: Average gradients
All GPUs: Update model with averaged gradients
```

**Code** (PyTorch DDP):
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group("nccl")

# Wrap model
model = MyModel()
model = DDP(model, device_ids=[local_rank])

# Training loop (same as single GPU!)
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()  # Gradients automatically synchronized!
    optimizer.step()
```

**Advantages**:
- Simple to implement
- Scales well for large batches
- No model code changes

**Limitations**:
- Model must fit on single device
- Communication overhead (gradient sync)
- Batch size scaling (large batch → worse generalization)

**Effective batch size**:
```
4 GPUs × 32 per GPU = 128 effective batch size

May need to adjust learning rate:
lr_scaled = lr_base × sqrt(num_gpus)
```

### What's model parallelism?

**Model parallelism**: Split model across devices, each device has partial model.

**When to use**:
- Model too large for single device
- Example: GPT-3 (175B params ≈ 700GB) doesn't fit on 80GB GPU

**Types**:

**1. Tensor parallelism**: Split individual layers
```
Large matrix multiplication: Y = XW

Split W across 2 GPUs:
W = [W1 | W2]

GPU 0: Y1 = X @ W1
GPU 1: Y2 = X @ W2

Concatenate: Y = [Y1 | Y2]
```

**2. Pipeline parallelism**: Split layers across devices
```
Model: Layer1 → Layer2 → Layer3 → Layer4

GPU 0: Layer1
GPU 1: Layer2
GPU 2: Layer3
GPU 3: Layer4

Forward: GPU0 → GPU1 → GPU2 → GPU3
Backward: GPU3 → GPU2 → GPU1 → GPU0
```

**Example** (simple pipeline):
```python
# Model split
class ModelPart1(nn.Module):  # On GPU 0
    def __init__(self):
        self.layer1 = nn.Linear(1000, 1000)

class ModelPart2(nn.Module):  # On GPU 1
    def __init__(self):
        self.layer2 = nn.Linear(1000, 10)

# Forward pass
x = x.to('cuda:0')
x = model_part1(x)
x = x.to('cuda:1')  # Transfer to GPU 1
x = model_part2(x)
```

**Challenges**:
- **Pipeline bubbles**: GPUs idle waiting for data
- **Communication**: Data transfer between devices
- **Load balancing**: Uneven layer sizes

**Solution** (microbatching):
```
Instead of: [Batch1_GPU0] → [Batch1_GPU1] → [Batch1_GPU2] (GPUs idle)

Microbatch:
[B1_GPU0] → [B1_GPU1] → [B1_GPU2]
            [B2_GPU0] → [B2_GPU1] → [B2_GPU2]
                        [B3_GPU0] → [B3_GPU1] → ...

Overlaps computation, reduces idle time
```

### What's pipeline parallelism?

**Pipeline parallelism**: Split model into stages, process multiple microbatches concurrently.

**Problem with naive model parallelism**:
```
4-layer model on 4 GPUs (sequential):

Time →
GPU0: [Fwd L1]              [Bwd L1]
GPU1:         [Fwd L2]              [Bwd L2]
GPU2:                 [Fwd L3]              [Bwd L3]
GPU3:                         [Fwd L4]              [Bwd L4]

Pipeline bubble: 75% GPU idle!
```

**Solution**: Microbatching
```
Split batch into 4 microbatches:

Time →
GPU0: [F1] [F2] [F3] [F4]     [B4] [B3] [B2] [B1]
GPU1:     [F1] [F2] [F3] [F4] [B4] [B3] [B2] [B1]
GPU2:         [F1] [F2] [F3] [F4] [B4] [B3] [B2]
GPU3:             [F1] [F2] [F3] [F4] [B4] [B3]

Much less idle time!
```

**GPipe** (Google):
- Automatic microbatching
- Re-compute activations to save memory
- Trade computation for memory

**Code** (conceptual):
```python
# Naive (75% bubble)
for batch in dataloader:
    x = layer1(x)  # GPU 0
    x = layer2(x)  # GPU 1  (GPU 0 idle!)
    x = layer3(x)  # GPU 2  (GPU 0,1 idle!)
    x = layer4(x)  # GPU 3  (GPU 0,1,2 idle!)

# Pipeline (25% bubble)
microbatches = split_batch(batch, num_microbatches=4)
for mb in microbatches:
    # All GPUs working on different microbatches
    async_forward(mb)
```

**Efficiency**:
```
Ideal: 4 GPUs → 4× speedup

Naive model parallelism:
- Pipeline bubble = 75%
- Actual speedup: 1.3×

Pipeline parallelism (8 microbatches):
- Pipeline bubble = 12.5%
- Actual speedup: 3.5×
```

**Trade-offs**:
- More microbatches → less bubble, but more memory (need to store intermediate activations)
- Gradient accumulation across microbatches

### What's PyTorch DDP?

**PyTorch DDP** (DistributedDataParallel): PyTorch's data parallel training across multiple GPUs/nodes.

**Key features**:
1. **Automatic gradient synchronization**: All-reduce during backward
2. **Efficient**: Overlaps communication with computation
3. **Multi-node**: Works across machines
4. **Fault tolerance**: Can resume from checkpoints

**Setup**:
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. Initialize process group
dist.init_process_group(
    backend='nccl',  # NCCL for GPUs, gloo for CPUs
    init_method='env://',  # Use environment variables
    world_size=4,     # Total processes
    rank=rank         # This process's rank (0-3)
)

# 2. Set device
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')

# 3. Create model and wrap with DDP
model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank])

# 4. Use DistributedSampler for data
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset, num_replicas=4, rank=rank)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

# 5. Training loop (looks like single GPU!)
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Shuffle differently each epoch
    for batch in dataloader:

### What's Ray and how does it differ from Spark?

**Ray**: A unified framework for scaling AI and Python applications.

**Key Features**:
1.  **Actor Pattern**: Stateful distributed objects (unlike Spark's stateless tasks).
2.  **Task-based**: Fine-grained, dynamic task execution.
3.  **Ecosystem**: Ray Train (distributed training), Ray Tune (hyperparameter tuning), Ray Serve (model serving), RLlib (reinforcement learning).

**Ray vs Spark**:

| Feature | Spark | Ray |
| :--- | :--- | :--- |
| **Core Abstraction** | RDD (Data) | Actor/Task (Compute) |
| **State** | Stateless (mostly) | Stateful (Actors) |
| **Granularity** | Coarse (dataset partitions) | Fine (individual functions/classes) |
| **Best For** | Data processing, ETL, SQL | ML training, RL, Serving, Heterogeneous tasks |
| **Scheduling** | Centralized, bulk | Decentralized, low latency |

**Code Example (Ray Remote Function)**:
```python
import ray
ray.init()

@ray.remote
def square(x):
    return x * x

# Launch parallel tasks
futures = [square.remote(i) for i in range(4)]
# Retrieve results
print(ray.get(futures))  # [0, 1, 4, 9]
```

**Code Example (Ray Actor)**:
```python
@ray.remote(num_gpus=1)
class Counter:
    def __init__(self):
        self.value = 0
    def increment(self):
        self.value += 1
        return self.value

counter = Counter.remote()
print(ray.get(counter.increment.remote()))
```

### What's Horovod?

**Horovod**: Distributed deep learning training framework (Uber) supporting TensorFlow, Keras, PyTorch, and Apache MXNet.

**Core Concept**: Based on **MPI (Message Passing Interface)** and the **Ring All-Reduce** algorithm.

**Key Features**:
1.  **Easy Integration**: Minimal code changes to existing scripts.
2.  **Efficient**: Uses efficient inter-GPU communication (NCCL) and Ring All-Reduce.
3.  **Platform Agnostic**: Works with multiple DL frameworks.

**How it works**:
- Wraps the optimizer.
- Averages gradients across all workers using Ring All-Reduce after each step.
- Broadcasts initial model weights to all workers to ensure synchronization.

**Code Example (PyTorch with Horovod)**:
```python
import torch
import horovod.torch as hvd

# 1. Initialize Horovod
hvd.init()

# 2. Pin GPU to local rank
torch.cuda.set_device(hvd.local_rank())

# 3. Partition dataset
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

# 4. Wrap Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01 * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

# 5. Broadcast initial variables
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
```

**Horovod vs DDP**:
- **DDP**: Built into PyTorch, usually faster for PyTorch-only workflows, no MPI dependency.
- **Horovod**: Framework-agnostic, easier to set up on some HPC clusters with MPI support.
        batch = batch.to(device)

        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, labels)
        loss.backward()  # Gradients automatically all-reduced!
        optimizer.step()
```

**How it works**:
```
During backward():
1. Each GPU computes gradients locally
2. DDP hooks into autograd
3. All-reduce gradients across GPUs (average)
4. Each GPU has same averaged gradient
5. optimizer.step() updates model (now synchronized)
```

**Optimization** (gradient bucketing):
```
Instead of:
- Compute all gradients
- Then all-reduce all at once

DDP does:
- Compute gradient for layer N
- Immediately all-reduce gradient for layer N-1 (overlap!)
- Compute gradient for layer N-1
- Immediately all-reduce gradient for layer N-2
...

Hides communication latency!
```

**Launch** (4 GPUs):
```bash
# Single node, 4 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py

# Multi-node (2 nodes × 4 GPUs = 8 GPUs)
# Node 0:
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=1234 \
    train.py

# Node 1:
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=1234 \
    train.py
```

### What's all-reduce?

**All-reduce**: Collective operation where all processes contribute data and receive the result.

**Goal**: Each process has a value, compute reduction (sum/average) and send result to all processes.

**Example** (sum):
```
4 GPUs, each has a gradient tensor

Before:
GPU 0: [1, 2, 3]
GPU 1: [4, 5, 6]
GPU 2: [7, 8, 9]
GPU 3: [10, 11, 12]

After all-reduce (sum):
GPU 0: [22, 26, 30]  # 1+4+7+10, 2+5+8+11, 3+6+9+12
GPU 1: [22, 26, 30]
GPU 2: [22, 26, 30]
GPU 3: [22, 26, 30]
```

**Naive approach** (broadcast):
```
1. Send all gradients to GPU 0
2. GPU 0 computes sum
3. GPU 0 broadcasts result to all

Communication: 3N (receive) + 3N (broadcast) = 6N
Bottleneck: GPU 0 bandwidth
```

**Better: Ring all-reduce** (see next question)

**Variants**:
- **all-reduce**: All get result
- **reduce**: Only one process gets result
- **broadcast**: One sends to all
- **all-gather**: Each sends to all (concatenate)
- **reduce-scatter**: Reduce then scatter pieces

**In ML training**:
```python
# Pseudocode for distributed SGD
for batch in dataloader:
    # Each GPU computes gradients on its batch
    gradients = compute_gradients(batch)

    # All-reduce: average gradients across GPUs
    avg_gradients = all_reduce(gradients, op=AVERAGE)

    # All GPUs update model with averaged gradients
    model.update(avg_gradients)
```

**Frameworks**:
- **NCCL**: NVIDIA's optimized GPU communication
- **MPI**: Message Passing Interface (general-purpose)
- **Gloo**: Facebook's CPU/GPU communication

### What's ring all-reduce?

**Ring all-reduce**: Efficient all-reduce algorithm that avoids bandwidth bottlenecks.

**Problem with naive all-reduce**:
```
Send all to one GPU → bottleneck on that GPU's bandwidth
```

**Ring all-reduce solution**:
Arrange GPUs in a ring, perform reduce-scatter then all-gather.

**Algorithm** (4 GPUs, simplified):

**Phase 1: Reduce-Scatter**
```
Each GPU has array split into N chunks (N = num GPUs)

GPU 0: [A0, A1, A2, A3]
GPU 1: [B0, B1, B2, B3]
GPU 2: [C0, C1, C2, C3]
GPU 3: [D0, D1, D2, D3]

N-1 steps, each GPU sends one chunk to next:

Step 1:
GPU 0 → GPU 1: A3    GPU 1 → GPU 2: B0    etc.
GPU 1 computes: B3' = A3 + B3

Step 2:
GPU 0 → GPU 1: A2    GPU 1 → GPU 2: B3'   etc.
GPU 2 computes: C3' = B3' + C3 = A3+B3+C3

Step 3:
GPU 1 → GPU 2: B2'   (contains A2+B2+C2)
GPU 2 → GPU 3: C3'   (contains A3+B3+C3+D3 - DONE!)

After N-1 steps:
GPU 0: [?, ?, ?, A0+B0+C0+D0]
GPU 1: [A1+B1+C1+D1, ?, ?, ?]
GPU 2: [?, A2+B2+C2+D2, ?, ?]
GPU 3: [?, ?, A3+B3+C3+D3, ?]
```

**Phase 2: All-Gather**
```
Each GPU sends its complete chunk around the ring

After N-1 more steps:
GPU 0: [A0+B0+C0+D0, A1+B1+C1+D1, A2+B2+C2+D2, A3+B3+C3+D3]
GPU 1: [A0+B0+C0+D0, A1+B1+C1+D1, A2+B2+C2+D2, A3+B3+C3+D3]
GPU 2: [A0+B0+C0+D0, A1+B1+C1+D1, A2+B2+C2+D2, A3+B3+C3+D3]
GPU 3: [A0+B0+C0+D0, A1+B1+C1+D1, A2+B2+C2+D2, A3+B3+C3+D3]
```

**Complexity**:
```
Data size: S
Num GPUs: N

Naive all-reduce:
- Communication: (N-1) × S per GPU
- Bandwidth bottleneck: One GPU receives (N-1)S

Ring all-reduce:
- Communication: 2(N-1) × S/N = 2S(N-1)/N per GPU
- No bottleneck: All links used equally
- Latency: 2(N-1) steps

For large S: Ring is N/2 times more efficient!
```

**Why it works**:
- No single GPU is a bottleneck
- All GPUs send/receive simultaneously
- Bandwidth-optimal for large messages

**Visualization** (4 GPUs):
```
     GPU 0
      ↑ ↓
GPU 3 ← → GPU 1
      ↑ ↓
     GPU 2

Each arrow is a bidirectional link
All links used simultaneously
```

---

## Part 4: Fault Tolerance & Reliability

### What's fault tolerance?

**Fault tolerance**: System's ability to continue operating despite failures.

**Types of failures**:
1. **Node failure**: Machine crashes
2. **Network partition**: Nodes can't communicate
3. **Data corruption**: Bit flips, disk errors
4. **Byzantine failures**: Nodes behave maliciously

**Fault tolerance techniques**:

**1. Replication**:
```
Data replicated on 3 nodes
1 node fails → data still available on 2 others
```

**2. Checkpointing**:
```
Save state every N steps
Failure → restart from last checkpoint
```

**3. Redundant computation**:
```
Compute task on 2 nodes
Take first result, ignore second
(Expensive but fast recovery)
```

**4. Heartbeats**:
```
Nodes send "I'm alive" messages
No heartbeat → assume failure
```

**ML training example**:
```python
# Fault tolerance in distributed training
for epoch in range(num_epochs):
    try:
        for batch in dataloader:
            loss = train_step(batch)

            # Save checkpoint every 1000 steps
            if step % 1000 == 0:
                save_checkpoint(model, optimizer, epoch, step)

    except NodeFailure:
        # Load last checkpoint
        epoch, step = load_checkpoint(model, optimizer)
        # Resume training
        continue
```

**Trade-offs**:
- **Replication**: Extra storage cost
- **Checkpointing**: Slower (I/O overhead)
- **Redundant computation**: Extra compute cost

### What's checkpointing?

**Checkpointing**: Saving system state periodically to recover from failures.

**What to save**:
1. **Model weights**: parameters
2. **Optimizer state**: momentum, learning rate schedule
3. **Training state**: epoch, step, RNG state
4. **Data state**: position in dataset

**Example** (PyTorch):
```python
def save_checkpoint(model, optimizer, epoch, step, path):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
        'loss': current_loss,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    torch.set_rng_state(checkpoint['rng_state'])
    return checkpoint['epoch'], checkpoint['step']

# Training loop
for epoch in range(start_epoch, num_epochs):
    for step, batch in enumerate(dataloader):
        loss = train_step(batch)

        # Checkpoint every hour or every N steps
        if time.time() - last_checkpoint_time > 3600:
            save_checkpoint(model, optimizer, epoch, step,
                          f'checkpoint_epoch{epoch}_step{step}.pt')
            last_checkpoint_time = time.time()
```

**Checkpointing strategies**:

**1. Time-based**: Every N minutes
```python
if time.time() - last_checkpoint > 3600:  # Every hour
    save_checkpoint()
```

**2. Step-based**: Every N steps
```python
if step % 1000 == 0:  # Every 1000 steps
    save_checkpoint()
```

**3. Loss-based**: When loss improves
```python
if loss < best_loss:
    best_loss = loss
    save_checkpoint()  # Save best model
```

**4. Rotating checkpoints**: Keep last K checkpoints
```python
# Keep last 5 checkpoints
checkpoints = [f'ckpt_{i}.pt' for i in range(5)]
save_checkpoint(checkpoints[step % 5])  # Circular buffer
```

**Distributed checkpointing**:
```python
# Only rank 0 saves to avoid conflicts
if dist.get_rank() == 0:
    save_checkpoint(model, optimizer, epoch, step)

# Or: Each rank saves its model shard
torch.save(model.state_dict(), f'model_rank{rank}.pt')
```

**Async checkpointing** (doesn't block training):
```python
import threading

def async_save_checkpoint(state_dict, path):
    # Copy to CPU in background
    cpu_state = {k: v.cpu() for k, v in state_dict.items()}
    threading.Thread(target=torch.save, args=(cpu_state, path)).start()

# Training continues while checkpoint saves!
```

**Cost analysis**:
```
Model: 1B parameters = 4GB (fp32)
Optimizer (Adam): 8GB (2× for momentum, variance)
Total checkpoint: 12GB

Save time (to SSD): ~10 seconds
Save time (to network storage): ~60 seconds

Checkpoint every 1000 steps:
- If step takes 0.1s: 100s compute, 10s checkpoint → 9% overhead
```

### What's replication?

**Replication**: Storing multiple copies of data across different nodes.

**Why replicate**:
1. **Fault tolerance**: Survive node failures
2. **Read scalability**: Serve reads from multiple replicas
3. **Geographic distribution**: Low latency for users worldwide

**Replication factor**:
```
Replication factor = 3 (common)
Data stored on 3 nodes
Can tolerate 2 node failures
```

**Example** (HDFS):
```
File: 1GB
Block size: 128MB
Replication: 3×

File split into 8 blocks (128MB each)
Each block stored on 3 different nodes

Block 1: Node A, Node C, Node E
Block 2: Node B, Node D, Node F
...

Total storage: 1GB × 3 = 3GB
```

**Replication strategies**:

**1. Synchronous replication**:
```
Write to primary
Primary writes to replicas
Wait for all acks
Return success

Pros: Strong consistency
Cons: High latency (wait for slowest replica)
```

**2. Asynchronous replication**:
```
Write to primary
Return success immediately
Primary replicates to replicas in background

Pros: Low latency
Cons: Potential data loss if primary fails before replication
```

**3. Quorum-based**:
```
Write to W replicas (W < N)
Read from R replicas
If W + R > N: consistent

Example: N=3, W=2, R=2
Write succeeds after 2 acks (faster than sync)
Read from 2 replicas guarantees seeing latest write
```

**Read strategies**:

**1. Read from primary**: Strong consistency, but primary is bottleneck

**2. Read from any replica**: Fast, but may read stale data

**3. Read from nearest replica**: Low latency, eventual consistency

**ML training example**:
```
Training data: 10TB
Stored in HDFS with replication=3

Node failure: Training continues
- HDFS automatically serves data from replica
- Training doesn't notice the failure
```

**Checkpointing with replication**:
```python
# Save checkpoint to distributed filesystem
# (automatically replicated)
torch.save(checkpoint, 'hdfs://checkpoints/model_step1000.pt')

# Even if one node fails, checkpoint is safe
```

### What's the two-phase commit?

**Two-phase commit (2PC)**: Protocol for atomic distributed transactions.

**Goal**: Either all nodes commit transaction or none do (atomicity).

**Phases**:

**Phase 1: Prepare (Voting)**
```
Coordinator: "Can you commit transaction T?"
Node A: "Yes, I'm ready"
Node B: "Yes, I'm ready"
Node C: "No, I cannot" (e.g., constraint violated)
```

**Phase 2: Commit/Abort**
```
If all voted YES:
    Coordinator: "Commit T"
    All nodes: Commit and release locks
Else:
    Coordinator: "Abort T"
    All nodes: Rollback and release locks
```

**Example** (bank transfer):
```
Transaction: Transfer $100 from Account A to Account B

Account A on Node 1
Account B on Node 2

Phase 1 (Prepare):
Coordinator → Node 1: "Can you deduct $100 from A?"
Node 1: Check balance, lock account
Node 1 → Coordinator: "Yes"

Coordinator → Node 2: "Can you add $100 to B?"
Node 2: Check account exists, lock account
Node 2 → Coordinator: "Yes"

Phase 2 (Commit):
Coordinator → Node 1: "Commit"
Node 1: A = A - 100, unlock

Coordinator → Node 2: "Commit"
Node 2: B = B + 100, unlock

Both committed → Transaction successful
```

**Failure scenarios**:

**Participant failure during prepare**:
```
Node A: "Yes"
Node B: (crashed - no response)

Coordinator: Timeout → Abort
Coordinator → Node A: "Abort"
Node A: Rollback
```

**Coordinator failure after prepare**:
```
All nodes voted "Yes"
Coordinator crashes before sending commit

Nodes: Stuck holding locks!
Solution: Nodes timeout and abort
        OR: New coordinator elected (reads prepare log)
```

**Problems with 2PC**:
1. **Blocking**: Nodes wait for coordinator (holding locks)
2. **Single point of failure**: Coordinator
3. **Not partition-tolerant**: Network split → stuck

**Alternatives**:
- **3-phase commit**: Adds pre-commit phase (reduces blocking)
- **Paxos/Raft**: Consensus protocols (more fault-tolerant)
- **Eventual consistency**: Avoid distributed transactions entirely

**ML context**:
```
Less common in ML training (usually use eventual consistency)

But used in:
- Distributed checkpointing (ensure all workers save atomically)
- Parameter server updates (ensure consistency)
```

### How do you handle node failures?

**Node failure handling** depends on system architecture:

**1. Stateless nodes** (easy):
```
Web servers, stateless workers

Failure:
- Load balancer detects failure (health check)
- Routes traffic to other nodes
- Optionally: Launch replacement node

No data loss (no state)
```

**2. Stateful nodes** (harder):
```
Database, distributed training

Failure:
- Detect failure (heartbeat timeout)
- Recover state (from checkpoint or replica)
- Reassign work to other nodes
```

**Detection**:

**Heartbeats**:
```python
# Worker sends heartbeat every 10s
last_heartbeat = time.time()

# Coordinator checks
if time.time() - last_heartbeat > 30:
    # Worker considered dead
    handle_failure(worker_id)
```

**Recovery strategies**:

**Strategy 1: Restart from checkpoint**
```
Training on 8 GPUs
GPU 3 fails

Action:
1. Detect failure
2. All GPUs load last checkpoint
3. Resume training

Cost: Lost progress since last checkpoint
```

**Strategy 2: Elastic training** (continue with fewer nodes)
```python
# PyTorch elastic
# Can shrink/grow number of workers

8 GPUs → GPU 3 fails → Continue with 7 GPUs
Adjust batch size or learning rate
```

**Strategy 3: Redundant computation**
```
Compute task on 2 nodes
Use result from whoever finishes first
If one fails, still have the other

Expensive: 2× compute
Used in: Google's BigTable, critical systems
```

**Strategy 4: Replication** (for data)
```
Data replicated 3×
Node with data fails
Read from replica instead

HDFS automatically handles this
```

**ML training example**:
```python
import torch.distributed as dist

def train_with_fault_tolerance():
    try:
        for epoch in range(num_epochs):
            for batch in dataloader:
                loss = train_step(batch)

                # Checkpoint periodically
                if step % 1000 == 0:
                    save_checkpoint()

    except RuntimeError as e:
        if "NCCL" in str(e):  # Communication failure
            # Node failed during training

            # Option 1: Restart all from checkpoint
            load_checkpoint()
            dist.init_process_group(...)  # Re-init with fewer nodes

            # Option 2: Exit and let job scheduler restart
            sys.exit(1)
```

**Kubernetes example** (automatic restart):
```yaml
apiVersion: v1
kind: Pod
spec:
  restartPolicy: Always  # Automatically restart on failure
  containers:
  - name: worker
    image: training:latest
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      periodSeconds: 30
      failureThreshold: 3
```

---

## Part 5: Communication Patterns

### What's a parameter server?

**Parameter server**: Centralized architecture where workers fetch/update shared parameters from servers.

**Architecture**:
```
         Parameter Servers
         [PS 1] [PS 2] [PS 3]
            ↑ ↓    ↑ ↓    ↑ ↓
         Workers
         [W1] [W2] [W3] [W4] [W5]
```

**How it works**:
```
1. Worker pulls parameters from PS
2. Worker computes gradients on local data
3. Worker pushes gradients to PS
4. PS updates parameters
5. Repeat
```

**Example** (distributed SGD):
```python
# Parameter server (simplified)
class ParameterServer:
    def __init__(self, model):
        self.params = model.parameters()

    def pull(self):
        return self.params

    def push(self, gradients):
        # Update parameters
        for param, grad in zip(self.params, gradients):
            param -= learning_rate * grad

# Worker
def worker():
    while True:
        # Pull latest parameters
        params = ps.pull()

        # Compute gradients
        batch = get_batch()
        gradients = compute_gradients(params, batch)

        # Push gradients
        ps.push(gradients)
```

**Synchronous PS**:
```
PS waits for all workers before updating

Step:
1. All workers pull params (version v)
2. All workers compute gradients
3. PS waits for all gradients
4. PS updates params (version v+1)
5. Repeat

Pros: Consistent (all workers use same version)
Cons: Slow workers block everyone (stragglers)
```

**Asynchronous PS**:
```
PS updates immediately when receiving gradients

Step:
1. Worker 1 pulls params (version v)
2. Worker 2 pulls params (version v+3)  # Stale!
3. Worker 1 pushes gradients → PS updates to v+1
4. Worker 2 pushes gradients → PS updates to v+4
5. Workers pull different versions

Pros: No waiting, faster
Cons: Stale gradients, potential divergence
```

**Partitioning** (large models):
```
Model: 10B parameters
3 parameter servers

PS 1: Parameters[0:3.3B]
PS 2: Parameters[3.3B:6.6B]
PS 3: Parameters[6.6B:10B]

Workers pull/push to all 3 PS
```

**Advantages**:
- Simple to implement
- Easy to scale workers
- Handles dynamic worker pool

**Disadvantages**:
- PS can be bottleneck (network bandwidth)
- Single point of failure (mitigated by replication)
- Not as efficient as all-reduce for dense updates

**Modern alternative**: All-reduce (Ring all-reduce)
- More efficient for dense gradients
- No central bottleneck
- Used in PyTorch DDP, Horovod

### What's peer-to-peer communication?

**Peer-to-peer (P2P)**: Nodes communicate directly without central coordinator.

**vs. Parameter Server**:
```
Parameter Server:
Worker ↔ PS ↔ Worker
Centralized, PS can be bottleneck

Peer-to-peer:
Worker ↔ Worker ↔ Worker
Decentralized, no bottleneck
```

**Topologies**:

**1. Ring**:
```
W0 → W1 → W2 → W3 → W0

Each worker communicates with 2 neighbors
Used in: Ring all-reduce
```

**2. Tree**:
```
        W0
       ↙  ↘
     W1    W2
    ↙  ↘  ↙  ↘
  W3  W4 W5  W6

Reduce up the tree, broadcast down
Used in: Hierarchical all-reduce
```

**3. Mesh** (full connectivity):
```
W0 ↔ W1
↕  ↗ ↕
W2 ↔ W3

Every worker can talk to every other worker
Expensive: N² connections
```

**Example** (gossip protocol):
```python
# Each worker periodically shares state with random peer

def gossip_step():
    # Select random peer
    peer = random.choice(all_workers)

    # Exchange information
    my_state = get_local_state()
    peer_state = peer.get_state()

    # Merge states (e.g., average)
    new_state = merge(my_state, peer_state)
    update_local_state(new_state)

# Eventually all workers converge to same state
```

**Decentralized SGD**:
```python
# Each worker maintains model
# Periodically synchronize with neighbors

for epoch in range(num_epochs):
    for step in range(steps_per_epoch):
        # Local SGD step
        gradients = compute_gradients(batch)
        model.update(gradients)

        # Periodic synchronization
        if step % sync_freq == 0:
            # Average with neighbors
            neighbor_models = [get_model(n) for n in neighbors]
            model = average_models([model] + neighbor_models)
```

**Advantages**:
- No central bottleneck
- Fault tolerant (no single point of failure)
- Scalable

**Disadvantages**:
- Complex to implement
- Synchronization more difficult
- Network topology matters

**Use cases**:
- **Ring all-reduce**: Distributed training
- **BitTorrent**: File sharing
- **Blockchain**: Decentralized ledger
- **Gossip protocols**: Eventual consistency

### What's collective communication?

**Collective communication**: Operations involving all processes in a group.

**Types**:

**1. Broadcast**:
```
One sends to all

Before:
Rank 0: [1, 2, 3]
Rank 1: [?, ?, ?]
Rank 2: [?, ?, ?]

After broadcast from rank 0:
Rank 0: [1, 2, 3]
Rank 1: [1, 2, 3]
Rank 2: [1, 2, 3]
```

**2. Reduce**:
```
All send to one, which computes reduction

Before:
Rank 0: [1, 2, 3]
Rank 1: [4, 5, 6]
Rank 2: [7, 8, 9]

After reduce (sum) to rank 0:
Rank 0: [12, 15, 18]  # 1+4+7, 2+5+8, 3+6+9
Rank 1: [4, 5, 6]     # Unchanged
Rank 2: [7, 8, 9]     # Unchanged
```

**3. All-reduce**:
```
All contribute, all receive result

Before:
Rank 0: [1, 2, 3]
Rank 1: [4, 5, 6]
Rank 2: [7, 8, 9]

After all-reduce (sum):
Rank 0: [12, 15, 18]
Rank 1: [12, 15, 18]
Rank 2: [12, 15, 18]
```

**4. Gather**:
```
One collects from all

Before:
Rank 0: [1, 2]
Rank 1: [3, 4]
Rank 2: [5, 6]

After gather to rank 0:
Rank 0: [1, 2, 3, 4, 5, 6]
Rank 1: [3, 4]
Rank 2: [5, 6]
```

**5. All-gather**:
```
All collect from all

Before:
Rank 0: [1, 2]
Rank 1: [3, 4]
Rank 2: [5, 6]

After all-gather:
Rank 0: [1, 2, 3, 4, 5, 6]
Rank 1: [1, 2, 3, 4, 5, 6]
Rank 2: [1, 2, 3, 4, 5, 6]
```

**6. Scatter**:
```
One sends different data to each

Before:
Rank 0: [1, 2, 3, 4, 5, 6]
Rank 1: [?, ?]
Rank 2: [?, ?]

After scatter from rank 0:
Rank 0: [1, 2]
Rank 1: [3, 4]
Rank 2: [5, 6]
```

**PyTorch example**:
```python
import torch.distributed as dist

# All-reduce (sum)
tensor = torch.tensor([rank])
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# tensor now contains sum of all ranks

# Broadcast
if rank == 0:
    tensor = torch.tensor([1, 2, 3])
else:
    tensor = torch.zeros(3)
dist.broadcast(tensor, src=0)
# All ranks now have [1, 2, 3]

# Gather
tensor = torch.tensor([rank])
if rank == 0:
    gather_list = [torch.zeros(1) for _ in range(world_size)]
    dist.gather(tensor, gather_list, dst=0)
    # gather_list = [[0], [1], [2], ...]
else:
    dist.gather(tensor, dst=0)
```

**ML training usage**:
```python
# Distributed SGD uses all-reduce

# Each worker computes gradients
gradients = compute_gradients(batch)

# All-reduce to average gradients
dist.all_reduce(gradients, op=dist.ReduceOp.AVG)

# All workers update with averaged gradients
model.update(gradients)
```

### What's gradient accumulation?

**Gradient accumulation**: Accumulate gradients over multiple mini-batches before updating.

**Why**:
1. **Simulate larger batch**: GPU memory limited
2. **Pipeline parallelism**: Accumulate over microbatches
3. **Gradient compression**: Reduce communication frequency

**Without accumulation**:
```python
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()  # Update every batch
```

**With accumulation**:
```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps  # Normalize
    loss.backward()  # Accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()        # Update every 4 batches
        optimizer.zero_grad()   # Reset gradients
```

**Effective batch size**:
```
Per-GPU batch: 32
Accumulation steps: 4
Num GPUs: 8

Effective batch = 32 × 4 × 8 = 1024
```

**Example** (large model training):
```
Model: 10B parameters
GPU memory: 80GB
Max batch per GPU: 1 (!)

Solution: Accumulate over 64 steps
Effective batch: 1 × 64 = 64 (reasonable)
```

**Distributed training + accumulation**:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model)
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()  # Gradients synchronized by DDP!

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Trade-offs**:
```
Advantages:
+ Simulate large batch with limited memory
+ Reduce communication (update less frequently)
+ Same math as large batch

Disadvantages:
- Slower (more forward/backward passes per update)
- Same memory for activations (still need batch=1 activations)
- Effective batch size affects generalization
```

**Pipeline parallelism**:
```
Microbatches: Split batch into 4 microbatches

Forward/backward on microbatch 1 → accumulate gradients
Forward/backward on microbatch 2 → accumulate gradients
Forward/backward on microbatch 3 → accumulate gradients
Forward/backward on microbatch 4 → accumulate gradients
Update parameters

Enables pipeline to stay full!
```

### What's asynchronous vs synchronous training?

**Synchronous training**: All workers wait for each other before updating.

**Asynchronous training**: Workers update independently, no waiting.

**Synchronous SGD**:
```
Step:
1. All workers compute gradients on different batches
2. All-reduce: Average gradients across workers
3. All workers wait for all-reduce to complete
4. All workers update model (now synchronized)
5. Repeat

Timeline:
Worker 0: [Compute] → [Wait for all-reduce] → [Update]
Worker 1: [Compute] → [Wait for all-reduce] → [Update]
Worker 2: [Compute.................] → [Wait] → [Update]
                                        ↑ Fast workers wait for slow!
```

**Asynchronous SGD**:
```
Step:
1. Worker pulls latest parameters from parameter server
2. Worker computes gradients
3. Worker pushes gradients to parameter server
4. Parameter server updates immediately (no waiting)
5. Worker repeats (may pull stale parameters)

Timeline:
Worker 0: [Pull v=1] → [Compute] → [Push] → [Pull v=5] → [Compute] → ...
Worker 1: [Pull v=2] → [Compute.......] → [Push] → [Pull v=6] → ...
Worker 2: [Pull v=3] → [Compute] → [Push] → [Pull v=7] → ...
                                     ↑ No waiting!
```

**Comparison**:

| Aspect | Synchronous | Asynchronous |
|--------|-------------|--------------|
| **Consistency** | All workers use same parameters | Stale gradients |
| **Speed** | Limited by slowest worker | No waiting |
| **Convergence** | Stable | Can diverge |
| **Stragglers** | Major problem | No impact |
| **Implementation** | All-reduce (DDP) | Parameter server |
| **Communication** | High (all-reduce every step) | Lower (no sync) |

**Stale gradients problem** (async):
```
Worker pulls parameters version 100
Worker computes gradients (takes time)
Meanwhile, parameters updated to version 105
Worker pushes gradients (based on version 100!)
Update applied to version 105 → stale gradient

If staleness is large: can diverge or slow convergence
```

**Solutions for staleness**:

**1. Staleness threshold**:
```python
max_staleness = 10

if current_version - worker_version > max_staleness:
    reject_gradient()  # Too stale
```

**2. Elastic averaging SGD (EASGD)**:
```python
# Periodic synchronization
local_steps = 10

for step in range(local_steps):
    # Local updates (async)
    gradients = compute_gradients()
    model.update(gradients)

# Sync: Average local models with global model
model = alpha * model + (1 - alpha) * global_model
```

**3. Learning rate scaling**:
```python
# Reduce learning rate for stale gradients
age = current_version - gradient_version
lr_scaled = lr / (1 + age)
```

**When to use**:

**Synchronous**:
- Stable convergence critical
- Workers have similar speed
- Small-scale (stragglers manageable)
- **Most common for ML**

**Asynchronous**:
- Large-scale (1000s of workers)
- Heterogeneous workers (different speeds)
- Can tolerate some divergence
- Need maximum throughput

**Hybrid** (best of both worlds):
```python
# Synchronous within a node (8 GPUs)
# Asynchronous across nodes (100 nodes)

Within node: DDP (synchronous all-reduce)
Across nodes: Delayed gradient averaging (async)
```

---

**End of Distributed Systems Q&A**
