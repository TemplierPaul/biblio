# Multi-Task Multi-Behavior MAP-Elites (MTMB-ME)

## Overview

**MTMB-MAP-Elites** is a Quality-Diversity algorithm that finds **many diverse high-quality solutions for many tasks**, combining the strengths of MAP-Elites (diversity within a task) and Multi-Task MAP-Elites (leveraging similarity across tasks).

**Authors**: Timothée Anne, Jean-Baptiste Mouret (Inria Nancy - Grand Est, LORIA, CNRS)
**Paper**: "Multi-Task Multi-Behavior MAP-Elites"
**Venue**: Preprint, 2023
**ArXiv**: 2305.01264

---

## Core Problem

### Problem Formulation

Given a **set of tasks** $T_{1:n} \in \mathcal{T}$, find **as many diverse solutions as possible for each task**:

$$\begin{array}{l}
\text{maximize} \sum_{i=1}^{n} m_i \\
\text{s.t. } \forall i, \forall j \le m_i, \text{fitness}(T_i, c_i^j) = f_{max} \\
\text{s.t. } \forall i, \forall k \ne j, \mathcal{F}(T_i, c_i^j) \ne \mathcal{F}(T_i, c_i^k)
\end{array}$$

where:
- $m_i$ = number of solutions found for task $T_i$
- $c_i^j$ = $j$-th command (solution) for task $T_i$
- $f_{max}$ = maximum fitness (optimal)
- $\mathcal{F}$ = transition function mapping (task, command) → behavior

**Constraints**:
1. All solutions must have optimal fitness
2. All solutions must have different behaviors (diversity)

### Motivating Example: Robot Grasping

**Problem**: Find various grasps for different objects

- **Task**: Specific object to be grasped
- **Solutions**: Different grasps for that object
- **Why diversity?**: Choose appropriate grasp when object is partially obstructed

### Comparison to Existing Approaches

| Method | Diversity | Multi-Task | Output |
|--------|-----------|------------|--------|
| **MAP-Elites** | ✅ Many diverse solutions | ❌ Single task | Archive for 1 task |
| **Multi-Task MAP-Elites** | ❌ One solution per task | ✅ Many tasks | 1 solution × n tasks |
| **MTMB-MAP-Elites** | ✅ Many diverse solutions | ✅ Many tasks | Archive for each task |

---

## Algorithm

### Core Idea

**Leverage similarity between tasks** to find diverse solutions more efficiently than running MAP-Elites independently on each task.

**Key insight**: Tasks often share partial solutions → crossover between tasks explores promising regions faster than random search.

### Complete Algorithm (Implementable)

```python
# INITIALIZATION
budget = B  # e.g., 25,000
n_tasks = n  # e.g., 200
tasks = sample_random_tasks(task_space, n_tasks)

# Initialize archives (one per task)
# archives[task_id][behavior] = (command, fitness)
archives = {task_id: {} for task_id in range(n_tasks)}

# Random initialization until n elites
evals_used = 0
while count_total_elites(archives) < n_tasks and evals_used < budget:
    task = random.choice(tasks)
    command = random_command(command_space)

    behavior = F(task, command)  # Transition function
    fitness = fitness_function(task, command)

    # Add to archive (always accept during initialization)
    if behavior not in archives[task_id]:
        archives[task_id][behavior] = (command, fitness)

    evals_used += 1

# MAIN LOOP
while evals_used < budget:

    # SELECT COMMAND (crossover between 2 tasks)
    # 1. Pick 2 random tasks that have at least one elite
    tasks_with_elites = [t for t in range(n_tasks)
                         if len(archives[t]) > 0]
    task_i = random.choice(tasks_with_elites)
    task_j = random.choice(tasks_with_elites)

    # 2. Pick random elite from each task
    behavior_i = random.choice(list(archives[task_i].keys()))
    behavior_j = random.choice(list(archives[task_j].keys()))
    command_i = archives[task_i][behavior_i][0]
    command_j = archives[task_j][behavior_j][0]

    # 3. Apply crossover + mutation
    command = crossover_and_mutate(command_i, command_j)

    # SELECT TASK (uniformly random)
    task_k = random.choice(range(n_tasks))

    # EVALUATE
    behavior = F(tasks[task_k], command)
    fitness = fitness_function(tasks[task_k], command)
    evals_used += 1

    # UPDATE ARCHIVE
    if behavior not in archives[task_k]:
        # New behavior → add
        archives[task_k][behavior] = (command, fitness)
    elif fitness > archives[task_k][behavior][1]:
        # Better fitness for existing behavior → replace
        archives[task_k][behavior] = (command, fitness)

    # (else: worse or equal fitness for existing behavior → discard)

return archives
```

### Detailed Operators

**1. Task Sampling** (initialization):
```python
def sample_random_tasks(task_space, n):
    """
    Sample n random tasks from continuous task space.

    For fault recovery example:
    - Sample robot postures
    - Sample wall configurations
    - Sample fault types
    - Duplicate each for 2 hand modes (right-only, both)
    """
    situations = []
    for _ in range(n // 2):  # n/2 situations
        posture = sample_random_posture()
        wall = sample_random_wall_config()
        fault = sample_random_fault()
        situations.append((posture, wall, fault))

    tasks = []
    for situation in situations:
        tasks.append(Task(situation, hand_mode='right'))
        tasks.append(Task(situation, hand_mode='both'))

    return tasks
```

**2. Crossover and Mutation**:
```python
def crossover_and_mutate(command_i, command_j,
                         mutation_prob=0.1, mutation_std=0.1):
    """
    Traditional genetic algorithm crossover + mutation.

    Args:
        command_i, command_j: Parent commands (arrays)
        mutation_prob: Probability of mutating each dimension
        mutation_std: Standard deviation of Gaussian mutation

    Returns:
        Offspring command
    """
    # CROSSOVER: Uniform crossover (50/50 per dimension)
    offspring = []
    for dim_i, dim_j in zip(command_i, command_j):
        if random.random() < 0.5:
            offspring.append(dim_i)
        else:
            offspring.append(dim_j)

    offspring = np.array(offspring)

    # MUTATION: Gaussian noise with probability
    for i in range(len(offspring)):
        if random.random() < mutation_prob:
            offspring[i] += np.random.normal(0, mutation_std)
            offspring[i] = np.clip(offspring[i], 0, 1)  # Clip to bounds

    return offspring
```

**3. Behavior Discretization**:
```python
def compute_behavior(reached_positions, grid_size=0.2):
    """
    Convert continuous reached positions to discrete behavior.

    Args:
        reached_positions: [(x1, z1), (x2, z2)] for both hands
                          or [(x1, z1)] for right-hand only
        grid_size: Size of discretization squares (e.g., 20cm = 0.2m)

    Returns:
        Behavior tuple (grid indices)
    """
    behavior = []
    for (x, z) in reached_positions:
        grid_x = int(x / grid_size)
        grid_z = int(z / grid_size)
        behavior.append((grid_x, grid_z))

    return tuple(behavior)  # Hashable for dict keys
```

### Key Components

**Task Space** $\mathcal{T}$:
- Defines command space $\mathcal{C}$
- Defines behavior space $\mathcal{B}$
- Provides fitness function: $\text{fitness}: \mathcal{T} \times \mathcal{C} \to \mathbb{R}$
- Provides transition function: $\mathcal{F}: \mathcal{T} \times \mathcal{C} \to \mathcal{B}$

**Terminology**:
- **Elite**: Command stored in archive (may not be optimal)
- **Solution**: Elite with maximum fitness $f_{max}$ (optimal)

### Crossover Strategy

**Why crossover between tasks?**
- Tasks assumed to have similarity
- Elites from different tasks may share useful structure
- Crossover explores combination of successful patterns
- More efficient than random search in command space

**Alternative**: Could use variation within single task (like MAP-Elites), but crossover across tasks leverages multi-task structure.

---

## Experimental Validation

### Domain: Humanoid Robot Fault Recovery

**Scenario**:
- Talos humanoid robot detects fault in leg (combination of amputated, passive, locked joints)
- Fault likely to cause fall
- Wall within arm's reach (known orientation and distance)
- **Goal**: Find successful contact positions on wall to regain balance

**Why diverse solutions?**
- Different faults require different contact strategies
- Repertoire of reflexes for different situations
- Robust to partial observability (unseen faults)

### Task Setup (200 tasks total)

**Task Space** $\mathcal{T}$:
- Robot posture (from reaching random hand targets for 4s)
- Wall configuration (distance and orientation)
- Fault type (6 joints of right leg: passive/locked/amputated combinations)
- Hand mode: right-hand only OR both hands
- **100 situations × 2 hand modes = 200 tasks**

**Command Space** $\mathcal{C} = X \times Z \times X \times Z$:
- **Target** contact positions on wall (high-level commands)
- $X = [x_{min}, x_{max}]$, $Z = [z_{min}, z_{max}]$ (reachable positions)
- Right-hand only: uses first 2 dimensions
- Both hands: uses all 4 dimensions
- Executed via whole-body controller (QP optimization at 500Hz)

**Behavior Space** $\mathcal{B}$:
- **Reached** contact positions on wall (not target!)
- Discretized into 20cm × 20cm squares
- 2D for right-hand only, 4D for both hands
- Empty map initialized, behaviors appended as discovered
- **Note**: Reached ≠ target due to model mismatch (undamaged model vs damaged robot)

**Fitness Function**:
- Time before simulation stops (max 10s)
- Stop conditions: auto-collision, fall (unplanned contact), or timeout
- $f_{max} = 10s$ (optimal = survive full timeout)
- Timeout chosen to balance: simulation time vs unstable solutions (tested at 15s: only 1/310 unstable)

### Baselines (all with B = 25,000 evaluations)

1. **Random Search**: Uniform sampling from command space $\mathcal{C}$
2. **Grid Search**: Fixed grid over $\mathcal{C}$ (5×5 for right-hand, 5×2 for both hands = 125 evals/task)
3. **Task-Wise MAP-Elites**: Independent MAP-Elites on each task (125 evals/task)

**Note**: Grid Search and Task-Wise MAP-Elites run sequentially per task (randomized order for fair comparison)

### Results (25 replications)

**Percentage of solved tasks** (≥ 1 solution):
- **MTMB-ME**: **67.8% ± 3.7%** ✅
- Grid Search: 57.9% ± 4.3%
- Random Search: 47.0% ± 2.8%
- Task-Wise MAP-Elites: 47.1% ± 2.6%

**Improvement**: +9.9% vs Grid, +20.8% vs Random/Task-Wise

**Solutions per solved task**:
- **MTMB-ME**: **10.2 ± 0.8** ✅✅
- Random Search: 4.9 ± 0.4
- Task-Wise MAP-Elites: 4.9 ± 0.4
- Grid Search: 3.4 ± 0.3

**Improvement**: **2× more solutions** than best baseline (Random/Task-Wise)

### Analysis

**Why MTMB-ME wins?**
1. **vs Random**: Exploits structure via crossover → samples from distribution near solutions
2. **vs Grid**: Adaptive sampling focuses on promising regions instead of uniform coverage
3. **vs Task-Wise MAP-Elites**: 125 evals/task insufficient for MAP-Elites to exploit (stuck in random search phase)

**Why Grid Search solves many tasks but few solutions?**
- Uniform coverage hits many task regions
- But wastes evaluations on low-probability regions
- No adaptation to discovered solutions

**Why Task-Wise MAP-Elites underperforms?**
- With 125 evals, barely gets past initialization
- Quick exploitation on few elites → early convergence
- Performs like random search

**MTMB-ME advantage**: Crossover from two elites (potentially from different tasks) indirectly samples from distribution centered on solution subspace → discards low-probability regions efficiently.

---

## Comparison to Related Methods

### vs MAP-Elites

| Aspect | MAP-Elites | MTMB-ME |
|--------|------------|---------|
| **Tasks** | 1 | n |
| **Selection** | Uniform from archive | Crossover between 2 elites from 2 tasks |
| **Evaluation** | On single task | On random task |
| **Archive** | 1 archive | n archives (1 per task) |
| **Use case** | Diversity for 1 problem | Diversity for many related problems |

### vs Multi-Task MAP-Elites

| Aspect | Multi-Task MAP-Elites | MTMB-ME |
|--------|----------------------|---------|
| **Solutions per task** | 1 (best) | Many (diverse) |
| **Archive** | 1 shared archive | n task-specific archives |
| **Goal** | Find best solution for each task | Find diverse solutions for each task |

### vs Multi-Task Optimization (CMA-ES, etc.)

| Aspect | Multi-Task Optimization | MTMB-ME |
|--------|-----------------------|---------|
| **Diversity** | No (finds optima) | Yes (illuminates behavior space) |
| **Output** | Best solutions | Archives of diverse solutions |
| **Suitable for** | Single best answer per task | Repertoires, robustness, exploration |

---

## When to Use MTMB-ME

**✅ Use MTMB-ME when**:
- Need **diverse** solutions for **multiple** tasks
- Tasks have similarity (share structure)
- Budget limited per task (can't run full MAP-Elites on each)
- Want repertoire of solutions (e.g., robot reflexes, grasping strategies)

**❌ Consider alternatives when**:
- Only need best solution per task → Multi-Task MAP-Elites
- Only have single task → MAP-Elites
- Tasks completely unrelated → Independent MAP-Elites
- Have large budget per task → Task-Wise MAP-Elites might work

---

## Future Directions (from paper)

1. **Dataset construction**: Use MTMB-ME to build dataset of diverse solutions with privileged knowledge (fault type)
2. **ML policy training**: Train ML model to select robust contact positions (1 or 2 hands) to prevent falls
3. **Alternative to RL**: Avoid RL exploration issues (e.g., PPO gets stuck using 1 hand due to deceptive self-collision fitness with 2 hands)

---

## Complete Implementation Details

### Archive Structure

```python
# STRUCTURE: One archive per task
# Each archive is a dictionary: behavior → (command, fitness)

# Example for 200 tasks:
archives = {}
for task_id in range(200):
    archives[task_id] = {}

# Adding an elite:
# archives[task_id][behavior_tuple] = (command_array, fitness_value)

# Example:
# Task 5, behavior (2, 3) = grid square (x=2, z=3)
# Command [0.5, 0.7, 0.3, 0.9], fitness 8.5
archives[5][(2, 3)] = (np.array([0.5, 0.7, 0.3, 0.9]), 8.5)
```

### Task Sampling Details

**Initialization** (random until n elites):
```python
def initialize_archives(tasks, command_space, budget_init):
    """
    Initialize by randomly sampling commands on random tasks.

    Stop when total number of elites >= n_tasks
    OR budget exhausted (whichever comes first).
    """
    archives = {i: {} for i in range(len(tasks))}
    evals = 0

    while count_total_elites(archives) < len(tasks):
        if evals >= budget_init:
            break

        # Random task
        task_id = np.random.randint(len(tasks))
        task = tasks[task_id]

        # Random command
        command = np.random.uniform(0, 1, command_space.dim)

        # Evaluate
        behavior = transition_function(task, command)
        fitness = fitness_function(task, command)
        evals += 1

        # Add to archive (always during init)
        if behavior not in archives[task_id]:
            archives[task_id][behavior] = (command, fitness)

    return archives, evals


def count_total_elites(archives):
    """Count total elites across all task archives."""
    return sum(len(archive) for archive in archives.values())
```

**Main loop** (uniform random task):
```python
# At each iteration:
task_k = np.random.randint(n_tasks)  # Uniform random in [0, n_tasks)

# No bias, no tournament, just pure uniform sampling
```

### Crossover Details (Genetic Algorithm Standard)

Likely standard genetic algorithm crossover:
- **Uniform crossover**: Each dimension from parent 1 or 2 with 50% probability
- **Arithmetic crossover**: $c = \alpha c_i + (1-\alpha) c_j$ for $\alpha \in [0,1]$
- **Mutation**: Gaussian noise added after crossover

### Computational Cost

**Evaluation budget**: $B = 25,000$
**Tasks**: $n = 200$
**Cost per task**: $\approx 125$ evaluations (vs 125 for baselines)

**Efficiency gain**: Crossover across tasks reduces wasted evaluations in unpromising regions.

---

## Worked Example

### Toy Problem: 2 Tasks, 3 Iterations

**Setup**:
- 2 tasks: T₀, T₁
- Command space: [0,1]² (2D)
- Behavior space: discrete grid (10×10)
- Budget: 3 iterations

**Initialization**:
```python
# Random init
archives = {0: {}, 1: {}}

# Task 0: random command [0.3, 0.7]
behavior = (3, 7)  # Grid square
fitness = 5.0
archives[0][(3, 7)] = ([0.3, 0.7], 5.0)

# Task 1: random command [0.8, 0.2]
behavior = (8, 2)
fitness = 7.0
archives[1][(8, 2)] = ([0.8, 0.2], 7.0)

# Now: 2 elites (1 per task) → stop init
```

**Iteration 1**:
```python
# 1. Select 2 tasks with elites
task_i = 0, task_j = 1

# 2. Get their elites
command_i = [0.3, 0.7]  # From T₀
command_j = [0.8, 0.2]  # From T₁

# 3. Crossover
# Dimension 0: pick from i → 0.3
# Dimension 1: pick from j → 0.2
offspring = [0.3, 0.2]

# 4. Mutation (suppose it adds [0.05, -0.03])
offspring = [0.35, 0.17]

# 5. Select random task
task_k = 1  # (random choice)

# 6. Evaluate
behavior = (3, 1)  # Grid for [0.35, 0.17] on task 1
fitness = 6.5

# 7. Update archive[1]
# behavior (3,1) is new → add it
archives[1][(3, 1)] = ([0.35, 0.17], 6.5)

# Now: 3 elites total (1 in T₀, 2 in T₁)
```

**Iteration 2**:
```python
# Select tasks: i=1, j=1 (can be same)
# Get elites: behavior (8,2) and (3,1)
# Crossover + mutate → [0.82, 0.18]
# Random task: k=0
# Evaluate on T₀: behavior (8,1), fitness 4.0
# Update: archives[0][(8,1)] = ([0.82, 0.18], 4.0)

# Now: 4 elites total (2 in T₀, 2 in T₁)
```

**Key insight**: Crossover between tasks shares successful patterns!

## Key Takeaways

1. **Multi-task diversity**: First method to find many diverse solutions for many tasks simultaneously
2. **Crossover advantage**: Leveraging similarity between tasks via crossover outperforms independent search
3. **Scalability**: Works with large task sets (200 tasks) and limited budget per task (125 evals)
4. **Practical impact**: 2× more solutions than baselines, critical for building robust repertoires
5. **Application**: Demonstrates value on real robotics problem (humanoid fault recovery)
6. **Implementation**: Simple structure (dict of dicts), standard GA operators, uniform task sampling

---

## Related

- [[MTMB-ME_detailed]] — Implementation details (to be created)
- [[MAP-Elites]] — Base single-task QD algorithm
- [[Multi-Task_MAP-Elites]] — Single solution per task variant
- [[GAME]] — Another multi-task variant (adversarial coevolution)

## References

- **Paper**: Anne & Mouret, "Multi-Task Multi-Behavior MAP-Elites", arXiv:2305.01264, 2023
- **Related**: MAP-Elites (Mouret & Clune, 2015), Multi-Task MAP-Elites (Pierrot et al., 2022)
- **Application**: D-Reflex (Chatzilygeroudis et al., 2018) - predecessor for single-hand fault recovery
- **Robot**: Talos humanoid robot in simulation
