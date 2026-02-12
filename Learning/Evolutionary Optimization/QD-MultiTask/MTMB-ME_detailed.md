# MTMB-MAP-Elites - Detailed Implementation

**Paper**: Anne & Mouret, 2023

---

## Complete Algorithm

```python
import numpy as np
import random

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_mtmb_me(n_tasks, task_space, command_space, behavior_space,
                       fitness_fn, budget_init):
    """
    Initialize archives by random sampling until each task has ≥1 elite.
    """
    archives = {task_id: {} for task_id in range(n_tasks)}
    tasks = sample_random_tasks(task_space, n_tasks)
    evals = 0

    while True:
        # Count elites
        total_elites = sum(len(archive) for archive in archives.values())
        if total_elites >= n_tasks or evals >= budget_init:
            break

        # Random task and command
        task_id = random.randint(0, n_tasks - 1)
        task = tasks[task_id]
        command = np.random.uniform(0, 1, command_space.dim)

        # Evaluate
        behavior = compute_behavior(task, command, behavior_space)
        fitness = fitness_fn(task, command)
        evals += 1

        # Add to archive (always during init)
        if behavior not in archives[task_id]:
            archives[task_id][behavior] = (command, fitness)

    return archives, tasks, evals


# ============================================================================
# MAIN LOOP
# ============================================================================

def run_mtmb_me(archives, tasks, n_tasks, fitness_fn, budget_total, 
                evals_so_far):
    """
    Main evolution loop with crossover between tasks.
    """
    evals = evals_so_far

    while evals < budget_total:
        # Find tasks with elites
        tasks_with_elites = [t for t in range(n_tasks) 
                            if len(archives[t]) > 0]

        if len(tasks_with_elites) == 0:
            break

        # ====================================================================
        # SELECTION: Pick 2 tasks and their random elites
        # ====================================================================

        task_i = random.choice(tasks_with_elites)
        task_j = random.choice(tasks_with_elites)

        behavior_i = random.choice(list(archives[task_i].keys()))
        behavior_j = random.choice(list(archives[task_j].keys()))

        command_i = archives[task_i][behavior_i][0]
        command_j = archives[task_j][behavior_j][0]

        # ====================================================================
        # VARIATION: Crossover + Mutation
        # ====================================================================

        command = crossover_and_mutate(command_i, command_j,
                                       mutation_prob=0.1,
                                       mutation_std=0.1)

        # ====================================================================
        # TASK SELECTION: Random task
        # ====================================================================

        task_k = random.randint(0, n_tasks - 1)
        task = tasks[task_k]

        # ====================================================================
        # EVALUATION
        # ====================================================================

        behavior = compute_behavior(task, command)
        fitness = fitness_fn(task, command)
        evals += 1

        # ====================================================================
        # ARCHIVE UPDATE
        # ====================================================================

        if behavior not in archives[task_k]:
            # New behavior → add
            archives[task_k][behavior] = (command, fitness)
        elif fitness > archives[task_k][behavior][1]:
            # Better fitness → replace
            archives[task_k][behavior] = (command, fitness)

    return archives, evals
```

---

## Detailed Operators

### 1. Crossover and Mutation

```python
def crossover_and_mutate(command_i, command_j,
                         mutation_prob=0.1, mutation_std=0.1):
    """
    Uniform crossover + Gaussian mutation.

    Args:
        command_i, command_j: Parent commands (numpy arrays)
        mutation_prob: Probability of mutating each dimension
        mutation_std: Standard deviation of Gaussian noise

    Returns:
        Offspring command (clipped to [0, 1])
    """
    offspring = []

    # UNIFORM CROSSOVER: Each dimension from parent 1 or 2 (50/50)
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
            offspring[i] = np.clip(offspring[i], 0, 1)

    return offspring
```

### 2. Behavior Computation

```python
def compute_behavior(task, command, grid_size=0.2):
    """
    Convert continuous behaviors to discrete grid cells.

    Args:
        task: Task instance (defines transition function)
        command: Command to execute
        grid_size: Size of grid cells (e.g., 0.2 = 20cm for robotics)

    Returns:
        behavior: Tuple of grid indices (hashable for dict keys)
    """
    # Execute command on task → get reached positions
    reached_positions = task.execute(command)

    behavior = []
    for (x, z) in reached_positions:
        grid_x = int(x / grid_size)
        grid_z = int(z / grid_size)
        behavior.append((grid_x, grid_z))

    return tuple(behavior)  # Hashable


def compute_behavior_1d(task, command, grid_size=0.2):
    """
    Simpler version: single continuous value → grid cell.
    """
    value = task.execute(command)  # Single float
    grid_idx = int(value / grid_size)
    return (grid_idx,)  # Return as 1-tuple
```

### 3. Task Sampling

```python
def sample_random_tasks(task_space, n_tasks):
    """
    Sample random tasks from task space.

    Example: Robot fault recovery with 100 situations × 2 hand modes
    """
    situations = []

    # Sample base situations
    for _ in range(n_tasks // 2):
        posture = sample_random_posture()
        wall_config = sample_random_wall_config()
        fault = sample_random_fault()
        situations.append((posture, wall_config, fault))

    # Create tasks: each situation with 2 hand modes
    tasks = []
    for situation in situations:
        tasks.append(Task(situation, hand_mode='right'))
        tasks.append(Task(situation, hand_mode='both'))

    return tasks


class Task:
    """Task definition with command and behavior spaces."""

    def __init__(self, parameters, hand_mode='right'):
        self.parameters = parameters
        self.hand_mode = hand_mode

    def execute(self, command):
        """
        Execute command on task.
        Returns: behavior (reached positions or continuous value)
        """
        # Simulator integration (pseudo-code)
        if self.hand_mode == 'right':
            reached = self._simulate_right_hand(command)
        else:
            reached = self._simulate_both_hands(command)

        return reached

    def fitness(self, command):
        """Return fitness (time before falling, etc.)"""
        # Pseudo-code
        return self._simulate_fitness(command)
```

---

## Archive Structure & Management

```python
def count_total_elites(archives):
    """Count total elites across all task archives."""
    return sum(len(archive) for archive in archives.values())


def get_archive_statistics(archives):
    """Compute archive statistics."""
    total_elites = count_total_elites(archives)
    solved_tasks = sum(1 for archive in archives.values() if len(archive) > 0)
    avg_solutions_per_task = (total_elites / solved_tasks 
                             if solved_tasks > 0 else 0)

    return {
        'total_elites': total_elites,
        'solved_tasks': solved_tasks,
        'avg_per_task': avg_solutions_per_task,
    }


def compute_qd_score(archives):
    """
    Compute total QD-Score across all tasks.
    QD-Score = sum of fitness of all elites.
    """
    score = 0.0
    for task_archive in archives.values():
        for (command, fitness) in task_archive.values():
            score += fitness
    return score
```

---

## Worked Example: 2 Tasks, 3 Iterations

### Setup

- 2 tasks: T₀, T₁
- Command space: [0,1]² (2D)
- Behavior space: discrete grid (10×10)
- Budget: 3 iterations

### Initialization

```python
archives = {0: {}, 1: {}}

# Task 0: random command [0.3, 0.7]
behavior_0 = (3, 7)  # Grid square
fitness_0 = 5.0
archives[0][(3, 7)] = ([0.3, 0.7], 5.0)

# Task 1: random command [0.8, 0.2]
behavior_1 = (8, 2)
fitness_1 = 7.0
archives[1][(8, 2)] = ([0.8, 0.2], 7.0)

# Now: 2 elites (1 per task) → stop init
```

### Iteration 1

```python
# Select 2 tasks with elites
task_i = 0, task_j = 1

# Get their random elites
command_i = [0.3, 0.7]  # From T₀
command_j = [0.8, 0.2]  # From T₁

# Crossover
# Dimension 0: random → pick from i → 0.3
# Dimension 1: random → pick from j → 0.2
offspring = [0.3, 0.2]

# Mutation (suppose Gaussian adds [0.05, -0.03])
offspring = [0.35, 0.17]

# Select random task
task_k = 1

# Evaluate
behavior = (3, 1)  # Grid cell for [0.35, 0.17] on task 1
fitness = 6.5

# Update archive[1]
# Behavior (3,1) is new → add
archives[1][(3, 1)] = ([0.35, 0.17], 6.5)

# Archives state:
# T₀: 1 elite
# T₁: 2 elites
# Total: 3 elites
```

### Iteration 2

```python
# Select tasks: i=1, j=1 (can be same task)
# Get elites: behavior (8,2) with fitness 7.0, and (3,1) with fitness 6.5
# Random pick: (8,2)
command_i = [0.8, 0.2]

# Random pick from same task
command_j = [0.35, 0.17]

# Crossover + mutate → [0.82, 0.18]
# Select random task: k=0
# Evaluate on T₀: behavior (8,1), fitness 4.0
# Compare to T₀: behavior (8,1) is new → add

archives[0][(8, 1)] = ([0.82, 0.18], 4.0)

# Archives state:
# T₀: 2 elites
# T₁: 2 elites
# Total: 4 elites
```

### Iteration 3

```python
# Select tasks: i=0, j=1
# Get elites: (3,7) from T₀, (8,2) from T₁
# Crossover + mutate → different command
# Select random task: k=0
# Evaluate and update

# Demonstrates: Crossover between tasks shares successful patterns!
```

---

## Hyperparameters & Design Choices

**Genetic Algorithm Operators**:
- `mutation_prob = 0.1`: 10% of dimensions mutate
- `mutation_std = 0.1`: Gaussian std for mutations
- Crossover: Uniform (50/50 per dimension)

**Task Sampling**:
- Uniform random task selection at each iteration
- No priority or bias (simple and effective)

**Behavior Discretization**:
- Grid-based (for robotics: ~20cm cells)
- Hashable behavior tuple for dict keys

**Archive Update**:
- Accept new behavior (always)
- Replace if better fitness for same behavior
- Discard if worse or equal fitness

---

## Key Implementation Notes

### 1. Task Representation

Tasks must provide:
- `execute(command)` → behavior (or reached positions)
- `fitness(command)` → float

### 2. Behavior Hashing

Behaviors must be hashable (tuples of ints) for dict lookup:
```python
behavior = (grid_x, grid_z)  # ✅ Hashable
archives[task_id][behavior] = (command, fitness)

behavior = [grid_x, grid_z]  # ❌ Not hashable
```

### 3. Efficiency

Cross-task crossover is most efficient when:
- Tasks have **moderate similarity** (crossover valuable)
- Evaluations **expensive** (need to minimize wasted evals)
- Archives already **initialized** (have good solutions to cross)

### 4. Scaling to Many Tasks

With n_tasks = 200:
- Total evaluations: 25,000
- Per-task budget: ~125 evaluations
- Independent MAP-Elites would barely explore at this budget
- Cross-task crossover makes it viable!

---

## Comparative Performance Analysis

### Why MTMB-ME Wins Over Task-Wise MAP-Elites

**Task-Wise MAP-Elites** (125 evals/task):
1. Initialization: ~25 evals to get 1 elite per cell
2. Remaining: ~100 evals for mutation/optimization
3. MAP-Elites convergence curve: Linear growth → plateau early
4. Result: Few solutions per task (4-5)

**MTMB-ME** (avg 125 evals/task):
1. Crossover from different tasks → start near solution subspace
2. Initialization: ~10 evals (faster to find first elite)
3. Crossover-based mutations: Higher success rate
4. Result: Many solutions per task (10+)

**Math**: Crossover indirectly samples from mixture of solution distributions → more efficient than random search.

---

## Extensions & Variations

### 1. Prioritized Task Selection

Instead of uniform random task:
```python
task_k = weighted_random_choice(tasks_with_few_solutions)
```
Focus on under-explored tasks.

### 2. Adaptive Crossover Probability

Increase crossover chance as archives grow:
```python
n_generations = (evals - evals_init) / batch_size
p_crossover = min(0.5, n_generations / 1000)
```

### 3. Task-Aware Mutation

Mutate more/less based on task-specific parameters:
```python
mutation_std = mutation_std_default * task.difficulty
```

---

## Related Algorithms

- **MAP-Elites**: Single task diversity
- **Multi-Task MAP-Elites**: Single solution per task (no behavior diversity)
- **MTMB-ME**: **This algorithm** - multiple solutions per task
- **PT-ME**: Continuous task space version (parametric-task)
- **GAME**: Adversarial multi-task variant

---

## References

- **Paper**: Anne & Mouret, "Multi-Task Multi-Behavior MAP-Elites", arXiv:2305.01264, 2023
- **Predecessor**: D-Reflex (Chatzilygeroudis et al., 2018) - single-hand fault recovery
- **Robot**: Talos humanoid in PyBullet simulation
- **Related**: MAP-Elites (Mouret & Clune, 2015), Multi-Task ME (Pierrot et al., 2022)
