# Extract-QD Framework - Implementation Details

## Authors & Paper

**Authors**: Manon Flageat, Johann Huber, François Helenon, Stephane Doncieux, Antoine Cully
**Paper**: "Extract-QD: A Generic Approach for Quality-Diversity in Noisy, Stochastic or Uncertain Domains"
**Venue**: GECCO 2025
**Institutions**: AIRL (Imperial College London), ISIR (Sorbonne Université)

---

## Extract-ME Algorithm (EME)

### Full Algorithm

```python
def extract_me(
    container: ContainerWithDepth,  # Grid with depth d
    eval_budget_per_gen: int,       # Total evaluations per generation (e.g., 1024)
    extraction_budget: float = 0.25, # 25% of budget for re-evaluation
    depth: int = 8,                  # Depth per cell
    generations: int,
    initial_samples: int = 1         # Samples on first evaluation
):
    """
    Extract-ME: Adaptive-sampling UQD via extraction mechanism

    Key innovation: Fixed proportion of evaluations (25%) dedicated
    to re-evaluating elites extracted from archive
    """

    # Initialization
    container.initialize(depth=depth)

    # Compute budgets
    n_offspring = int(eval_budget_per_gen * (1 - extraction_budget))  # 768
    n_extracts = eval_budget_per_gen - n_offspring                    # 256

    for gen in range(generations):
        # 1. SELECTION: Copy parents for offspring
        parents = container.select_uniform(n_offspring)

        # 2. VARIATION: Mutate to create offspring
        offspring = mutate(parents)

        # 3. EXTRACTION: Extract elites for re-evaluation
        extracted = container.extract(
            n_extracts,
            method='exponential_rank'  # Prob ∝ exp(-rank)
        )

        # 4. EVALUATION: Concatenate and evaluate
        solutions = concatenate(offspring, extracted)
        evals = evaluate_all(solutions, n_samples=initial_samples)

        # Each solution stores evaluation buffer
        for sol, eval_result in zip(solutions, evals):
            sol.buffer.append(eval_result)  # Accumulate samples

        # 5. ESTIMATION: Compute mean fitness and descriptor
        for sol in solutions:
            sol.fitness_est = mean(sol.buffer['fitness'])
            sol.desc_est = mean(sol.buffer['descriptor'])

        # 6. ADDITION: Add solutions to container
        container.add(solutions)  # Uses depth-ordering internally

    # Return top layer only
    return container.get_top_layer()
```

### Detailed Components

#### 1. Extraction Mechanism

**Exponential Rank-Based Probability**:

```python
def extract(container, n_extracts, method='exponential_rank'):
    """
    Extract elites from archive for re-evaluation

    Exponential probability: higher rank → higher extraction chance
    Enables large depth while maintaining pressure on top elites
    """
    extracted = []

    # For each extract
    for _ in range(n_extracts):
        # Select random cell
        cell = container.random_cell()

        if len(cell.depth) == 0:
            continue

        # Compute extraction probabilities
        d = len(cell.depth)
        ranks = np.arange(d)  # 0 (top) to d-1 (bottom)

        # Exponential decay: P(rank) ∝ exp(-α * rank)
        # α chosen s.t. top elite has ~50% of total prob
        alpha = np.log(d) / (d - 1) if d > 1 else 1.0
        probs = np.exp(-alpha * ranks)
        probs /= probs.sum()

        # Sample elite from depth
        idx = np.random.choice(d, p=probs)
        elite = cell.depth.pop(idx)  # Remove from depth

        extracted.append(elite)

        # Shift up remaining elites to fill gap
        cell.shift_up()

    return extracted
```

**Why exponential vs. linear?**
- **Exponential**: Higher pressure on top elites, works with larger depth
- **Linear**: More uniform sampling, better for small depth
- **Uniform**: No pressure, exploration-focused

**Relation to other methods**:
- **AS (Archive-Sampling)**: Extracts 100% of archive (all cells, all depths)
- **Adapt-ME**: Extracts only challenged elites (variable budget)
- **EME**: Extracts 25% budget, exponentially weighted

#### 2. Evaluation Buffer Management

```python
class Solution:
    """
    Solution with accumulated evaluation buffer
    """
    def __init__(self, genotype):
        self.genotype = genotype
        self.buffer = {
            'fitness': [],      # List of fitness samples
            'descriptor': []    # List of descriptor samples
        }
        self.n_evals = 0

    def add_evaluation(self, fitness, descriptor):
        """Append new evaluation to buffer"""
        self.buffer['fitness'].append(fitness)
        self.buffer['descriptor'].append(descriptor)
        self.n_evals += 1

    def estimate_performance(self):
        """Estimate from accumulated buffer"""
        return {
            'fitness': np.mean(self.buffer['fitness']),
            'descriptor': np.mean(self.buffer['descriptor']),
            'samples': self.n_evals
        }
```

**Drift handling**: Re-evaluated elites may belong to different cell
- Extract elite from cell A
- Re-evaluate → descriptor now maps to cell B
- Add to cell B (competes on fitness)
- Cell A has empty slot → filled by shifting depth upward

#### 3. Container with Depth

```python
class ContainerWithDepth:
    """
    Grid container maintaining depth d of elites per cell
    """
    def __init__(self, grid_shape, depth):
        self.grid = [[[] for _ in range(grid_shape[1])]
                     for _ in range(grid_shape[0])]
        self.depth = depth
        self.grid_shape = grid_shape

    def add(self, solution):
        """Add solution to appropriate cell's depth"""
        cell_idx = self.descriptor_to_cell(solution.desc_est)
        cell = self.grid[cell_idx[0]][cell_idx[1]]

        if len(cell) < self.depth:
            # Cell not full, add directly
            cell.append(solution)
        else:
            # Cell full, compete on fitness
            worst_idx = np.argmin([s.fitness_est for s in cell])
            if solution.fitness_est > cell[worst_idx].fitness_est:
                cell[worst_idx] = solution

        # Re-order depth by fitness (descending)
        cell.sort(key=lambda s: s.fitness_est, reverse=True)

    def get_top_layer(self):
        """Return only top elite from each cell"""
        archive = []
        for row in self.grid:
            for cell in row:
                if len(cell) > 0:
                    archive.append(cell[0])  # Top elite only
        return archive
```

**Depth benefits**:
- **d=1**: Standard MAP-Elites (no memory buffer)
- **d=8**: EME default (reasonable memory, drift protection)
- **d=32**: Deep-Grid (more memory, higher cost)

#### 4. Depth-Ordering Operator

```python
def depth_ordering_fitness(cell):
    """
    Standard depth-ordering: fitness-based ranking

    Used in: ME, ME-Sampling, AS, Adapt-ME, EME, Deep-Grid
    """
    cell.sort(key=lambda s: s.fitness_est, reverse=True)
    return cell

def depth_ordering_weighted(cell, w_fitness=0.5, w_reprod=0.5):
    """
    Weighted depth-ordering: fitness + reproducibility

    Used in: ME-Weighted, AS-Weighted

    Reproducibility = 1 - spread of descriptor distribution
    """
    for s in cell:
        # Estimate reproducibility from descriptor variance
        desc_std = np.std(s.buffer['descriptor'], axis=0)
        s.reproducibility = 1.0 / (1.0 + np.linalg.norm(desc_std))

        # Weighted score
        s.score = w_fitness * s.fitness_est + w_reprod * s.reproducibility

    cell.sort(key=lambda s: s.score, reverse=True)
    return cell

def depth_ordering_delta(cell, delta_f, delta_r):
    """
    Delta comparison: implement user preferences

    Used in: ME-Delta, AS-Delta

    A replaces B if:
    - A.fitness > B.fitness + delta_f, OR
    - A.fitness > B.fitness - delta_f AND A.reprod > B.reprod + delta_r, OR
    - A.reprod > B.reprod + delta_r AND A.fitness > B.fitness - delta_f
    """
    # Complex replacement rules implementing trade-off preferences
    # See paper Section 2.2 for full details
    pass

def depth_ordering_pareto(cell):
    """
    Pareto-based ordering: multi-objective (fitness, reproducibility)

    Used in: MOME-Reprod, MOME-X

    Maintains Pareto fronts within each cell
    """
    # Non-dominated sorting on (fitness, reproducibility)
    fronts = non_dominated_sort(cell, objectives=['fitness', 'reproducibility'])

    # Flatten fronts (front 0 first, then front 1, etc.)
    ordered = []
    for front in fronts:
        ordered.extend(front)
    return ordered
```

---

## Framework Modules

### Complete Module Specification

#### Module 1: Selection Operator

**Purpose**: Choose parents to create offspring

```python
# Uniform selection (ME, EME, AS, Adapt-ME)
def select_uniform(container, n):
    return [random.choice(container.all_elites()) for _ in range(n)]

# Fitness-proportional (Deep-Grid)
def select_fitness_proportional(container, n):
    """Select within cell depth, weighted by fitness"""
    parents = []
    for _ in range(n):
        cell = random.choice(container.filled_cells())
        # Within cell, select proportional to fitness
        fitnesses = [s.fitness_est for s in cell.depth]
        probs = fitnesses / np.sum(fitnesses)
        parents.append(np.random.choice(cell.depth, p=probs))
    return parents
```

#### Module 2: Variation Operator

**Purpose**: Mutate parents into offspring

```python
# Gaussian mutation (ME, EME, AS)
def gaussian_mutation(parent, sigma=0.1):
    offspring = parent.copy()
    offspring.genotype += np.random.normal(0, sigma, size=offspring.genotype.shape)
    return offspring

# Polynomial mutation (common in evolutionary computation)
def polynomial_mutation(parent, eta=20):
    # Simulated Binary Crossover-style mutation
    pass

# Gradient-based (PGA-ME, Extract-PGA)
def td3_variation(parent, replay_buffer, actor, critic):
    """
    Policy gradient variation from PGA-ME
    Uses Twin Delayed DDPG (TD3) for RL tasks
    """
    # Train actor-critic on replay buffer
    # Return updated policy
    pass
```

#### Module 3: Extraction Operator

**Purpose**: Choose which and how many elites to re-evaluate

```python
# No extraction (ME, ME-Sampling, Deep-Grid)
def extract_none(container):
    return []

# Full archive extraction (AS, AS-Weighted, AS-Delta)
def extract_all(container):
    """Extract entire archive content (top layer + depths)"""
    extracted = []
    for cell in container.all_cells():
        extracted.extend(cell.depth)
        cell.depth = []  # Empty cell
    return extracted

# Adaptive extraction (Adapt-ME)
def extract_challenged(container, offspring):
    """Extract elites challenged by offspring"""
    extracted = []
    for off in offspring:
        cell = container.get_cell(off.desc_est)
        if len(cell.depth) > 0:
            elite = cell.depth[0]  # Top elite
            if needs_reevaluation(elite, off):
                extracted.append(cell.depth.pop(0))
    return extracted

# EME exponential extraction
def extract_exponential(container, n_extracts):
    """Extract n_extracts elites with exponential rank probability"""
    # See earlier code block
    pass
```

**Budget comparison**:
- **ME-Sampling**: 0 extracts, 32 samples × n_offspring
- **AS**: (C × d) extracts, 1 sample × (n_offspring + C×d)
- **Adapt-ME**: ≤ (b + bd) extracts, variable samples
- **EME**: (0.25 × budget) extracts, 1 sample × budget

#### Module 4: Container with Depth

Already covered above. Key: depth=1 is standard QD, depth>1 is UQD.

#### Module 5: Depth-Ordering Operator

Already covered above (4 variants).

#### Module 6: Samples

**Purpose**: Number of samples on first evaluation

- **ME**: 1 sample (deterministic assumption)
- **ME-Sampling**: 32 samples (fixed high sampling)
- **EME**: 1 sample (adaptive via re-evaluation)
- **ARIA**: 1024-2048 samples (extreme fixed sampling)

---

## UQD Approaches in the Framework

### Comprehensive Mapping Table

| Method | Selection | Variation | Extraction | Container | Depth | Depth-Ordering | Samples |
|--------|-----------|-----------|------------|-----------|-------|----------------|---------|
| **ME** | Uniform | Gaussian | None (0) | Grid | 1 | Fitness | 1 |
| **ME-Sampling** | Uniform | Gaussian | None (0) | Grid | 1 | Fitness | 32 |
| **ME-Reprod** | Uniform | Gaussian | None (0) | Grid | 1 | Reproducibility | 32 |
| **ME-Weighted** | Uniform | Gaussian | None (0) | Grid | 1 | Weighted (f, r) | 32 |
| **ME-Low-Spread** | Uniform | Gaussian | None (0) | Grid | 1 | Strict (f ↑, r ↑) | 32 |
| **ME-Delta** | Uniform | Gaussian | None (0) | Grid | 1 | Delta (δ_f, δ_r) | 32 |
| **MOME-X** | Crowding | Gaussian | None (0) | MOME | Pareto | Pareto (f, r) | 32 |
| **Deep-Grid** | Fitness-prop | Gaussian | None (0) | Grid | 32 | Seniority+Fitness | 1 |
| **AS** | Uniform | Gaussian | All (C×d) | Grid | 8 | Fitness | 1 |
| **AS-Weighted** | Uniform | Gaussian | All (C×d) | Grid | 8 | Weighted (f, r) | 1 |
| **AS-Delta** | Uniform | Gaussian | All (C×d) | Grid | 8 | Delta (δ_f, δ_r) | 1 |
| **Adapt-ME** | Uniform | Gaussian | Challenged (≤b+bd) | Grid | 8 | Fitness | 1 |
| **EME** | Uniform | Gaussian | Exponential (0.25×B) | Grid | 8 | Fitness | 1 |
| **EPGA** | Uniform | TD3 | Exponential (0.25×B) | Grid | 8 | Fitness | 1 |
| **ARIA** | Edge-only | NES | None (0) | Grid | 1 | Fitness+Reprod | 1024-2048 |

**Legend**:
- **C**: Number of cells in grid
- **d**: Depth per cell
- **b**: Offspring per generation
- **B**: Total evaluation budget per generation

### Key Relationships

**Fixed-sampling family** (high initial samples, no extraction):
- ME-Sampling, ME-Reprod, ME-Weighted, ME-Low-Spread, ME-Delta, MOME-X

**Adaptive-sampling family** (low initial samples, strategic extraction):
- AS, AS-Weighted, AS-Delta, Adapt-ME, EME

**Difference**:
- **Fixed**: Sample everything heavily upfront, no adaptation
- **Adaptive**: Sample lightly, re-evaluate promising solutions

---

## Hyperparameters

### EME Default Configuration

```python
EME_CONFIG = {
    # Extraction
    'extraction_budget': 0.25,        # 25% of budget for re-evaluation
    'extraction_method': 'exponential_rank',

    # Container
    'depth': 8,                       # Elites per cell
    'grid_shape': (16, 16),           # 256 cells (task-dependent)

    # Evaluation
    'initial_samples': 1,              # Samples on first eval
    'eval_budget_per_gen': 1024,       # Total evals per generation

    # Variation
    'mutation_sigma': 0.1,             # Gaussian std for mutation

    # Selection
    'selection': 'uniform',

    # Depth-ordering
    'ordering': 'fitness',
}
```

### Task-Specific Settings

**Arm (Fitness/Desc Noise)**:
- Grid: 16×16 (256 cells)
- Descriptor: End-effector position (2D)
- Uncertainty: Gaussian noise on fitness OR descriptor

**Hexapod**:
- Grid: 32×32 (1024 cells)
- Descriptor: Final position (2D)
- Uncertainty: Random initial state

**Walker**:
- Grid: 16×16 (256 cells)
- Descriptor: Feet contact pattern (2D)
- Neural network: (17 obs → [8] → 6 actions)
- Uncertainty: Random initial joints position/velocity

**Ant**:
- Grid: 32×32 (1024 cells)
- Descriptor: Final position (2D)
- Neural network: (29 obs → [8] → 8 actions)
- Uncertainty: Random initial joints position/velocity

---

## Extract-PGA (EPGA)

### Algorithm

```python
def extract_pga(
    container: ContainerWithDepth,
    eval_budget_per_gen: int = 128,
    extraction_budget: float = 0.25,
    depth: int = 8,
    generations: int,
):
    """
    Extract-PGA: EME with TD3-based variation

    Swap variation operator: Gaussian → TD3 policy gradient
    All other components identical to EME
    """

    # Initialize
    container.initialize(depth=depth)
    replay_buffer = ReplayBuffer()
    actor = ActorNetwork()
    critic = CriticNetwork()

    n_offspring = int(eval_budget_per_gen * (1 - extraction_budget))  # 96
    n_extracts = eval_budget_per_gen - n_offspring                    # 32

    for gen in range(generations):
        # Selection (same as EME)
        parents = container.select_uniform(n_offspring)

        # Variation: TD3 instead of Gaussian
        offspring = []
        for parent in parents:
            # Train policy via gradient descent
            child = td3_improve(parent, replay_buffer, actor, critic)
            offspring.append(child)

        # Extraction (same as EME)
        extracted = container.extract(n_extracts, method='exponential_rank')

        # Evaluation (same as EME)
        solutions = concatenate(offspring, extracted)
        evals, transitions = evaluate_all_rl(solutions)  # Also collect transitions

        # Store transitions in replay buffer
        replay_buffer.add(transitions)

        # Estimation (same as EME)
        for sol, eval_result in zip(solutions, evals):
            sol.buffer.append(eval_result)
            sol.fitness_est = mean(sol.buffer['fitness'])
            sol.desc_est = mean(sol.buffer['descriptor'])

        # Addition (same as EME)
        container.add(solutions)

    return container.get_top_layer()
```

### Results

**Tasks**: QD-Gym (HalfCheetah, Ant, Humanoid)

**Comparison** (Corrected QD-Score):
- **PGA**: Baseline (no uncertainty handling)
- **ME**: Worst (deterministic QD)
- **EPGA**: **Best** (p < 5×10⁻³)

**Key insight**: Re-evaluating 25% of elites improves QD-RL at **no additional cost**
- Same total evaluations
- Same evaluations per generation
- Better estimation → better archive

---

## Performance Estimation Techniques

### Problem 1: Lucky Solutions

**Issue**: Single evaluation favors outliers
- Solution A: mean fitness = 0.8, std = 0.1
- Solution B: mean fitness = 0.6, std = 0.3
- Single eval: B gets 0.9 (lucky), A gets 0.8 → B replaces A
- Truth: A is better on average

**EME solution**: Re-evaluate B multiple times
- After 5 evals: B average = 0.65 → A replaces B back
- Extraction ensures promising solutions get re-evaluated

### Problem 2: Drift

**Issue**: Re-evaluated elite belongs to different cell
- Elite in cell (5, 3): descriptor = [0.35, 0.25]
- Re-evaluate: descriptor = [0.42, 0.28] → maps to cell (6, 4)
- Cell (5, 3) now empty!

**EME solution**: Depth provides backup
- Cell (5, 3) has depth=8 elites
- Extract top elite → shifts up elite #2
- No empty cell, no information loss

### Problem 3: Exploration vs. Exploitation

**Issue**: Too much re-evaluation → no exploration
- AS: Re-evaluates 100% of archive → no offspring when archive large
- Adapt-ME: Re-evaluates challenged elites → sequential, slow

**EME solution**: Fixed 25% budget
- 75% offspring (exploration)
- 25% extraction (exploitation)
- Parallelizable, independent of archive size

---

## Metrics

### Illusory vs. Corrected Archive

**Illusory Archive**:
- Archive returned by algorithm
- Uses estimated fitness/descriptor (may be biased)
- Not trustworthy in UQD

**Corrected Archive**:
- Re-evaluate all illusory solutions with 512 samples
- Use median fitness/descriptor (ground truth approximation)
- Build new archive from scratch with ground truth
- Metrics computed on corrected archive

### Corrected QD-Score

```python
def corrected_qd_score(illusory_archive, n_ground_truth_samples=512):
    """
    Compute QD-Score on corrected archive

    Steps:
    1. Re-evaluate all illusory solutions heavily (512 samples)
    2. Compute ground truth (median of samples)
    3. Build corrected archive from scratch using ground truth
    4. Sum fitness in corrected archive
    """
    # Re-evaluate
    ground_truth = {}
    for sol in illusory_archive:
        fitness_samples = [evaluate(sol) for _ in range(n_ground_truth_samples)]
        desc_samples = [evaluate_descriptor(sol) for _ in range(n_ground_truth_samples)]

        ground_truth[sol] = {
            'fitness': np.median(fitness_samples),
            'descriptor': np.median(desc_samples, axis=0)
        }

    # Build corrected archive
    corrected_container = Container(depth=1)  # No depth for final archive
    for sol, gt in ground_truth.items():
        sol_copy = sol.copy()
        sol_copy.fitness_est = gt['fitness']
        sol_copy.desc_est = gt['descriptor']
        corrected_container.add(sol_copy)

    # Compute QD-Score
    qd_score = sum(s.fitness_est for s in corrected_container.all_elites())
    return qd_score
```

### Average Samples

```python
def average_samples(illusory_archive):
    """
    Average number of samples per solution in top layer

    Indicates sampling efficiency
    """
    top_layer = [s for cell in archive.cells for s in cell.depth[:1]]
    return np.mean([s.n_evals for s in top_layer])
```

**Interpretation**:
- **Low (1-5)**: Mostly new offspring, little re-evaluation
- **Medium (5-20)**: Balanced exploration/exploitation
- **High (>20)**: Heavy re-evaluation, good estimation

**EME vs. AS**:
- **EME**: Higher average samples on critical solutions (exponential weighting)
- **AS**: Uniform samples across all solutions (less efficient)

---

## Implementation Tips

### JAX/QDax Integration

```python
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.standard_emitters import MixingEmitter
import jax
import jax.numpy as jnp

def extract_me_qdax(
    init_genotypes: jnp.ndarray,
    env_batch_size: int,
    depth: int = 8,
):
    """
    EME implementation in QDax

    QDax provides:
    - Parallel evaluation via vmap
    - GPU acceleration
    - Repertoire (container) management
    """

    # Create repertoire with depth
    repertoire = MapElitesRepertoire.init_with_depth(
        genotypes=init_genotypes,
        depth=depth,
    )

    # Custom emitter with extraction
    emitter = ExtractEmitter(
        extraction_ratio=0.25,
        extraction_method='exponential',
    )

    # Training loop
    for gen in range(num_generations):
        # Generate offspring (75% of budget)
        offspring, _ = emitter.emit(repertoire, ...)

        # Extract elites (25% of budget)
        extracted = emitter.extract(repertoire, ...)

        # Concatenate
        genotypes = jnp.concatenate([offspring, extracted], axis=0)

        # Evaluate (parallelized via vmap)
        fitnesses, descriptors, _, _ = scoring_fn(genotypes)

        # Update buffers
        repertoire = repertoire.update_buffers(genotypes, fitnesses, descriptors)

        # Add to repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses)

    return repertoire.get_top_layer()
```

### Extraction Parallelization

**Key advantage over Adapt-ME**: Full parallelization

```python
# EME: Fully parallel
offspring = mutate_batch(parents)  # Parallel
extracted = extract_batch(archive)  # Parallel, known size
solutions = concatenate(offspring, extracted)
evals = evaluate_batch(solutions)  # Fully parallel, fixed size

# Adapt-ME: Sequential dependencies
offspring = mutate_batch(parents)  # Parallel
for off in offspring:
    eval_off = evaluate(off)  # Sequential
    if challenges_elite(off, elite):
        eval_elite = evaluate(elite)  # Sequential
        # Can't know # evals ahead of time
```

### Memory Management

**Evaluation buffers**:
- Each solution stores all its evaluations
- Memory grows with re-evaluations
- For depth=8, archive size=1024: ~8k solutions tracked
- At 100 samples/solution: 800k evaluations stored

**Optimization**:
```python
# Option 1: Limit buffer size
MAX_BUFFER_SIZE = 100
if len(sol.buffer['fitness']) > MAX_BUFFER_SIZE:
    sol.buffer['fitness'] = sol.buffer['fitness'][-MAX_BUFFER_SIZE:]

# Option 2: Online mean/variance
class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0  # Sum of squared deviations

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def variance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else 0
```

---

## Experimental Results

### Main Study: UQD Task Suite

**Tasks**: Arm (Fitness/Desc Noise), Hexapod, Walker, Ant

**Results** (Corrected QD-Score, mean ± std):

| Task | EME | AS | Adapt-ME | Deep-Grid | ME-Sampling | ME |
|------|-----|----|----|----|----|---|
| **Arm Fit Noise** | **2.1±0.1** | 2.0±0.1 | 1.8±0.2 | **2.1±0.1** | **2.1±0.1** | 1.5±0.2 |
| **Arm Desc Noise** | **1.9±0.1** | 1.7±0.2 | 1.6±0.3 | 1.2±0.3 | **1.9±0.1** | 1.0±0.2 |
| **Hexapod** | **150±10** | N/A | 130±15 | **148±12** | **149±11** | 90±20 |
| **Walker** | **2400±200** | 2200±300 | 1800±400 | 2100±300 | 2300±250 | 1200±500 |
| **Ant** | **1800±150** | N/A | 1200±300 | 1000±400 | 1500±200 | 800±300 |

**Key findings**:
1. **EME best or tied-best on all tasks** (only method with this property)
2. **AS undefined** on Hexapod/Ant (archive too large for full re-evaluation)
3. **Adapt-ME struggles** on high-D tasks (too much re-evaluation)
4. **Deep-Grid struggles** with descriptor noise (drift issues)

### QDRL Study: Extract-PGA

**Tasks**: HalfCheetah, Ant, Humanoid (QD-Gym)

**Results** (Corrected QD-Score):

| Task | EPGA | PGA | ME |
|------|------|-----|-----|
| **HalfCheetah** | **4200±300** | 3800±400 | 2500±600 |
| **Ant** | **3500±250** | 3000±350 | 1800±500 |
| **Humanoid** | **2800±400** | 2300±500 | 1200±700 |

**Significance**: p < 5×10⁻³ (EPGA > PGA on all tasks)

**Key insight**: Accounting for uncertainty improves QD-RL at **no extra cost**

---

## Comparison to Related Work

### EME vs. AS

| Aspect | EME | AS |
|--------|-----|-----|
| **Extraction** | 25% budget | 100% archive |
| **Scalability** | Archive-size independent | Requires small archives |
| **Sampling efficiency** | Exponential weighting | Uniform sampling |
| **Parallelization** | Full | Full |
| **When undefined** | Never | Large archives |

### EME vs. Adapt-ME

| Aspect | EME | Adapt-ME |
|--------|-----|----------|
| **Extraction** | Random (25% budget) | Challenged elites |
| **Parallelization** | Full | Partial (sequential evals) |
| **Budget** | Fixed per generation | Variable |
| **Speed** | Fast | Slow (10-100× slower) |
| **Coverage** | Entire archive | Only challenged regions |

### EME vs. Deep-Grid

| Aspect | EME | Deep-Grid |
|--------|-----|-----------|
| **Extraction** | Explicit re-evaluation | Implicit (offspring near parents) |
| **Selection** | Uniform | Fitness-proportional |
| **Descriptor noise** | Handles well | Struggles (drift) |
| **Fitness noise** | Handles well | Handles well |

### EME vs. ME-Sampling

| Aspect | EME | ME-Sampling |
|--------|-----|-------------|
| **Initial samples** | 1 | 32 |
| **Re-evaluation** | Adaptive (25%) | None |
| **Exploration** | Better | Hindered |
| **Total samples** | Lower | Higher |
| **Performance** | Better on complex tasks | Better on simple tasks |

---

## Key Takeaways

1. **25% extraction budget**: Sweet spot for exploration/exploitation trade-off
2. **Exponential rank weighting**: Enables large depth (d=8) while maintaining pressure
3. **Depth = 8**: Sufficient for drift protection without excessive memory
4. **Framework modularity**: Easy to swap components (e.g., Gaussian → TD3)
5. **Archive-size independence**: Scales to large archives (Hexapod, Ant)
6. **Parallelizable**: Unlike Adapt-ME, fully parallel evaluation
7. **First-guess method**: Performs well across diverse tasks
8. **No extra cost**: Improves existing QD algorithms (PGA → EPGA) at same budget

---

## Related

- [[Extract_QD]] — High-level overview
- [[MAP-Elites]] — Base QD algorithm
- [[POET]] / [[Enhanced_POET]] — Open-ended learning under uncertainty
- [[GAME]] — Adversarial QD (could benefit from Extract-QD for noisy evaluations)

## References

- **Paper**: Flageat et al., "Extract-QD: A Generic Approach for Quality-Diversity in Noisy, Stochastic or Uncertain Domains", GECCO 2025
- **Code**: (to be released upon acceptance)
- **QDax**: QD library in JAX (Chalumeau et al., 2024)
- **Related UQD**: Adaptive-Sampling (Rakicevic et al.), Archive-Sampling (Flageat et al. 2023), Deep-Grid (Flageat et al. 2020)
