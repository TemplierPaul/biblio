# JEDi: Detailed Implementation Guide

## Algorithm Overview

JEDi combines three main components:
1. **Weighted Gaussian Process**: Learn behavior→fitness mapping from repertoire
2. **Pareto Front Target Selection**: Choose promising + uncertain behaviors
3. **ES with WTFS**: Optimize toward targets using weighted scoring

---

## Weighted Gaussian Process Implementation

### Standard GP Formulation

**Prediction at query point x**:

```
Mean: μ(x) = k(x, X) · (K + σₙ² I)⁻¹ · y

Variance: σ²(x) = k(x, x) - k(x, X) · (K + σₙ² I)⁻¹ · k(X, x)
```

Where:
- `x`: Query behavior (BD vector)
- `X`: Archive behaviors (N × d matrix)
- `y`: Archive fitness values (N × 1 vector)
- `k(·,·)`: RBF kernel function
- `K`: Kernel covariance matrix (N × N)
- `σₙ²`: Observation noise variance
- `I`: Identity matrix

**RBF Kernel**:
```
k(x, x') = σf² · exp(-||x - x'||² / (2 · l²))
```

Where:
- `σf²`: Signal variance (output scale)
- `l`: Length scale (determines smoothness)

### Weighted GP Modification

**Problem**: Standard GP treats all observations equally. When archive has varying evaluation budgets per cell, this misrepresents confidence.

**Solution**: Weight matrix based on evaluation counts.

```
Mean: μ(x) = k(x, X) · (K + σₙ² W)⁻¹ · y

Variance: σ²(x) = k(x, x) - k(x, X) · (K + σₙ² W)⁻¹ · k(X, x)
```

Where **W** is diagonal weight matrix:
```
W_ii = 1 / n_i
```
`n_i` = number of evaluations in cell i

**Interpretation**:
- Many evaluations → Small weight → Low uncertainty
- Few evaluations → Large weight → High uncertainty
- Reflects true confidence in each archive region

### Implementation

```python
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

class WeightedGaussianProcess:
    """
    Weighted GP for behavior-fitness landscape modeling.
    """
    def __init__(self, kernel_lengthscale=1.0, kernel_variance=1.0,
                 noise_variance=1e-4):
        """
        Args:
            kernel_lengthscale: RBF length scale l
            kernel_variance: RBF signal variance σf²
            noise_variance: Observation noise σₙ²
        """
        self.l = kernel_lengthscale
        self.sigma_f2 = kernel_variance
        self.sigma_n2 = noise_variance

    def rbf_kernel(self, x1, x2):
        """
        Compute RBF kernel between points.

        Args:
            x1: (N1, d) array
            x2: (N2, d) array

        Returns:
            K: (N1, N2) kernel matrix
        """
        # Compute pairwise squared distances
        dists_sq = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
        # RBF kernel
        K = self.sigma_f2 * jnp.exp(-dists_sq / (2 * self.l ** 2))
        return K

    def fit(self, X_train, y_train, weights):
        """
        Fit GP to training data.

        Args:
            X_train: (N, d) archive behaviors
            y_train: (N,) archive fitness values
            weights: (N,) evaluation count weights (1/n_i)

        Returns:
            self with cached parameters
        """
        self.X_train = X_train
        self.y_train = y_train

        # Compute kernel matrix
        K = self.rbf_kernel(X_train, X_train)

        # Add weighted noise: K + σₙ² W
        W_diag = weights
        K_y = K + self.sigma_n2 * jnp.diag(W_diag)

        # Cholesky decomposition for efficiency
        self.L = cho_factor(K_y, lower=True)

        # Precompute K⁻¹ y
        self.alpha = cho_solve(self.L, y_train)

        return self

    def predict(self, X_test):
        """
        Predict mean and variance at test points.

        Args:
            X_test: (M, d) test behaviors

        Returns:
            mean: (M,) predicted fitness
            variance: (M,) prediction variance
        """
        # Compute kernel between test and train
        K_s = self.rbf_kernel(X_test, self.X_train)  # (M, N)

        # Compute mean: K_s · α
        mean = K_s @ self.alpha

        # Compute variance
        # First solve: (K + σₙ² W)⁻¹ K_s^T
        v = cho_solve(self.L, K_s.T)  # (N, M)

        # Variance: k(x*, x*) - K_s · v
        K_ss = self.rbf_kernel(X_test, X_test)
        variance = jnp.diag(K_ss) - jnp.sum(K_s * v.T, axis=1)

        # Ensure non-negative variance (numerical stability)
        variance = jnp.maximum(variance, 1e-10)

        return mean, variance
```

### Hyperparameter Optimization

```python
def optimize_gp_hyperparameters(X, y, weights):
    """
    Optimize GP hyperparameters via maximum likelihood.

    Args:
        X, y, weights: Training data

    Returns:
        best_params: (lengthscale, signal_variance, noise_variance)
    """
    from scipy.optimize import minimize

    def neg_log_likelihood(params):
        l, sigma_f2, sigma_n2 = params

        # Compute kernel
        dists_sq = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        K = sigma_f2 * jnp.exp(-dists_sq / (2 * l ** 2))
        K_y = K + sigma_n2 * jnp.diag(weights)

        # Log-likelihood
        try:
            L = jnp.linalg.cholesky(K_y)
            alpha = cho_solve((L, True), y)
            log_likelihood = (
                -0.5 * y @ alpha
                - jnp.sum(jnp.log(jnp.diag(L)))
                - 0.5 * len(y) * jnp.log(2 * jnp.pi)
            )
            return -log_likelihood
        except:
            return 1e10  # Penalize invalid parameters

    # Initial guess
    x0 = [1.0, 1.0, 1e-4]

    # Optimize
    result = minimize(
        neg_log_likelihood,
        x0,
        bounds=[(0.1, 10.0), (0.1, 10.0), (1e-6, 1e-2)],
        method='L-BFGS-B'
    )

    return result.x
```

---

## Target Selection via Pareto Front

### Multi-Objective Formulation

**Objectives**:
1. **Maximize predicted fitness**: μ(b)
2. **Maximize uncertainty**: σ(b)

**Pareto Front**: Set of behaviors where improving one objective requires worsening the other.

### Implementation

```python
def compute_pareto_front(means, variances):
    """
    Compute Pareto front of (mean, variance) points.

    Args:
        means: (N,) predicted fitness values
        variances: (N,) prediction variances

    Returns:
        pareto_indices: Indices of Pareto-optimal points
    """
    N = len(means)
    is_pareto = jnp.ones(N, dtype=bool)

    for i in range(N):
        # Check if any other point dominates point i
        # Point j dominates i if: mean[j] >= mean[i] AND var[j] >= var[i]
        # (with at least one strict inequality)
        dominated = jnp.any(
            (means > means[i]) & (variances >= variances[i]) |
            (means >= means[i]) & (variances > variances[i])
        )
        is_pareto = is_pareto.at[i].set(~dominated)

    pareto_indices = jnp.where(is_pareto)[0]
    return pareto_indices


def select_targets(gp, archive_centroids, n_targets):
    """
    Select target behaviors from Pareto front.

    Args:
        gp: Fitted WeightedGaussianProcess
        archive_centroids: (M, d) all possible behaviors
        n_targets: Number of targets to select

    Returns:
        target_behaviors: (n_targets, d) selected targets
    """
    # Predict at all centroids
    means, variances = gp.predict(archive_centroids)

    # Compute Pareto front
    pareto_indices = compute_pareto_front(means, variances)

    # Randomly sample n_targets from Pareto front
    if len(pareto_indices) >= n_targets:
        selected_indices = jax.random.choice(
            jax.random.PRNGKey(0),
            pareto_indices,
            shape=(n_targets,),
            replace=False
        )
    else:
        # If Pareto front smaller than n_targets, sample all + random others
        remaining = n_targets - len(pareto_indices)
        other_indices = jnp.setdiff1d(
            jnp.arange(len(means)),
            pareto_indices
        )
        additional = jax.random.choice(
            jax.random.PRNGKey(0),
            other_indices,
            shape=(remaining,),
            replace=False
        )
        selected_indices = jnp.concatenate([pareto_indices, additional])

    target_behaviors = archive_centroids[selected_indices]
    return target_behaviors
```

---

## Weighted Target-Fitness Score (WTFS)

### Formulation

**Constrained Optimization Problem**:
```
max_φ f(φ)  subject to  ||d(φ) - d_target|| ≤ ε
```

**Lagrangian Relaxation (WTFS)**:
```
S_WTFS^α(φ) = α · S_target(φ) + (1 - α) · S_fitness(φ)
```

**Target Score** (distance to target):
```
S_target(φ) = 1 - (||d(φ) - d_target|| - d_min) / (d_max - d_min)
```

**Fitness Score** (normalized fitness):
```
S_fitness(φ) = (f(φ) - f_min) / (f_max - f_min)
```

Where:
- `d(φ)`: Behavior descriptor of policy φ
- `f(φ)`: Fitness of policy φ
- `d_min, d_max`: Min/max observed distances to target
- `f_min, f_max`: Min/max observed fitness values

### Implementation

```python
class WTFS_Scorer:
    """
    Weighted Target-Fitness Score for ES selection.
    """
    def __init__(self, target_behavior, alpha=0.5):
        """
        Args:
            target_behavior: (d,) target BD vector
            alpha: Balance parameter [0, 1]
        """
        self.target = target_behavior
        self.alpha = alpha

        # Running statistics
        self.f_min = jnp.inf
        self.f_max = -jnp.inf
        self.d_min = jnp.inf
        self.d_max = -jnp.inf

    def compute_score(self, behaviors, fitnesses):
        """
        Compute WTFS scores for a batch of policies.

        Args:
            behaviors: (B, d) behavior descriptors
            fitnesses: (B,) fitness values

        Returns:
            scores: (B,) WTFS scores
        """
        # Update running statistics
        self.f_min = jnp.minimum(self.f_min, jnp.min(fitnesses))
        self.f_max = jnp.maximum(self.f_max, jnp.max(fitnesses))

        # Compute distances to target
        distances = jnp.linalg.norm(behaviors - self.target, axis=1)
        self.d_min = jnp.minimum(self.d_min, jnp.min(distances))
        self.d_max = jnp.maximum(self.d_max, jnp.max(distances))

        # Normalize target score
        if self.d_max > self.d_min:
            S_target = 1 - (distances - self.d_min) / (self.d_max - self.d_min)
        else:
            S_target = jnp.ones_like(distances)

        # Normalize fitness score
        if self.f_max > self.f_min:
            S_fitness = (fitnesses - self.f_min) / (self.f_max - self.f_min)
        else:
            S_fitness = jnp.ones_like(fitnesses)

        # Weighted combination
        scores = self.alpha * S_target + (1 - self.alpha) * S_fitness

        return scores
```

### α Annealing Schedule

```python
def alpha_schedule(iteration, total_iterations, alpha_init=1.0, alpha_final=0.0):
    """
    Linear annealing schedule for α parameter.

    Args:
        iteration: Current JEDi loop
        total_iterations: Total JEDi loops
        alpha_init: Starting α (exploration)
        alpha_final: Ending α (exploitation)

    Returns:
        alpha: Current α value
    """
    progress = iteration / total_iterations
    alpha = alpha_init + (alpha_final - alpha_init) * progress
    return jnp.clip(alpha, alpha_final, alpha_init)
```

**Alternative: Exponential Decay**
```python
def alpha_schedule_exp(iteration, alpha_init=1.0, decay_rate=0.95):
    """Exponential decay schedule."""
    return alpha_init * (decay_rate ** iteration)
```

---

## ES Integration

### Sep-CMA-ES for Small Genomes

```python
from evosax import Sep_CMA_ES

def run_es_emitter(
    es,
    target_behavior,
    initial_policy,
    environment,
    num_generations,
    alpha
):
    """
    Run single ES emitter toward target behavior.

    Args:
        es: EvoSax ES instance
        target_behavior: Target BD
        initial_policy: Starting policy parameters
        environment: Evaluation environment
        num_generations: ES generations to run
        alpha: WTFS balance parameter

    Returns:
        repertoire_updates: All evaluated (behavior, fitness, genotype)
    """
    # Initialize ES state
    es_params = es.default_params
    es_state = es.initialize(initial_policy)

    # Initialize WTFS scorer
    scorer = WTFS_Scorer(target_behavior, alpha)

    repertoire_updates = []

    for gen in range(num_generations):
        # Sample population
        genotypes, es_state = es.ask(es_state, es_params)

        # Evaluate in environment
        behaviors = []
        fitnesses = []
        for genotype in genotypes:
            behavior, fitness = environment.evaluate(genotype)
            behaviors.append(behavior)
            fitnesses.append(fitness)
            repertoire_updates.append((behavior, fitness, genotype))

        behaviors = jnp.array(behaviors)
        fitnesses = jnp.array(fitnesses)

        # Compute WTFS scores
        scores = scorer.compute_score(behaviors, fitnesses)

        # Update ES distribution
        es_state = es.tell(es_state, genotypes, scores, es_params)

    return repertoire_updates
```

### LM-MA-ES for Large Genomes

```python
from evosax import LM_MA_ES

# Similar structure, but LM-MA-ES uses limited-memory matrix adaptation
# More memory-efficient for large neural network policies (thousands of parameters)

es = LM_MA_ES(
    popsize=256,
    num_dims=policy_size,
    elite_ratio=0.5
)
```

---

## Complete JEDi Algorithm

```python
def jedi(
    environment,
    num_loops,
    num_es_emitters,
    num_es_generations,
    es_population_size,
    repertoire_size,
    behavior_dim,
    alpha_init=1.0,
    alpha_final=0.0
):
    """
    Complete JEDi algorithm.

    Args:
        environment: Evaluation environment
        num_loops: Number of JEDi loops
        num_es_emitters: Parallel ES instances (typically 4)
        num_es_generations: ES generations per target (100-1000)
        es_population_size: ES population size (16-256)
        repertoire_size: MAP-Elites archive size (1024)
        behavior_dim: Dimensionality of behavior space
        alpha_init/alpha_final: WTFS annealing schedule

    Returns:
        final_repertoire: Filled MAP-Elites archive
    """
    # Initialize repertoire
    repertoire = initialize_repertoire(repertoire_size, behavior_dim)

    # Fill with random policies
    for _ in range(repertoire_size):
        policy = jax.random.normal(jax.random.PRNGKey(0), (policy_size,))
        behavior, fitness = environment.evaluate(policy)
        repertoire.add(behavior, fitness, policy)

    # Main JEDi loops
    for loop in range(num_loops):
        # Step 1: Train Weighted GP
        archive_behaviors = repertoire.get_behaviors()
        archive_fitnesses = repertoire.get_fitnesses()
        archive_counts = repertoire.get_evaluation_counts()

        weights = 1.0 / archive_counts

        gp = WeightedGaussianProcess()
        gp.fit(archive_behaviors, archive_fitnesses, weights)

        # Step 2: Select targets from Pareto front
        all_centroids = repertoire.get_all_centroids()
        targets = select_targets(gp, all_centroids, num_es_emitters)

        # Step 3: Run ES emitters
        alpha = alpha_schedule(loop, num_loops, alpha_init, alpha_final)

        for target in targets:
            # Find nearest policy in repertoire
            nearest_policy = repertoire.get_nearest(target)

            # Initialize ES
            es = Sep_CMA_ES(
                popsize=es_population_size,
                num_dims=policy_size
            )

            # Run ES toward target
            updates = run_es_emitter(
                es,
                target,
                nearest_policy,
                environment,
                num_es_generations,
                alpha
            )

            # Add all sampled genotypes to repertoire
            for behavior, fitness, genotype in updates:
                repertoire.add(behavior, fitness, genotype)

    return repertoire
```

---

## Hyperparameters

### Algorithm Configuration

| Parameter | Maze Tasks | Brax Tasks | Description |
|-----------|------------|------------|-------------|
| num_loops | 10-20 | 5-10 | JEDi loops |
| num_es_emitters | 4 | 4 | Parallel ES instances |
| num_es_generations | 100 | 500-1000 | ES gens per target |
| es_population_size | 16 | 256 | ES samples per gen |
| repertoire_size | 1024 | 1024 | Archive cells |
| alpha_init | 1.0 | 1.0 | Initial α |
| alpha_final | Task-specific | Task-specific | Final α |

### Task-Specific α Values

| Task | α_final | Rationale |
|------|---------|-----------|
| Maze A | 0.3 | Moderate exploration |
| Maze B | 0.3 | Moderate exploration |
| Maze C | 0.5 | Balanced |
| Maze Quad B | 0.7 | High exploration (hard) |
| HalfCheetah | 0.5 | Balanced |
| Walker2D | 0.3 | More exploitation |
| AntMaze | 0.0 | Pure exploitation |

**Tuning Guidance**:
- **Hard exploration** (deceptive fitness): Higher α_final (0.5-0.7)
- **Smooth fitness landscapes**: Lower α_final (0.0-0.3)
- **Unknown task**: Start with α_final = 0.3, adjust based on performance

### ES Hyperparameters

**Sep-CMA-ES** (small genomes):
```python
es_params = {
    'sigma_init': 0.05,
    'elite_ratio': 0.5,
    'mean_decay': 0.0,
    'n_devices': 1
}
```

**LM-MA-ES** (large genomes):
```python
es_params = {
    'sigma_init': 0.05,
    'elite_ratio': 0.5,
    'lrate_mean': 1.0,
    'lrate_cov': 0.6
}
```

### GP Hyperparameters

**Initial Values** (optimized during training):
```python
gp_params = {
    'lengthscale': 1.0,
    'signal_variance': 1.0,
    'noise_variance': 1e-4
}
```

### Network Architectures

**Maze Tasks**:
```python
network = {
    'layers': [8],  # Single hidden layer
    'activation': 'relu',
    'output_activation': 'tanh'
}
```

**Brax Tasks**:
```python
network = {
    'layers': [256, 256],  # Two hidden layers
    'activation': 'tanh',
    'output_activation': 'tanh'
}
```

---

## Computational Requirements

### Parallelization

**Total Batch Size**:
```
batch_size = num_es_emitters × es_population_size
```

**Example** (Brax):
- 4 emitters × 256 population = 1024 evaluations/generation
- Fully parallelizable across CPU/GPU cores

### Runtime Estimates

**Maze Tasks**:
- 10 JEDi loops × 100 ES generations × 64 evaluations = 64,000 total evals
- ~1-2 hours on 32 CPU cores

**Brax Tasks**:
- 10 JEDi loops × 500 ES generations × 1024 evaluations = 5.12M total evals
- ~4-8 hours on 64 CPU cores or 1 GPU

---

## Implementation Tips

### 1. Warm-Start Repertoire

```python
def warm_start_repertoire(repertoire, environment, num_samples=1000):
    """
    Initialize repertoire with diverse random policies.
    """
    for _ in range(num_samples):
        policy = jax.random.normal(jax.random.PRNGKey(0), (policy_size,))
        behavior, fitness = environment.evaluate(policy)
        repertoire.add(behavior, fitness, policy)
    return repertoire
```

**Benefit**: Provides initial data for GP training.

### 2. GP Retraining Frequency

**Options**:
1. **Every JEDi loop** (recommended): Fresh GP each loop
2. **Every N loops**: Faster but less adaptive
3. **Incremental updates**: Update GP with new data only

**Trade-off**: Retraining cost vs. prediction accuracy

### 3. Evaluation Count Tracking

```python
class Repertoire:
    def __init__(self, num_cells):
        self.fitness = jnp.full(num_cells, -jnp.inf)
        self.behaviors = jnp.zeros((num_cells, behavior_dim))
        self.genotypes = []
        self.eval_counts = jnp.zeros(num_cells)  # Track evaluations

    def add(self, behavior, fitness, genotype):
        cell_idx = self.behavior_to_cell(behavior)
        self.eval_counts = self.eval_counts.at[cell_idx].add(1)

        if fitness > self.fitness[cell_idx]:
            self.fitness = self.fitness.at[cell_idx].set(fitness)
            self.behaviors = self.behaviors.at[cell_idx].set(behavior)
            self.genotypes[cell_idx] = genotype
```

### 4. Batch GP Predictions

```python
def batch_predict(gp, X_test, batch_size=1000):
    """
    Predict in batches to avoid memory issues.
    """
    num_batches = len(X_test) // batch_size + 1
    means = []
    variances = []

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X_test))
        batch = X_test[start:end]

        mean, var = gp.predict(batch)
        means.append(mean)
        variances.append(var)

    return jnp.concatenate(means), jnp.concatenate(variances)
```

---

## Common Pitfalls

### 1. α Too Low

**Problem**: Pure exploitation, ES converges prematurely without exploration.

**Solution**: Ensure α_final ≥ 0.3 for hard exploration tasks.

### 2. GP Overfitting

**Problem**: GP memorizes noise in archive.

**Solution**: Tune noise_variance parameter (1e-6 to 1e-3), or use cross-validation.

### 3. Pareto Front Empty

**Problem**: All points dominated by single point (no diversity).

**Solution**: Check GP predictions; may need to increase noise or adjust kernel lengthscale.

### 4. ES Stagnation

**Problem**: ES stuck in local optimum despite behavior targets.

**Solution**: Increase α (more focus on reaching targets), or restart ES with higher σ_init.

---

## Extensions

### 1. Task-Adaptive α

```python
def adaptive_alpha(repertoire_coverage, fitness_stagnation):
    """
    Adjust α based on progress metrics.
    """
    if fitness_stagnation:  # Not improving
        alpha = 0.7  # Explore more
    elif repertoire_coverage < 0.5:
        alpha = 0.5  # Balanced
    else:
        alpha = 0.2  # Exploit more
    return alpha
```

### 2. Multi-Fidelity GP

```python
# Use cheap low-fidelity evaluations to guide expensive high-fidelity
gp_lowfid = WeightedGP()
gp_lowfid.fit(X_cheap, y_cheap, weights_cheap)

# Transfer predictions to high-fidelity GP
gp_highfid = MultiFidelityGP(gp_lowfid)
```

### 3. Hierarchical Targets

```python
# First select coarse-grained regions, then fine-grained targets within
regions = select_targets(gp, coarse_centroids, num_regions)
for region in regions:
    fine_targets = select_targets_in_region(gp, region, num_targets_per_region)
```

---

## Summary

JEDi's implementation combines:
1. **Weighted GP** for behavior-fitness landscape learning (accounts for sparse exploration)
2. **Pareto front selection** for diverse, promising targets
3. **WTFS with α annealing** for smooth exploration-exploitation transitions
4. **Parallel ES emitters** for efficient optimization

**Key Takeaway**: The weighted GP and α annealing are critical. Standard GP + fixed α significantly underperforms.

**Practical Recommendation**: Start with α_final = 0.3, adjust based on task. Use 4 ES emitters for parallelization. Retrain GP every JEDi loop for best results.

---

**See [[JEDi]] for high-level overview and motivation.**
