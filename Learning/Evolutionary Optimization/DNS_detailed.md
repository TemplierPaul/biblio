# DNS: Detailed Implementation Guide

## Algorithm Overview

DNS (Dominated Novelty Search) is a Quality-Diversity algorithm implemented as a Genetic Algorithm with specialized competition function. The core mechanism: solutions compete based on behavioral distance to better-performing solutions.

---

## Core Competition Function

### Mathematical Formulation

For each solution $i$ in population:

**Step 1: Identify Dominating Solutions**
```math
\mathcal{D}_i = \{j \in \{1, \ldots, N\} \mid f_j > f_i\}
```
Set of all solutions with strictly greater fitness.

**Step 2: Compute Distances in Descriptor Space**
```math
d_{ij} = \|\mathbf{d}_i - \mathbf{d}_j\|_2 \quad \forall j \in \mathcal{D}_i
```
Euclidean distance between descriptors.

**Step 3: Calculate Dominated Novelty Score**
```math
\tilde{f}_i = \begin{cases}
+\infty & \text{if } |\mathcal{D}_i| = 0 \\
\frac{1}{k} \sum_{j \in \mathcal{K}_i} d_{ij} & \text{otherwise}
\end{cases}
```

Where $\mathcal{K}_i$ contains indices of $k$ solutions in $\mathcal{D}_i$ with smallest distances to solution $i$.

**Interpretation**:
- $\tilde{f}_i = +\infty$: No fitter solutions exist → local fitness maximum → always preserved
- Large $\tilde{f}_i$: Behaviorally distant from better solutions → preserved (stepping stone)
- Small $\tilde{f}_i$: Behaviorally similar to better solutions → eliminated (redundant)

---

## Implementation

### Python Implementation

```python
import numpy as np
from typing import List, Tuple

class DNS:
    """
    Dominated Novelty Search algorithm.
    """
    def __init__(self, population_size: int, k: int = 5):
        """
        Args:
            population_size: Number of solutions to maintain
            k: Number of nearest fitter neighbors for competition
        """
        self.N = population_size
        self.k = k

    def compute_competition_fitness(self,
                                   fitness: np.ndarray,
                                   descriptors: np.ndarray) -> np.ndarray:
        """
        Compute dominated novelty score for all solutions.

        Args:
            fitness: (N,) array of fitness values
            descriptors: (N, d) array of behavior descriptors

        Returns:
            competition_fitness: (N,) array of competition fitness values
        """
        N = len(fitness)
        competition_fitness = np.zeros(N)

        for i in range(N):
            # Step 1: Identify fitter solutions
            fitter_mask = fitness > fitness[i]
            fitter_indices = np.where(fitter_mask)[0]

            # Step 2: Handle case with no fitter solutions
            if len(fitter_indices) == 0:
                competition_fitness[i] = np.inf
                continue

            # Step 3: Compute distances to fitter solutions
            distances = np.linalg.norm(
                descriptors[fitter_indices] - descriptors[i],
                axis=1
            )

            # Step 4: Find k-nearest fitter solutions
            k_actual = min(self.k, len(distances))
            k_nearest_distances = np.partition(distances, k_actual - 1)[:k_actual]

            # Step 5: Average distance to k-nearest
            competition_fitness[i] = np.mean(k_nearest_distances)

        return competition_fitness
```

### Vectorized Implementation (Faster)

```python
def compute_competition_fitness_vectorized(self,
                                          fitness: np.ndarray,
                                          descriptors: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of dominated novelty scores.

    More efficient for large populations.
    """
    N = len(fitness)
    competition_fitness = np.zeros(N)

    # Compute all pairwise distances
    # Shape: (N, N)
    distances = np.linalg.norm(
        descriptors[:, np.newaxis, :] - descriptors[np.newaxis, :, :],
        axis=2
    )

    # Mask for fitter solutions: (N, N) boolean array
    # fitter_mask[i, j] = True if j is fitter than i
    fitter_mask = fitness[np.newaxis, :] > fitness[:, np.newaxis]

    for i in range(N):
        # Get distances to fitter solutions
        fitter_distances = distances[i, fitter_mask[i]]

        if len(fitter_distances) == 0:
            competition_fitness[i] = np.inf
            continue

        # K-nearest fitter distances
        k_actual = min(self.k, len(fitter_distances))
        k_nearest = np.partition(fitter_distances, k_actual - 1)[:k_actual]

        competition_fitness[i] = np.mean(k_nearest)

    return competition_fitness
```

---

## Complete Algorithm

### Main Loop

```python
def run_dns(self,
           initial_population: np.ndarray,
           evaluate_fn,
           reproduction_fn,
           num_generations: int,
           batch_size: int) -> List[Tuple[np.ndarray, float, np.ndarray]]:
    """
    Run DNS for specified number of generations.

    Args:
        initial_population: (N, genome_dim) initial genomes
        evaluate_fn: Function mapping genome to (fitness, descriptor)
        reproduction_fn: Function generating offspring from population
        num_generations: Number of generations to run
        batch_size: Number of offspring per generation

    Returns:
        archive: List of (genome, fitness, descriptor) tuples
    """
    # Initialize
    population = initial_population.copy()

    for gen in range(num_generations):
        # Step 1: Reproduction - generate offspring
        offspring = reproduction_fn(population, batch_size)

        # Step 2: Combine population and offspring
        combined = np.concatenate([population, offspring], axis=0)

        # Step 3: Evaluation
        fitness = []
        descriptors = []
        for genome in combined:
            f, d = evaluate_fn(genome)
            fitness.append(f)
            descriptors.append(d)

        fitness = np.array(fitness)
        descriptors = np.array(descriptors)

        # Step 4: Competition - compute dominated novelty scores
        competition_fitness = self.compute_competition_fitness(
            fitness, descriptors
        )

        # Step 5: Selection - keep top-N by competition fitness
        # Handle infinity values (local maxima)
        inf_mask = np.isinf(competition_fitness)
        num_inf = np.sum(inf_mask)

        if num_inf >= self.N:
            # More local maxima than population size
            # Among infinite values, select by raw fitness
            inf_indices = np.where(inf_mask)[0]
            selected_inf = inf_indices[
                np.argsort(fitness[inf_indices])[-self.N:]
            ]
            selected_indices = selected_inf
        elif num_inf > 0:
            # Some local maxima + some finite values
            inf_indices = np.where(inf_mask)[0]
            finite_indices = np.where(~inf_mask)[0]

            # Keep all infinite values
            # Fill remaining with best finite values
            num_finite_needed = self.N - num_inf
            selected_finite = finite_indices[
                np.argsort(competition_fitness[finite_indices])[-num_finite_needed:]
            ]
            selected_indices = np.concatenate([inf_indices, selected_finite])
        else:
            # No local maxima, select top-N by competition fitness
            selected_indices = np.argsort(competition_fitness)[-self.N:]

        population = combined[selected_indices]

    # Return final population as archive
    archive = []
    for genome in population:
        f, d = evaluate_fn(genome)
        archive.append((genome, f, d))

    return archive
```

---

## Reproduction Operators

### Standard Mutation

```python
def iso_dd_mutation(population: np.ndarray,
                   batch_size: int,
                   sigma: float = 0.01) -> np.ndarray:
    """
    Isotropic Gaussian mutation with sigma decay.

    Args:
        population: (N, genome_dim) parent genomes
        batch_size: Number of offspring to generate
        sigma: Mutation strength

    Returns:
        offspring: (batch_size, genome_dim) mutated genomes
    """
    # Select parents (median-based selection)
    # Evaluate all parents, select above-median for reproduction
    selected_parents = population[
        np.random.choice(len(population), batch_size)
    ]

    # Apply Gaussian mutation
    noise = np.random.normal(0, sigma, size=selected_parents.shape)
    offspring = selected_parents + noise

    return offspring
```

### Crossover (Optional)

```python
def crossover(population: np.ndarray,
             batch_size: int) -> np.ndarray:
    """
    Uniform crossover between random parent pairs.

    Args:
        population: (N, genome_dim) parent genomes
        batch_size: Number of offspring to generate

    Returns:
        offspring: (batch_size, genome_dim) crossed genomes
    """
    offspring = []

    for _ in range(batch_size):
        # Select two random parents
        parent1, parent2 = population[
            np.random.choice(len(population), 2, replace=False)
        ]

        # Uniform crossover
        mask = np.random.rand(len(parent1)) < 0.5
        child = np.where(mask, parent1, parent2)

        offspring.append(child)

    return np.array(offspring)
```

---

## Distance Metrics

### Euclidean Distance (Default)

```python
def euclidean_distance(d1: np.ndarray, d2: np.ndarray) -> float:
    """
    L2 distance in descriptor space.

    Args:
        d1, d2: Behavior descriptors

    Returns:
        distance: L2 norm
    """
    return np.linalg.norm(d1 - d2)
```

### Alternative Metrics

```python
def manhattan_distance(d1: np.ndarray, d2: np.ndarray) -> float:
    """L1 distance (Manhattan)."""
    return np.sum(np.abs(d1 - d2))


def cosine_distance(d1: np.ndarray, d2: np.ndarray) -> float:
    """Cosine distance (1 - cosine similarity)."""
    similarity = np.dot(d1, d2) / (
        np.linalg.norm(d1) * np.linalg.norm(d2)
    )
    return 1 - similarity


def mahalanobis_distance(d1: np.ndarray, d2: np.ndarray,
                        cov_inv: np.ndarray) -> float:
    """Mahalanobis distance (accounts for covariance)."""
    diff = d1 - d2
    return np.sqrt(diff @ cov_inv @ diff)
```

**Usage Note**: Euclidean distance is standard and works well. Alternative metrics can be explored for specific domains where descriptor space has non-uniform importance or correlations.

---

## Hyperparameters

### Core Parameters

| Parameter | Default | Range | Sensitivity | Notes |
|-----------|---------|-------|-------------|-------|
| k | 5 | [1, 10] | Low | Number of nearest fitter neighbors |
| Population size N | Task-dependent | [100, 10000] | Medium | Standard GA parameter |
| Batch size B | Task-dependent | [10, 1000] | Medium | Offspring per generation |
| Mutation σ | 0.01 | [0.001, 0.1] | High | Exploration-exploitation balance |

### Parameter Sensitivity Analysis

**k (Number of Nearest Neighbors)**:

Tested values: k ∈ {1, 2, 5, 10}

| k | Performance Impact |
|---|-------------------|
| 1 | Slightly worse (too local) |
| 2 | Competitive |
| 5 | **Recommended (best overall)** |
| 10 | Competitive (slightly slower) |

**Recommendation**: Set k = 5 and do not tune further. Algorithm is robust to this choice.

---

## Integration with Existing Frameworks

### QDax Integration

```python
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire

class DNS_QDax:
    """
    DNS integrated with QDax framework.
    """
    def __init__(self, config):
        self.config = config
        self.dns = DNS(
            population_size=config.population_size,
            k=config.k
        )

    def update(self, repertoire, genotypes, fitnesses, descriptors):
        """
        QDax-compatible update function.

        Args:
            repertoire: Current QDax repertoire (can be ignored for DNS)
            genotypes: Batch of genotypes
            fitnesses: Batch of fitness values
            descriptors: Batch of behavior descriptors

        Returns:
            new_repertoire: Updated repertoire
        """
        # Combine with existing population
        all_genotypes = jnp.concatenate([
            repertoire.genotypes, genotypes
        ], axis=0)
        all_fitnesses = jnp.concatenate([
            repertoire.fitnesses, fitnesses
        ], axis=0)
        all_descriptors = jnp.concatenate([
            repertoire.descriptors, descriptors
        ], axis=0)

        # Compute competition fitness
        competition_fitness = self.dns.compute_competition_fitness(
            all_fitnesses, all_descriptors
        )

        # Select top-N
        selected_indices = jnp.argsort(competition_fitness)[
            -self.config.population_size:
        ]

        # Create new repertoire
        new_repertoire = MapElitesRepertoire(
            genotypes=all_genotypes[selected_indices],
            fitnesses=all_fitnesses[selected_indices],
            descriptors=all_descriptors[selected_indices],
            centroids=None  # DNS doesn't use centroids
        )

        return new_repertoire
```

### Pyribs Integration

```python
from ribs.archives import ArchiveBase

class DNS_Archive(ArchiveBase):
    """
    Pyribs-compatible DNS archive.
    """
    def __init__(self, solution_dim, k=5, population_size=1000):
        self.solution_dim = solution_dim
        self.k = k
        self.max_size = population_size

        # Storage
        self._genotypes = []
        self._fitnesses = []
        self._descriptors = []

    def add(self, solution, objective, measures):
        """
        Add solution to archive (Pyribs interface).

        DNS doesn't add immediately; competition happens in batch.
        """
        self._genotypes.append(solution)
        self._fitnesses.append(objective)
        self._descriptors.append(measures)

        # Trigger competition if buffer full
        if len(self._genotypes) > self.max_size * 1.5:
            self._compete()

    def _compete(self):
        """Run DNS competition and select top solutions."""
        fitness = np.array(self._fitnesses)
        descriptors = np.array(self._descriptors)

        dns = DNS(population_size=self.max_size, k=self.k)
        competition_fitness = dns.compute_competition_fitness(
            fitness, descriptors
        )

        # Select top solutions
        selected = np.argsort(competition_fitness)[-self.max_size:]

        self._genotypes = [self._genotypes[i] for i in selected]
        self._fitnesses = [self._fitnesses[i] for i in selected]
        self._descriptors = [self._descriptors[i] for i in selected]
```

---

## Computational Complexity

### Time Complexity

**Per Generation**:
- Pairwise distance computation: $O(N^2 \cdot d)$
  - N = population size
  - d = descriptor dimensionality
- K-nearest search per solution: $O(N \log k)$
- Total: $O(N^2 \cdot d + N^2 \log k) \approx O(N^2 \cdot d)$

**Comparison to MAP-Elites**:
- MAP-Elites: $O(N \cdot d)$ (grid cell lookup)
- DNS: $O(N^2 \cdot d)$ (pairwise distances)

**Trade-off**: DNS is quadratic in population size but:
- Independent of descriptor dimensionality structure
- No grid overhead for high-D spaces
- Vectorizable for GPU acceleration

### Space Complexity

**Memory Requirements**:
- Population storage: $O(N \cdot (g + d))$
  - g = genome size
  - d = descriptor size
- Distance matrix (if precomputed): $O(N^2)$

**Comparison to MAP-Elites**:
- MAP-Elites: $O(c^d \cdot g)$ (c cells per dimension)
- DNS: $O(N \cdot g)$

For high-D, DNS is more memory-efficient (linear vs. exponential in d).

---

## Optimizations

### 1. Incremental Distance Updates

```python
class DNS_Incremental:
    """
    DNS with incremental distance updates.
    """
    def __init__(self, population_size, k=5):
        self.N = population_size
        self.k = k
        self.distance_cache = {}

    def compute_distance_cached(self, d1, d2, i, j):
        """Compute or retrieve cached distance."""
        key = (min(i, j), max(i, j))
        if key not in self.distance_cache:
            self.distance_cache[key] = np.linalg.norm(d1 - d2)
        return self.distance_cache[key]

    def update_cache(self, new_descriptors):
        """Update cache with new solutions."""
        # Only compute distances involving new solutions
        N_old = len(self.distance_cache)
        for i in range(N_old, len(new_descriptors)):
            for j in range(i):
                self.compute_distance_cached(
                    new_descriptors[i],
                    new_descriptors[j],
                    i, j
                )
```

### 2. GPU Acceleration (JAX)

```python
import jax
import jax.numpy as jnp

@jax.jit
def compute_competition_fitness_jax(fitness, descriptors, k):
    """
    JAX-accelerated competition fitness computation.
    """
    N = fitness.shape[0]

    # Pairwise distances (vectorized)
    diff = descriptors[:, jnp.newaxis, :] - descriptors[jnp.newaxis, :, :]
    distances = jnp.linalg.norm(diff, axis=2)

    # Fitter mask
    fitter_mask = fitness[jnp.newaxis, :] > fitness[:, jnp.newaxis]

    # For each solution, compute average distance to k-nearest fitter
    def compute_single(i):
        fitter_dists = jnp.where(
            fitter_mask[i],
            distances[i],
            jnp.inf  # Mask non-fitter solutions
        )

        # Check if any fitter solutions exist
        has_fitter = jnp.any(jnp.isfinite(fitter_dists))

        # K-nearest (use partition for efficiency)
        k_nearest = jnp.partition(fitter_dists, k)[:k]
        mean_dist = jnp.mean(k_nearest)

        return jnp.where(has_fitter, mean_dist, jnp.inf)

    competition_fitness = jax.vmap(compute_single)(jnp.arange(N))
    return competition_fitness
```

### 3. Approximate Nearest Neighbors

For very large populations (N > 10,000):

```python
from sklearn.neighbors import NearestNeighbors

def compute_competition_fitness_approximate(self, fitness, descriptors):
    """
    Approximate k-nearest fitter neighbors using KD-tree.
    """
    N = len(fitness)
    competition_fitness = np.zeros(N)

    for i in range(N):
        # Get fitter solutions
        fitter_mask = fitness > fitness[i]
        fitter_indices = np.where(fitter_mask)[0]

        if len(fitter_indices) == 0:
            competition_fitness[i] = np.inf
            continue

        # Build KD-tree for fitter solutions
        fitter_descriptors = descriptors[fitter_indices]
        nn = NearestNeighbors(n_neighbors=min(self.k, len(fitter_indices)))
        nn.fit(fitter_descriptors)

        # Query k-nearest
        distances, _ = nn.kneighbors([descriptors[i]])
        competition_fitness[i] = np.mean(distances[0])

    return competition_fitness
```

**Trade-off**: Faster for large N but approximate (may miss exact k-nearest).

---

## Debugging and Visualization

### Check Competition Fitness Distribution

```python
def debug_competition_fitness(fitness, descriptors, k=5):
    """
    Visualize competition fitness distribution.
    """
    dns = DNS(population_size=len(fitness), k=k)
    comp_fitness = dns.compute_competition_fitness(fitness, descriptors)

    # Check for infinite values (local maxima)
    num_inf = np.sum(np.isinf(comp_fitness))
    print(f"Local maxima (infinite competition fitness): {num_inf}")

    # Plot distribution
    finite_comp_fitness = comp_fitness[np.isfinite(comp_fitness)]

    plt.figure(figsize=(10, 6))
    plt.hist(finite_comp_fitness, bins=50, alpha=0.7)
    plt.xlabel('Competition Fitness')
    plt.ylabel('Count')
    plt.title('DNS Competition Fitness Distribution')
    plt.show()

    # Correlation with raw fitness
    plt.figure(figsize=(10, 6))
    plt.scatter(fitness[np.isfinite(comp_fitness)],
               finite_comp_fitness, alpha=0.5)
    plt.xlabel('Raw Fitness')
    plt.ylabel('Competition Fitness')
    plt.title('Raw vs Competition Fitness')
    plt.show()
```

### Visualize Descriptor Space Coverage

```python
from sklearn.decomposition import PCA

def visualize_dns_archive(genotypes, fitness, descriptors):
    """
    Visualize DNS archive in 2D descriptor space.
    """
    # PCA to 2D if needed
    if descriptors.shape[1] > 2:
        pca = PCA(n_components=2)
        descriptors_2d = pca.fit_transform(descriptors)
    else:
        descriptors_2d = descriptors

    # Color by fitness
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        descriptors_2d[:, 0],
        descriptors_2d[:, 1],
        c=fitness,
        cmap='viridis',
        alpha=0.6
    )
    plt.colorbar(scatter, label='Fitness')
    plt.xlabel('Descriptor Dimension 1')
    plt.ylabel('Descriptor Dimension 2')
    plt.title('DNS Archive (colored by fitness)')
    plt.show()
```

---

## Common Pitfalls

### 1. Integer Overflow with Infinite Values

**Problem**: `np.inf` in competition fitness can cause selection issues

**Solution**: Handle infinity explicitly in selection

```python
# Separate infinite from finite values
inf_mask = np.isinf(competition_fitness)
finite_mask = ~inf_mask

# Prioritize infinite (local maxima)
# Then select best finite values
```

### 2. Empty Fitter Set for All Solutions

**Problem**: If all solutions have same fitness, all get infinite competition fitness

**Solution**: Add small random noise to break ties

```python
if np.all(np.isinf(competition_fitness)):
    # All solutions are equally fit
    # Select randomly or by other criterion
    competition_fitness = np.random.rand(len(fitness))
```

### 3. Descriptor Space Normalization

**Problem**: Descriptors with different scales bias distance computation

**Solution**: Normalize descriptors

```python
def normalize_descriptors(descriptors):
    """
    Z-score normalization per dimension.
    """
    mean = np.mean(descriptors, axis=0)
    std = np.std(descriptors, axis=0)
    return (descriptors - mean) / (std + 1e-8)
```

### 4. K Too Large

**Problem**: If k > number of fitter solutions, distances include duplicates

**Solution**: Already handled in implementation with `k_actual = min(k, len(fitter_indices))`

---

## Summary

DNS implementation requires:
1. **Competition function**: Distance to k-nearest fitter solutions
2. **Selection**: Top-N by competition fitness (handling infinities)
3. **Standard GA components**: Reproduction, evaluation, selection

**Key Implementation Insights**:
- Vectorize distance computations for efficiency
- Handle local maxima (infinite competition fitness) explicitly
- Normalize descriptors if dimensions have different scales
- Parameter k = 5 is robust; no tuning needed

**Practical Recommendation**: Start with provided implementation, test on simple domain, then optimize with JAX/GPU if needed for large populations.

---

**See [[DNS]] for high-level overview and when to use DNS vs. MAP-Elites.**
