# GAME - Detailed Implementation Notes

> **Quick overview**: [[GAME]]

## Paper Information

**Title**: Tournament Informed Adversarial Quality Diversity
**Authors**: Timothée Anne, Noah Syrkis, Meriem Elhosni, Florian Turati, Alexandre Manai, Franck Legendre, Alain Jaquier, Sebastian Risi
**Venue**: GECCO 2026
**Code**: https://github.com/Timothee-ANNE/GAME_tournament_informed

## Full Algorithm Specification

### Algorithm 1: GAME Main Loop

```python
def GAME(N_gen, N_task, N_cell, N_budget, task_selection):
    # Initialize with random Blue solutions as tasks
    tasks = [random_solution(S_Blue) for _ in range(N_task)]
    bootstrap_evals = []  # Empty initially

    for gen_id in range(N_gen):
        # Determine which side to optimize
        S = S_Red if gen_id % 2 == 1 else S_Blue

        # Run MTMB-ME
        archives = MTMB_ME(
            tasks=tasks,
            search_space=S,
            bootstrap=bootstrap_evals,
            N_cell=N_cell,
            N_budget=N_budget
        )

        # Select next generation of tasks
        tasks, bootstrap_evals = task_selection(
            archives=archives,
            old_tasks=tasks
        )

    return tasks
```

### Algorithm 2: MTMB-ME (Multi-Task Multi-Behavior MAP-Elites)

```python
def MTMB_ME(tasks, search_space, bootstrap, N_cell, N_budget):
    # Initialize N_task growing archives
    archives = [GrowingArchive(N_cell) for _ in range(len(tasks))]

    # Bootstrap from previous generation tournament
    for (task_id, solution, fitness, behavior) in bootstrap:
        archives[task_id].update(solution, fitness, behavior)

    # Main evaluation loop
    for i in range(N_budget):
        # Select random task
        task_id = random.randint(0, len(tasks) - 1)
        task = tasks[task_id]

        # Generate candidate solution
        if len(archives[task_id]) < N_init:
            solution = random_solution(search_space)
        else:
            parent = archives[task_id].select_random()
            solution = mutate(parent)

        # Evaluate against task
        fitness, behavior = evaluate(solution, task)

        # Update archive
        archives[task_id].update(solution, fitness, behavior)

    return archives
```

### Algorithm 3: Growing Archive Update

```python
class GrowingArchive:
    def __init__(self, max_size):
        self.max_size = max_size
        self.centroids = []       # Behavior centroids
        self.elites = []          # Current elites
        self.backup_elites = []   # Backup for drift repair

    def update(self, solution, fitness, behavior):
        if len(self.centroids) < self.max_size:
            # Add new cell
            self._add_cell(solution, fitness, behavior)
        else:
            # Check if behavior novel enough or fitness better
            distances = [dist(behavior, c) for c in self.centroids]
            d_min_centroids = min(pairwise_distances(self.centroids))
            d_closest = min(distances)
            closest_id = argmin(distances)

            if d_closest > d_min_centroids:
                # Novel behavior → growth
                self._grow(solution, fitness, behavior)
            elif fitness > self.elites[closest_id].fitness:
                # Better fitness → replacement
                self.elites[closest_id] = (solution, fitness, behavior)
                self.backup_elites[closest_id].append((solution, fitness, behavior))

    def _grow(self, solution, fitness, behavior):
        # Find closest pair of centroids
        j, k = argmin_pair(pairwise_distances(self.centroids))

        # Remove the one with smaller distance to other centroids
        dj = min_distance_to_others(self.centroids, j)
        dk = min_distance_to_others(self.centroids, k)
        remove_id = j if dj < dk else k

        # Replace with new centroid
        self.centroids[remove_id] = behavior
        self.elites[remove_id] = (solution, fitness, behavior)
        self.backup_elites[remove_id] = [(solution, fitness, behavior)]

        # Repair holes (elites that drifted to wrong cell)
        for i in range(len(self.centroids)):
            correct_cell = self._find_cell(self.elites[i].behavior)
            if correct_cell != i:
                # Elite drifted, restore from backup
                self.elites[i] = self._best_from_backup(i)

    def _find_cell(self, behavior):
        """Find closest centroid to behavior"""
        distances = [dist(behavior, c) for c in self.centroids]
        return argmin(distances)
```

## Task Selection Algorithms

### Behavior Task Selection (Original)

```python
def behavior_task_selection(archives, old_tasks):
    # Aggregate all elites and behaviors
    all_elites = []
    all_behaviors = []
    for archive in archives:
        for elite in archive.elites:
            all_elites.append(elite.solution)
            all_behaviors.append(elite.behavior)

    # K-means clustering on behaviors
    clusters = kmeans(all_behaviors, k=N_task)

    # Select elite of each cluster (by fitness)
    tasks = []
    for cluster in clusters:
        best_elite = max(cluster, key=lambda e: e.fitness)
        tasks.append(best_elite.solution)

    # Bootstrap tournament: new tasks vs old tasks
    bootstrap = []
    for new_task in tasks:
        for old_task in old_tasks:
            f1, b1 = evaluate(new_task, old_task)
            f2, b2 = evaluate(old_task, new_task)
            bootstrap.append((task_id_for(new_task), new_task, f1, b1))
            bootstrap.append((task_id_for(old_task), old_task, f2, b2))

    return tasks, bootstrap
```

### Ranking Task Selection (Proposed - Best)

```python
def ranking_task_selection(archives, old_tasks):
    # Aggregate all elites
    all_elites = []
    for archive in archives:
        all_elites.extend(archive.elites)

    # Tournament: all elites vs all old tasks
    tournament_matrix = np.zeros((len(all_elites), len(old_tasks)))

    for i, elite in enumerate(all_elites):
        for j, task in enumerate(old_tasks):
            fitness, _ = evaluate(elite.solution, task)
            tournament_matrix[i, j] = fitness

    # Compute ranking vectors
    ranking_vectors = []
    for i in range(len(all_elites)):
        fitness_vector = tournament_matrix[i, :]

        # Double argsort gives ranking
        ranking = np.argsort(np.argsort(fitness_vector))

        # Normalize to [-1, 1]
        normalized_ranking = 2 * ranking / (len(old_tasks) - 1) - 1

        ranking_vectors.append(normalized_ranking)

    # Cluster by ranking vectors
    clusters = kmeans(ranking_vectors, k=N_task)

    # Select elite of each cluster (by average fitness)
    tasks = []
    for cluster in clusters:
        avg_fitnesses = [np.mean(tournament_matrix[e.id, :]) for e in cluster]
        best_elite = cluster[argmax(avg_fitnesses)]
        tasks.append(best_elite.solution)

    # Bootstrap with tournament evaluations involving selected elites
    bootstrap = extract_evaluations_with_selected(tournament_matrix, tasks)

    return tasks, bootstrap
```

### Pareto Task Selection (Proposed - Alternative)

```python
def pareto_task_selection(archives, old_tasks):
    # Aggregate all elites
    all_elites = []
    for archive in archives:
        all_elites.extend(archive.elites)

    # Tournament: all elites vs all old tasks
    fitness_vectors = []
    for elite in all_elites:
        fitness_vec = []
        for task in old_tasks:
            fitness, _ = evaluate(elite.solution, task)
            fitness_vec.append(fitness)
        fitness_vectors.append(np.array(fitness_vec))

    # Apply NSGA-III to select N_task elites
    # Each task = one objective to maximize
    selected_indices = nsga3_select(
        fitness_vectors,
        n_select=N_task
    )

    tasks = [all_elites[i].solution for i in selected_indices]

    # Bootstrap tournament
    bootstrap = []
    for new_task in tasks:
        for old_task in old_tasks:
            f, b = evaluate(new_task, old_task)
            bootstrap.append((task_id, new_task, f, b))

    return tasks, bootstrap
```

## Adversarial QD Measures - Detailed Computation

### Setup: Inter-Variant Tournament

```python
def inter_variant_tournament(variants, N_task, N_rep):
    """
    Evaluate all final tasks from all variants against each other

    Args:
        variants: List of variant results (each has N_task tasks)
        N_task: Number of tasks per variant
        N_rep: Number of replications per variant

    Returns:
        Tournament results matrix
    """
    # Collect all solutions
    all_red = []
    all_blue = []

    for variant in variants:
        for rep in range(N_rep):
            red_tasks = variant.red_tasks[rep]
            blue_tasks = variant.blue_tasks[rep]
            all_red.extend(red_tasks)
            all_blue.extend(blue_tasks)

    # Evaluate all vs all
    tournament = np.zeros((len(all_red), len(all_blue)))

    for i, red in enumerate(all_red):
        for j, blue in enumerate(all_blue):
            fitness_red, _ = evaluate(red, blue)
            tournament[i, j] = fitness_red

    return tournament, all_red, all_blue
```

### Measure 1: Win Rate

```python
def compute_win_rate(S_red, all_blue, tournament):
    """
    Maximum win rate achieved by any solution in S_red
    """
    win_rates = []

    for s_red in S_red:
        # Count wins against all blue opponents
        wins = 0
        for s_blue in all_blue:
            fitness = tournament[s_red.id, s_blue.id]
            if fitness > 0.5:
                wins += 1

        win_rate = wins / len(all_blue)
        win_rates.append(win_rate)

    return max(win_rates)
```

### Measure 2: ELO Score

```python
def compute_elo_score(S_red, all_solutions, tournament):
    """
    Compute ELO scores and return max normalized rank
    """
    # Initialize ELO scores
    elo = {sol.id: 1500 for sol in all_solutions}

    # Update ELO from all matchups
    K = 32  # ELO K-factor

    for i, sol1 in enumerate(all_solutions):
        for j, sol2 in enumerate(all_solutions):
            if i == j:
                continue

            fitness1 = tournament[i, j]

            # Expected scores
            expected1 = 1 / (1 + 10**((elo[j] - elo[i])/400))

            # Actual score (1 if win, 0 if loss, 0.5 if tie)
            actual1 = 1 if fitness1 > 0.5 else (0 if fitness1 < 0.5 else 0.5)

            # Update
            elo[i] += K * (actual1 - expected1)

    # Rank all solutions by ELO
    ranked = sorted(all_solutions, key=lambda s: elo[s.id])
    ranks = {s.id: i / len(all_solutions) for i, s in enumerate(ranked)}

    # Return max rank of S_red solutions
    max_rank = max(ranks[s.id] for s in S_red)

    return max_rank
```

### Measure 3: Robustness

```python
def compute_robustness(S_red, all_blue, tournament):
    """
    Best worst-case fitness
    """
    robustness_scores = []

    for s_red in S_red:
        worst_case = min(
            tournament[s_red.id, s_blue.id]
            for s_blue in all_blue
        )
        robustness_scores.append(worst_case)

    return max(robustness_scores)
```

### Measure 4: Coverage

```python
def compute_coverage(S_red, all_red, all_blue, tournament, N_task):
    """
    Percentage of ranking-based clusters covered
    """
    # Compute ranking vectors for all red solutions
    ranking_vectors = []

    for s_red in all_red:
        fitness_vec = [tournament[s_red.id, s_blue.id] for s_blue in all_blue]
        ranking = np.argsort(np.argsort(fitness_vec))
        normalized = 2 * ranking / (len(all_blue) - 1) - 1
        ranking_vectors.append(normalized)

    # Cluster all solutions
    clusters = kmeans(ranking_vectors, k=N_task)

    # Find which clusters contain solutions from S_red
    covered_clusters = set()
    for s_red in S_red:
        cluster_id = clusters.predict([ranking_vectors[s_red.id]])[0]
        covered_clusters.add(cluster_id)

    coverage = len(covered_clusters) / N_task

    return coverage
```

### Measure 5: Expertise

```python
def compute_expertise(S_red, all_blue, tournament):
    """
    Minimum of maximum fitness against each opponent
    (worst counter-solution quality)
    """
    expertise_scores = []

    for s_blue in all_blue:
        # Find best counter from S_red
        best_counter_fitness = max(
            tournament[s_red.id, s_blue.id]
            for s_red in S_red
        )
        expertise_scores.append(best_counter_fitness)

    # Return worst counter-solution
    return min(expertise_scores)
```

### Measure 6: Adversarial QD-Score

```python
def compute_aqd_score(S_red, all_blue, tournament):
    """
    Minimum number of blue solutions needed to beat all red solutions

    This is a set cover problem - use greedy approximation
    """
    uncovered = set(S_red)
    selected_blue = []

    while uncovered:
        # Find blue solution that beats most uncovered red solutions
        best_blue = None
        best_coverage = 0

        for s_blue in all_blue:
            coverage = sum(
                1 for s_red in uncovered
                if tournament[s_red.id, s_blue.id] < 0.5
            )
            if coverage > best_coverage:
                best_coverage = coverage
                best_blue = s_blue

        if best_coverage == 0:
            # No blue can beat remaining red (some red are unbeatable)
            break

        # Add best blue to selection
        selected_blue.append(best_blue)

        # Remove covered red solutions
        uncovered = {
            s_red for s_red in uncovered
            if tournament[s_red.id, best_blue.id] >= 0.5
        }

    return len(selected_blue)
```

## Implementation Tips

### Parallelization

**JAX advantages**:
```python
import jax
import jax.numpy as jnp

# Vectorized evaluation
@jax.jit
@jax.vmap  # Vectorize over batch dimension
def evaluate_batch(solutions, tasks):
    """Evaluate batch of solutions against batch of tasks"""
    # All operations in JAX for GPU acceleration
    return fitnesses, behaviors

# Parallel execution on GPU
fitnesses, behaviors = evaluate_batch(
    jnp.array(solutions),
    jnp.array(tasks)
)
```

### Visual Embedding (CLIP)

```python
from transformers import CLIPProcessor, CLIPModel

class VEMBehaviorExtractor:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def extract_behavior(self, game_state):
        """
        Extract behavior descriptor from game visualization
        """
        # Render game state to image
        image = render_game(game_state)

        # Get CLIP embedding
        inputs = self.processor(images=image, return_tensors="pt")
        embedding = self.model.get_image_features(**inputs)

        return embedding.detach().numpy().flatten()
```

### Mutation Operator

```python
def mutate_mlp(parent_weights, mutation_rate=0.3, noise_std=0.1):
    """
    Mutate MLP weights with Gaussian noise

    Args:
        parent_weights: Dictionary of layer weights
        mutation_rate: Fraction of weights to mutate
        noise_std: Standard deviation of Gaussian noise
    """
    child_weights = {}

    for layer_name, weights in parent_weights.items():
        # Create mutation mask
        mask = np.random.random(weights.shape) < mutation_rate

        # Apply Gaussian noise to selected weights
        noise = np.random.normal(0, noise_std, weights.shape)
        child_weights[layer_name] = weights + mask * noise

    return child_weights
```

## Experimental Configuration

### Hyperparameters Used in Paper

```python
CONFIG = {
    'n_generations': 10,
    'n_tasks': 50,
    'n_cells': 20,
    'n_budget': 100_000,  # Per generation

    # Neural network
    'hidden_layers': [32, 16],
    'activation': 'tanh',

    # Mutation
    'mutation_rate': 0.3,
    'mutation_std': 0.1,

    # Tournament
    'n_replications': 20,

    # Task selection variants
    'ranking_epsilon': 0.25,  # Extract probability
}
```

### Computational Requirements

**Per replication**:
- Evaluations: 1.5M total (including tournament)
- GPU time: ~4-6 hours (depends on game complexity)
- Memory: ~8GB GPU RAM

**Full experiment** (4 variants × 20 reps × 3 games):
- Total GPU hours: ~1000-1500
- Recommended: Multi-GPU cluster

## Validation

### Statistical Testing

```python
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

def compare_variants(measure_values, variant_names):
    """
    Pairwise comparison with Holm-Bonferroni correction

    Args:
        measure_values: Dict[variant_name -> array of values]
        variant_names: List of variant names

    Returns:
        Dict of p-values and significance
    """
    # Find best performing variant
    best_variant = max(
        variant_names,
        key=lambda v: np.median(measure_values[v])
    )

    # Compare all others to best
    p_values = []
    for variant in variant_names:
        if variant == best_variant:
            p_values.append(1.0)
        else:
            _, p = wilcoxon(
                measure_values[best_variant],
                measure_values[variant],
                alternative='greater'
            )
            p_values.append(p)

    # Holm-Bonferroni correction
    reject, p_corrected, _, _ = multipletests(
        p_values,
        alpha=0.05,
        method='holm'
    )

    return {
        variant: {
            'p_value': p,
            'significant': r
        }
        for variant, p, r in zip(variant_names, p_corrected, reject)
    }
```

## Key Implementation Insights

1. **Growing archives** essential for unknown behavior space
2. **Tournament size** grows quadratically - consider approximate versions for large populations
3. **CLIP embeddings** work well but domain-specific features may be better
4. **JAX parallelization** critical for computational efficiency
5. **Ranking vectors** more informative than raw fitness for adversarial diversity

## References

- Paper: Anne et al., "Tournament Informed Adversarial Quality Diversity", GECCO 2026
- Code: https://github.com/Timothee-ANNE/GAME_tournament_informed
- PATA-EC: Wang et al., "Enhanced POET", NeurIPS 2020
- MTMB-ME: Anne et al., "Multi-Task Multi-Behavior MAP-Elites", GECCO 2023

## Related

- [[GAME]] — High-level overview
- [[MAP-Elites]] — Base QD algorithm
- [[Enhanced_POET]] — PATA-EC concept
- [[Quality_Diversity]] — General QD concepts
