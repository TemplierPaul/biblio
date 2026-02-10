# MAP-Elites — Detailed Implementation Notes

> **Quick overview**: [[MAP_Elites]]

## Paper

**Title**: Illuminating Search Spaces by Mapping Elites

**Authors**: Jean-Baptiste Mouret, Jeff Clune

**Year**: 2015

## Core Algorithm (Full Pseudocode)

```python
class MAPElites:
    def __init__(self, bd_dims, bins_per_dim, mutation_fn):
        self.grid_shape = tuple([bins_per_dim] * bd_dims)
        self.archive = {}  # bin_index → (solution, fitness, behavior)
        self.mutation_fn = mutation_fn

    def run(self, iterations, init_population):
        # Seed archive
        for individual in init_population:
            fitness, behavior = evaluate(individual)
            bin_idx = self.discretize(behavior)
            self._try_add(bin_idx, individual, fitness, behavior)

        # Main loop
        for _ in range(iterations):
            parent = random.choice(list(self.archive.values()))
            offspring = self.mutation_fn(parent.solution)
            fitness, behavior = evaluate(offspring)
            bin_idx = self.discretize(behavior)
            self._try_add(bin_idx, offspring, fitness, behavior)

    def _try_add(self, bin_idx, solution, fitness, behavior):
        if bin_idx not in self.archive or fitness > self.archive[bin_idx].fitness:
            self.archive[bin_idx] = Entry(solution, fitness, behavior)

    def discretize(self, behavior):
        return tuple(int(b * bins) for b, bins in zip(behavior, self.grid_shape))
```

## CVT-MAP-Elites

Replaces grid discretization with Centroidal Voronoi Tessellation:

```python
from scipy.spatial import cKDTree

class CVTMAPElites(MAPElites):
    def __init__(self, bd_dims, num_niches, ...):
        # Pre-compute centroids via k-means
        samples = np.random.uniform(0, 1, (100000, bd_dims))
        _, self.centroids = kmeans2(samples, k=num_niches)
        self.tree = cKDTree(self.centroids)

    def discretize(self, behavior):
        _, idx = self.tree.query(behavior)
        return idx
```

## JAX / QDax Implementation

```python
import qdax
from qdax.core.map_elites import MAPElites

# QDax provides hardware-accelerated MAP-Elites
map_elites = MAPElites(
    scoring_function=scoring_fn,
    emitter=qdax.core.emitters.MixEmitter(
        mutation_fn=qdax.core.emitters.mutation_fn,
        variation_fn=qdax.core.emitters.iso_dd_fn
    ),
    metrics_function=qdax.utils.metrics.default_qd_metrics
)

repertoire, emitter_state, key = map_elites.init(
    init_genotypes, centroids, key
)

for generation in range(num_gens):
    repertoire, emitter_state, metrics, key = map_elites.update(
        repertoire, emitter_state, key
    )
```

## Metrics

| Metric | Formula | Measures |
|--------|---------|----------|
| **QD-Score** | Σ fitness(s) for s in archive | Quality × Diversity |
| **Coverage** | \|filled_bins\| / \|total_bins\| | Exploration breadth |
| **Max Fitness** | max(fitness) in archive | Best single solution |
| **Mean Fitness** | mean(fitness) in archive | Average quality |

## Hyperparameters

| Parameter | Value Range |
|-----------|-------------|
| Grid resolution | 50–1000 bins/dim |
| CVT niches | 1024–65536 |
| Mutation rate | 0.01–0.1 |
| Crossover prob | 0–0.5 |
| Init population | 100–10000 |
| Iso-DD mutation σ | 0.005–0.05 |

## Code Resources

- [QDax (JAX)](https://github.com/adaptive-intelligent-robotics/QDax)
- [pyribs](https://pyribs.org/)
- [sferes2 (C++)](https://github.com/sferes2/sferes2)

## References

- [Illuminating Search Spaces (Mouret & Clune, 2015)](https://arxiv.org/abs/1504.04909)
- [CVT-MAP-Elites (Vassiliades et al.)](https://ieeexplore.ieee.org/document/8000667)
- [QDax: Accelerated QD (Lim et al., 2022)](https://qdax.readthedocs.io)

## Related

- [[MAP_Elites]] — Quick overview
- [[Quality_Diversity]] — QD paradigm
- [[Evolutionary Optimization]] — Parent topic
