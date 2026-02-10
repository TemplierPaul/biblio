# MAP-Elites

## Definition
MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) is the foundational Quality-Diversity algorithm. It discretizes a user-defined **behavior descriptor** space into a grid/tessellation and maintains the highest-performing solution in each cell.

## Core Algorithm

```
Initialize archive as empty grid
Seed with random solutions

for iteration = 1 to max:
    parent ← random_select(archive)
    offspring ← mutate(parent)
    (fitness, behavior) ← evaluate(offspring)
    bin ← discretize(behavior)
    if archive[bin] is empty OR fitness > archive[bin].fitness:
        archive[bin] ← (offspring, fitness, behavior)
```

## Key Concepts
- **Behavior Descriptor (BD)**: User-defined features characterizing *how* a solution behaves
- **Archive**: Grid of bins indexed by BD; stores best solution per bin
- **QD-Score**: Σ fitness(s) for all s in archive — measures both quality and diversity
- **Coverage**: Fraction of bins filled

## Variants
| Variant | Innovation |
|---------|-----------|
| **CVT-MAP-Elites** | Centroidal Voronoi Tessellation (handles high-dim BD) |
| **CMA-MAP-Elites** | Uses CMA-ES for mutation → better continuous optimization |
| **PGA-MAP-Elites** | Policy Gradient + QD for RL tasks |
| **DCRL-MAP-Elites** | Depth-Conditioned RL (GECCO 2023 Best Paper) |
| **Parametric-Task ME** | Task descriptor as additional objective |

## Interview Relevance
- **QD vs single-objective?** QD finds diverse repertoire; useful when you need multiple solutions
- **Curse of dimensionality?** CVT-MAP-Elites handles high-dim BD via Voronoi
- **Link to PSRO?** QD can generate diverse opponent populations for game-theoretic training

> Detailed implementation: [[MAP_Elites_detailed]]

## Related
- [[Quality_Diversity]] — QD paradigm overview
- [[Evolutionary Optimization]] — Parent topic
