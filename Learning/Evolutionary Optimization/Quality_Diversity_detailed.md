# Quality-Diversity — Detailed Notes

> **Quick overview**: [[Quality_Diversity]]

## QD Algorithms in Depth

### 1. MAP-Elites
See [[MAP_Elites_detailed]] for full details.

### 2. Novelty Search with Local Competition (NSLC)

```
for each individual:
    novelty = mean_distance_to_k_nearest(behavior_archive)
    local_quality = rank_among_k_nearest(fitness)
    score = novelty + λ · local_quality
```

### 3. CMA-MAP-Elites

Combines CMA-ES with MAP-Elites:
```
Each archive cell has its own CMA-ES distribution
Mutation: sample from cell's CMA-ES
Offspring fitness → update CMA-ES of landing cell
```

### 4. Policy-Gradient MAP-Elites (PGA-MAP-Elites)

```
Two emitter types running concurrently:
    GA Emitter: Genetic variation (crossover + mutation)
    PG Emitter: Policy gradient updates (actor-critic)

Combined in QDax as MixEmitter
```

### 5. DCRL-MAP-Elites (GECCO 2023 Best Paper)

Depth-Conditioned RL: conditions the policy on a "depth" parameter that controls how close to archive boundaries the agent should operate.

## Comparison Table

| Algorithm | Mutation | Scalability | RL-compatible | Key Strength |
|-----------|----------|-------------|---------------|-------------|
| **MAP-Elites** | Random | Medium | No | Simplicity |
| **CVT-MAP-Elites** | Random | High | No | High-dim BD |
| **CMA-MAP-Elites** | CMA-ES | Medium | No | Better optimization |
| **PGA-MAP-Elites** | PG + GA | High | Yes | RL + QD |
| **DCRL-MAP-Elites** | RL | High | Yes | SOTA in RL-QD |

## Applications

1. **Robot repertoire**: Library of gaits for different terrains
2. **Game AI**: Diverse NPC behaviors, opponent pools for PSRO
3. **Drug discovery**: Diverse high-affinity molecules
4. **Design**: Aerodynamic shapes, architecture

## QDax Framework (JAX)

```python
# Hardware-accelerated QD in JAX
import qdax

# Key modules:
# qdax.core.map_elites — MAP-Elites
# qdax.core.emitters — Mutation/variation operators
# qdax.environments — Brax-based benchmarks
# qdax.utils.metrics — QD-Score, coverage, etc.
```

## Code Resources

- [QDax (JAX)](https://github.com/adaptive-intelligent-robotics/QDax) — GPU-accelerated
- [pyribs](https://pyribs.org/) — Python QD library
- [sferes2 (C++)](https://github.com/sferes2/sferes2) — Original framework

## References

- [Quality-Diversity: A New Frontier (Pugh et al., 2016)](https://www.frontiersin.org/articles/10.3389/frobt.2016.00040)
- [MAP-Elites (Mouret & Clune, 2015)](https://arxiv.org/abs/1504.04909)
- [QDax (Lim et al., 2022)](https://qdax.readthedocs.io)

## Related

- [[Quality_Diversity]] — Quick overview
- [[MAP_Elites]] / [[MAP_Elites_detailed]] — Core QD algorithm
- [[Evolutionary Optimization]] — Parent topic
