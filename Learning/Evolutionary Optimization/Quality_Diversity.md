# Quality-Diversity Optimization

## Definition
Quality-Diversity (QD) is an optimization paradigm that seeks a **diverse collection of high-performing solutions** rather than a single optimum. QD simultaneously optimizes for:
1. **Quality**: Solutions should perform well
2. **Diversity**: Solutions should be behaviorally different

## Core Concepts

### QD Score
```
QD-Score = Σ performance(solution) for all solutions in archive
```
Captures both quality and diversity in a single metric.

### Behavior Space
- **Behavior Descriptors (BD)**: Features characterizing *how* a solution behaves
- **Archive**: Data structure storing best solution per behavioral niche
- User chooses BD based on domain knowledge

### Example (Robot Locomotion)
- **Objective**: Maximize distance traveled
- **Behavior**: (final_x, final_y) — where the robot ends up
- **Archive**: Grid over (x, y), best-walking solution per region

## Main QD Algorithms

| Algorithm | Key Idea | BD Space |
|-----------|----------|----------|
| **[[MAP_Elites]]** | Grid archive, best per cell | Discretized |
| **CMA-MAP-Elites** | CMA-ES mutation + MAP-Elites | Continuous |
| **PGA-MAP-Elites** | Policy Gradient + MAP-Elites | RL tasks |
| **NSLC** | Novelty + local competition | Distance-based |
| **NSGA-QD** | Multi-objective QD | Pareto-based |

## QD in RL and Game Theory

- **Diverse opponent pools** for PSRO population seeding
- **Behavioral exploration** for harder games
- **Curriculum generation** — diverse tasks for training
- **Robustness** — repertoire of fallback behaviors

## Interview Relevance
- **Single-objective vs QD?** QD is essential when you need behavior repertoires or robustness
- **QD-Score captures what?** Both quality AND diversity — unlike just max fitness
- **Link to MAP-Elites?** MAP-Elites is the canonical QD algorithm

> Detailed reading: [[Quality_Diversity_detailed]]

## Related
- [[MAP_Elites]] / [[MAP_Elites_detailed]] — Core QD algorithm
- [[Evolutionary Optimization]] — Parent topic
