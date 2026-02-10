# Self-Play — Detailed Implementation Notes

> **Quick overview**: [[SP]]

## Overview

Self-Play RL trains agents by competing against copies of themselves, creating an automatic curriculum where the opponent improves as the learner does.

## Fundamental Training Loop

```
Initialize: Policy π₀
for iteration t = 1 to convergence:
    1. π_opponent ← select_opponent(π₀, ..., πₜ₋₁)
    2. Collect trajectories: πₜ vs π_opponent
    3. Update πₜ via RL (PPO, SAC, etc.)
    4. Optionally add πₜ to opponent pool
```

## Key Variants

| Variant | Opponent Selection | Properties |
|---------|-------------------|------------|
| **Naive SP** | Latest policy πₜ₋₁ | Simple, prone to cycling |
| **Historical SP** | Uniform sample from Π = {π₀,...,πₜ₋₁} | More stable |
| **PFSP** | Prioritized: sample proportional to difficulty | Focuses on hard opponents |
| **Population-based** | Multiple coevolving agents | PSRO, NeuPL |

## Common Extensions

1. **Population-based**: [[PSRO]], [[NeuPL]]
2. **Diversity mechanisms**: Reward diversity, entropy bonuses
3. **Curriculum design**: Opponent difficulty scheduling
4. **Imperfect information**: [[FSP]], [[CFR]]

## Challenges

- **Cycling**: A beats B beats C beats A
- **Forgetting**: Losing competence against old strategies
- **Overfitting**: Narrow adaptation to opponent's style
- **Exploration**: Insufficient coverage of strategy space

## Metrics

| Metric | What it Measures |
|--------|-----------------|
| **Elo Rating** | Relative skill (dynamic ladder) |
| **Win Rate** | Against fixed baselines |
| **Exploitability** | Distance from Nash |
| **NashConv** | Sum of player exploitabilities |
| **Population Diversity** | Behavioral variance |

## Applications

- Board games (AlphaZero, AlphaGo)
- Card games (Pluribus, Libratus)
- Robotics (competitive manipulation)
- Multi-agent RL (StarCraft, DOTA)

## References

- [Emergent Complexity via Multi-Agent Competition](https://arxiv.org/abs/1710.03748)
- [AlphaZero (Silver et al., 2017)](https://arxiv.org/abs/1712.01815)
- [PSRO (Lanctot et al., 2017)](https://arxiv.org/abs/1711.00832)

## Related

- [[SP]] — Quick overview (vanilla self-play)
- [[AlphaZero]] / [[AlphaZero_detailed]] — Perfect-info self-play
- [[FSP]] / [[FSP_detailed]] — Imperfect-info self-play
- [[PSRO]] / [[PSRO_detailed]] — Population-based self-play
- [[NeuPL]] / [[NeuPL_detailed]] — Single-network population
