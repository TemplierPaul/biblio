# RL Methods for PSRO/NeuPL

## Overview
Modern RL algorithms used as **best-response oracles** within PSRO and NeuPL frameworks. The choice of RL method significantly impacts sample efficiency, stability, and overall PSRO convergence.

## Recommendation Table

| Method | Sample Eff. | Stability | Actions | Recommendation |
|--------|-------------|-----------|---------|----------------|
| **MPO** | High | Very High | Both | Best for sample-limited BR |
| **CrossQ** | Very High | High | Continuous | Best for continuous control |
| **SAC** | High | High | Continuous | Solid baseline |
| **TD3** | High | High | Continuous | Alternative to SAC |
| **PPO** | Medium | Very High | Both | Best for discrete actions |
| **DQN** | Medium | Medium | Discrete | Simple baseline |

## MPO (Maximum a Posteriori Policy Optimization)

EM-style updates with KL constraints:
- **E-step**: Reweight samples → q(a|s) ∝ exp(A(s,a)/η) · π_old(a|s)
- **M-step**: Fit policy → min KL(q || π) + λ·KL(π || π_old)
- Very stable; best when BR budget is limited

## CrossQ

BatchNorm in critic, no target networks, UTD=1:
- **Simple**: Fewer moving parts than SAC/TD3
- **Efficient**: Best sample efficiency with minimal complexity
- Best for continuous-action environments

## Integration Pattern

```python
def train_best_response(opponent_pop, meta_strategy, rl_algo):
    for episode in range(num_episodes):
        opp_idx = sample(meta_strategy)
        opponent = opponent_pop[opp_idx]
        trajectory = play_game(learner, opponent)
        rl_algo.update(trajectory)
    return learner
```

> Detailed implementation: [[RL_Methods_for_PSRO_detailed]]

## Related
- [[PSRO]] / [[NeuPL]] — Frameworks using these methods
- [[RNN_Policies]] — Recurrent policy implementations
- [[JAX_Tools]] — JAX ecosystem for implementation
