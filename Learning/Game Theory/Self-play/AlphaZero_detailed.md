# AlphaZero — Detailed Implementation Notes

> **Quick overview**: [[AlphaZero]]

## Paper

**Title**: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm

**Authors**: Silver et al., DeepMind

**Year**: 2017

**ArXiv**: [1712.01815](https://arxiv.org/abs/1712.01815)

## MCTS Details

Each node maintains: Q(s,a), N(s,a), P(s,a).

**Selection** (traverse tree):
```
a* = argmax(Q(s,a) + U(s,a))
U(s,a) = c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a))
```

**Expansion**: Evaluate leaf with NN → (p, v) = f_θ(s), init P(s,·) = p

**Backup**:
```
N(s,a) += 1
Q(s,a) += (v - Q(s,a)) / N(s,a)
```

**Play**: π(a|s) = N(s,a)^(1/τ) / Σ_b N(s,b)^(1/τ)

## Network Architecture

```
Input: Raw board representation s
    ↓
Residual CNN Backbone
    20 res blocks (Chess/Shogi) or 40 (Go)
    Each: Conv → BatchNorm → ReLU → Conv → BatchNorm → Skip
    ↓
Two heads:
    Policy: → Softmax(Linear(features)) = p
    Value:  → Tanh(Linear(features)) = v
```

## Loss Function

$$\mathcal{L} = \underbrace{(v - z)^2}_{\text{value}} - \underbrace{\pi^T \log \mathbf{p}}_{\text{policy}} + \underbrace{c\|\theta\|^2}_{\text{L2 reg}}$$

## Self-Play Data Generation

```
1. Start from initial state
2. For each move:
   - Run MCTS (800 sims) with current network
   - Sample action: a ~ N(s,a)^(1/τ)
   - Add Dirichlet noise at root: P = (1-ε)·p + ε·η
3. Store (s, π, z) for all positions
4. Set z = game outcome for all positions
```

## Evaluation

**Tournament-based**:
```
Play N games: new_network vs current_best
if win_rate > 55%: replace current_best
```

**Continuous**: Always use latest network, track Elo.

## Hyperparameters (Original)

| Parameter | Value |
|-----------|-------|
| MCTS sims/move | 800 (Chess/Shogi), 1600 (Go) |
| c_puct | 2.5 |
| Dirichlet α | 0.3 |
| Batch size | 4096 |
| LR | 0.2 → 0.02 → 0.002 → 0.0002 |
| Weight decay | 1e-4 |
| Temperature τ | 1.0 first 30 moves, then 0 |
| Replay buffer | Last 1M positions |
| Resign threshold | v < -0.9 |

### Simplified (e.g. Othello tutorial)

| Parameter | Value |
|-----------|-------|
| MCTS sims | 25/move |
| Episodes/iter | 100 |
| Iterations | 80 |
| Acceptance | 55% win rate |
| Hardware | Single K80 GPU, ~3 days |

## Results

| Game | Opponent | Result | Training Time |
|------|----------|--------|---------------|
| Chess | Stockfish 8 | 28-72-0 (W-D-L) | ~4h |
| Shogi | Elmo | 90-8-2 | ~2h |
| Go | AlphaGo Lee | 100-0 | ~8h |

## Metrics Logged

- Policy loss (cross-entropy), Value loss (MSE), Value accuracy
- Episode length, MCTS statistics, Elo rating, Win rate vs previous

## Code Resources

- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) — Multi-framework
- [michaelnny/alpha_zero](https://github.com/michaelnny/alpha_zero) — PyTorch
- [Simple AlphaZero Tutorial](https://suragnair.github.io/posts/alphazero.html)
- OpenSpiel library

## References

- [Mastering Chess and Shogi (Silver et al., 2017)](https://arxiv.org/abs/1712.01815)
- [Mastering Go without Human Knowledge](https://www.nature.com/articles/nature24270)
- [Science paper (2018)](https://www.science.org/doi/10.1126/science.aar6404)

## Related

- [[AlphaZero]] — Quick overview (with formulas and interview Q&A)
- [[Self-play/Self-play|Self-play]] — Core paradigm
- [[PSRO]] — Population-based extension
