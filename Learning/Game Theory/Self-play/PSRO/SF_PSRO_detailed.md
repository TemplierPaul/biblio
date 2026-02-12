# SF-PSRO — Detailed Implementation Notes

> **Quick overview**: [[SF_PSRO]]

## Paper
**Title**: Simulation-Free PSRO: Removing Game Simulation from Policy Space Response Oracles
**Venue**: AAMAS 2026
**Authors**: Liu et al. (BUPT)

## The Bottleneck: Game Simulation (GS)
In PSRO, evaluating the meta-game requires simulating matches between all combinations of strategies.
- **Cost**: $(M^N - (M-1)^N) \times K$ simulations per iteration.
- **Impact**: As $N$ (players) or $M$ (iterations) grows, GS dominates runtime, far exceeding the time spent training policies.

## Algorithm: Dynamic Window

Instead of an ever-growing population, SF-PSRO maintains a **Dynamic Window** of size $W$.

### Steps per Iteration:
1.  **Initialize**: Start with window $X^w$.
2.  **Train BR ($\beta_i$)**: Train against a meta-strategy $\sigma_{-i}$ derived from $X^w$ (e.g., Uniform or Regret-Matched).
    - **Filling**: During training, record results of $\beta_i$ vs $X^w_{-i}$ to update a "sketchy" payoff matrix $P_s$.
3.  **Update Window**:
    - Add $\beta_i$ to $X^w$.
    - If $|X^w| > W$:
        - Run **Nash Clustering** on $P_s$ to rank strategies.
        - **Eliminate** the weakest strategy (from the lowest-ranked cluster).

## Nash Clustering for Elimination
How to pick which strategy to delete?
- **Bad idea**: Lowest average payoff (might be a specialist that counters a strong strategy).
- **Good idea**: **Nash Clustering**.
    1. Compute Nash Equilibrium (NE) of the current window.
    2. Strategies in the support of NE form Cluster 1 (Top Tier).
    3. Remove them, re-compute NE on remainder. Support forms Cluster 2.
    4. Repeat until empty.
    5. Remove a strategy from the **last** cluster (lowest Relative Population Performance).

## Pseudocode

```python
def sf_psro(game, window_size=30):
    window = [RandomPolicy()]
    sketchy_matrix = zeros((1,1))
    
    while True:
        # 1. Train Best Response
        # (outcomes collected during training fill the sketchy matrix)
        new_strategy, outcomes = train_br(opponent_set=window)
        
        # 2. Update Sketchy Matrix
        sketchy_matrix = update(sketchy_matrix, outcomes)
        
        # 3. Update Window
        window.append(new_strategy)
        
        if len(window) > window_size:
            # 4. Identify strategy to eliminate
            clusters = nash_clustering(sketchy_matrix)
            worst_cluster = clusters[-1]
            # Pick strategy with smallest weight in that cluster's equilibrium
            victim = argmin(worst_cluster.weights)
            
            window.remove(victim)
            sketchy_matrix.remove_row_col(victim)
            
    return meta_strategy(window)
```

## Experimental Results
- **Leduc Poker**: Outperforms Vanilla PSRO and PSD-PSRO.
- **Goofspiel (2p & 3p)**: Competitive performance with significantly lower runtime.
- **Efficiency**: In 3-player Goofspiel, SF-PSRO is **~6x faster** than standard PSRO.

## Compatibility
SF-PSRO is a framework that can be combined with:
- **MSS methods**: MRCP (Minimum Regret Constrained Profiles), Anytime PSRO.
- **BRS methods**: Behavioral Diversity (PSD-PSRO).

## Related
- [[PSRO]] — The baseline being optimized.
- [[A_PSRO]] — Another efficient PSRO variant.
- [[Anytime_PSRO]] — Related concept (removing GS via no-regret updates).
