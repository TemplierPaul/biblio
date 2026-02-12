# SP-PSRO — Detailed Implementation Notes

> **Quick overview**: [[SP_PSRO]]

## Paper
**Title**: Self-Play PSRO: Toward Optimal Populations in Two-Player Zero-Sum Games
**Venue**: NeurIPS 2022
**Authors**: McAleer et al. (CMU/UCI)

## The Problem with Pure Strategies
In games like Uniform RPS (N actions), the Nash Equilibrium mixes uniformly over all N actions.
- **PSRO/APSRO**: Must add all N pure strategies one by one. Slow ($O(N)$).
- **SP-PSRO**: Adds a learned *mixed* strategy $\bar{\nu}$ that can match the Nash distribution immediately. Fast ($O(1)$).

## Algorithm: Three Concurrent Updates

SP-PSRO runs three processes simultaneously during each iteration:

1. **Opponent BR ($\beta_{-i}$)**: Trains against the current restricted meta-strategy $\pi^r_i$.
   - Objective: Exploit the current population.
   
2. **New Strategy ($\nu_i$)**: Trains against the Opponent BR ($\beta_{-i}$).
   - **Crucial**: Uses **off-policy data** from all interactions. Even when $\beta_{-i}$ plays against old policies in $\pi^r$, $\nu_i$ learns from that data as if it were valid self-play experience.
   
3. **Restricted Distribution ($\pi^r_i$)**: Updates via No-Regret (e.g., MWU) against $\beta_{-i}$.
   - Maintains the "Anytime" guarantee (non-increasing exploitability).

## Time-Averaging via Distillation

A raw RL policy oscillates. SP-PSRO stabilizes this by adding the **time-average** $\bar{\nu}$ to the population.

**Deep RL Implementation**:
- **Reservoir Sampling**: Store 2M steps of experience from $\nu_i$.
- **Distillation**: Train a fresh network $\bar{\nu}$ (supervised classification) to match the action distribution of the reservoir buffer.
- **Result**: $\bar{\nu}$ approximates the average strategy, which converges to Nash in self-play settings.

## Pseudocode

```python
def sp_psro_iteration(population, n_steps):
    # 1. Initialize
    nu = Policy()      # New mixed strategy
    beta = Policy()    # Opponent best response
    pi_r = Uniform(population + {nu}) # Restricted meta-strategy
    
    # 2. Train (Concurrent)
    for step in range(n_steps):
        # Sample match: pi_r vs beta
        traj = play(sample(pi_r), beta)
        
        # Update beta (On-policy vs pi_r)
        beta.update(traj)
        
        # Update nu (Off-policy vs beta)
        # Nu learns "what would happen if I played against beta"
        nu.update_off_policy(traj)
        
        # Update pi_r (No-RegretMWU)
        pi_r.update_weights(traj.rewards)

    # 3. Distill Time-Average
    nu_bar = distill(reservoir_buffer(nu))
    
    # 4. Add to population
    return population + {beta, nu_bar}
```

## Hyperparameters (Liar's Dice)

| Parameter | Value |
|-----------|-------|
| **Network** | MLP [128, 128] (DDQN) |
| **Buffer Size** | 50,000 |
| **Exploration** | Annealed $\epsilon$: $0.06 \to 0.001$ |
| **Forced $\nu$ Sampling** | $p=0.05$ (ensure $\nu$ gets some data) |
| **Distillation** | 10k epochs, reservoir size 2M |

## Experimental Results
- **Liar's Dice**: ~5x faster convergence than APSRO.
- **Battleship**: ~8x faster.
- **Repeated RPS**: ~15x faster.
- **Why?** Mixed strategies allow "jumping" to the solution rather than slowly building the support of the Nash Equilibrium.

## Related
- [[PSRO]] — Baseline
- [[A_PSRO]] — Advantage-based variant
- [[JAX_Tools]] — Implementation likely requires off-policy replay buffers (Flashbax)
