# Joint PSRO (JPSRO)

## Definition
Joint Policy Space Response Oracles (JPSRO) extends PSRO to general-sum n-player games by using joint policy distributions and computing Correlated Equilibria instead of Nash Equilibria.

## Motivation

### PSRO Limitations
- Designed for two-player zero-sum games
- Uses independent strategy mixing (Nash equilibrium)
- Doesn't capture coordination opportunities in general-sum games

### JPSRO Solution
- Handles n-player general-sum games
- Uses **joint distributions** over strategy profiles
- Computes Correlated Equilibrium (CE) or Coarse Correlated Equilibrium (CCE)
- Enables coordination beyond independent mixing

## Key Concepts

### Correlated Equilibrium (CE)
**Definition**: A mediator suggests a strategy profile. No player wants to deviate after seeing their recommended action.

**Formal**: Distribution $\mu$ over joint actions where for each player $i$ and each action $a_i$:
$$\sum_{a_{-i}} \mu(a_i, a_{-i}) u_i(a_i, a_{-i}) \geq \sum_{a_{-i}} \mu(a_i, a_{-i}) u_i(a'_i, a_{-i})$$
for all $a'_i$ (deviations).

**Example** (Traffic Light):
```
         Left    Right
Up       (5,5)   (0,0)
Down     (0,0)   (5,5)
```

**Nash**: Mix 50-50, get EU = 2.5 each
**CE**: Mediator flips coin, tells both to coordinate (Up,Left) or (Down,Right), EU = 5 each

### Coarse Correlated Equilibrium (CCE)
**Relaxation**: Players must commit to follow or deviate *before* seeing recommendation

**Formal**: No player wants to deviate from their marginal:
$$\mathbb{E}_\mu[u_i(a)] \geq \mathbb{E}_\mu[u_i(a'_i, a_{-i})]$$
for all deviations $a'_i$.

**Property**: Easier to compute than CE (linear program)

### Hierarchy
$$\text{Nash} \subseteq \text{CE} \subseteq \text{CCE}$$

**Implication**: CCE is broader solution concept, easier to find

## JPSRO Algorithm

### 1. Initialization
Start with initial population $\Pi_i^{(0)}$ for each player $i$

### 2. Meta-Game Construction
For current populations, construct empirical payoff table:
- Rows: Joint strategy profiles (all combinations)
- Values: Expected payoffs from simulation

### 3. Meta-Strategy Solver
Compute CCE (or CE) $\nu$ over joint policies

**Linear Program** (for CCE):
```
maximize    Σ ν(π) · u_i(π)  for all i (weighted sum)
subject to  ν(π) ≥ 0, Σ ν(π) = 1
            incentive constraints (no beneficial deviation)
```

**Common Objectives**:
- **Maximum Welfare (MW-CCE)**: Maximize sum of utilities
- **Maximum Gini (MG-CCE / MGCE)**: Maximize Gini impurity (recommended) - see [[MGCE]]
- **Maximum Entropy (MECE)**: Maximize Shannon entropy (alternative)
- **Random Vertex (RV-CCE)**: Sample vertex of CCE polytope

### 4. Best Response Computation
**Key Difference from PSRO**: Train against **marginal distribution** $\nu_i$

For each player $i$:
- Sample co-player strategies from $\nu_{-i}$ (joint marginal)
- Train BR policy $\pi_i^{new}$ using RL (PPO, SAC, etc.)

**Why Joint?**: Captures correlations between co-players

### 5. Population Update
Add new BRs to populations: $\Pi_i^{(k+1)} = \Pi_i^{(k)} \cup \{\pi_i^{new}\}$

### 6. Convergence Check
- Compute exploitability: max gain from deviating
- Stop when exploitability below threshold

## Comparison: PSRO vs JPSRO

| Aspect | PSRO | JPSRO |
|--------|------|-------|
| Players | 2 (mainly) | n (any) |
| Game type | Zero-sum | General-sum |
| Equilibrium | Nash | CE / CCE |
| Strategy mixing | Independent | Joint distribution |
| Coordination | No | Yes |
| BR per iteration | 1 per player | 1 per player (CE needs more) |
| Complexity | Medium | Higher |

## Computational Considerations

### CCE vs CE
- **CCE**: 1 BR per player per iteration
- **CE**: $|A_i|$ BRs per player (BR for each recommendation)
- **Tradeoff**: CE is stronger concept but computationally expensive

### Scalability
- Population grows linearly with iterations
- Meta-game size: $O(\prod_i |\Pi_i|)$ (exponential in players)
- LP solver: Efficient for moderate populations (~100 policies)

## Applications

### Multi-Agent Coordination
- Cooperative tasks (MuJoCo multi-agent)
- Mixed-motive games (Prisoner's Dilemma variants)
- Team games (Capture-the-Flag)

### General-Sum Games
- Partially cooperative/competitive
- Coordination dilemmas
- Social dilemmas

## NeuPL-JPSRO
**Combines** JPSRO with Neural Population Learning:
- Single conditional network for all policies
- CCE computation on neural population
- O(1) memory instead of O(N) policies

See [[NeuPL]] for details.

## Interview Relevance

**Common Questions**:
1. **PSRO vs JPSRO?** PSRO: Nash, 2-player zero-sum; JPSRO: CCE, n-player general-sum
2. **What's Correlated Equilibrium?** Mediator suggests actions, no incentive to deviate
3. **CCE vs CE?** CCE: commit before seeing recommendation; CE: react after seeing
4. **Why joint distribution?** Captures coordination opportunities beyond independent mixing
5. **Computational cost?** LP for meta-game (efficient); BR training (expensive, parallelizable)
6. **When use JPSRO?** General-sum games where coordination matters
7. **Convergence guarantee?** Yes, to CCE under exact BR
8. **Example where CE > Nash?** Traffic light game: CE = 5, Nash = 2.5

**Key Formulas**:
- CCE incentive: $\mathbb{E}_\mu[u_i(a)] \geq \mathbb{E}_\mu[u_i(a'_i, a_{-i})]$ for all $a'_i$
- Hierarchy: Nash ⊆ CE ⊆ CCE

**Key Insight**: JPSRO enables solving general-sum games by using joint strategy distributions (CE/CCE) that allow coordination beyond what independent mixing (Nash) can achieve.
