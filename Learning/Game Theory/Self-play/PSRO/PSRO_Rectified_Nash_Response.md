# PSRO with Rectified Nash Response (PSRO-rN)

## Paper
**Title**: Open-ended Learning in Symmetric Zero-sum Games
**Authors**: Balduzzi, Garnelo, Bachrach, Czarnecki, Perolat, Jaderberg, Graepel (DeepMind)
**Published**: ICML

## Core Problem: Nontransitivity

### Transitive vs Nontransitive Games
- **Transitive** (Chess, Go): Clear strength ordering → self-play works
- **Nontransitive** (Rock-Paper-Scissors): Strategic cycles → self-play fails

**Hodge Decomposition Theorem**: Any game = Transitive Component + Cyclic Component

## Gamescapes Framework

### Population Performance
Instead of individual agent strength, measure **population-level performance**:

$$v(\mathfrak{P}, \mathfrak{Q}) = \min_{q \in \mathfrak{Q}} \max_{p \in \mathfrak{P}} J(p, q)$$

**Intuition**: Worst-case quality of counter-solutions in population

### Gamescape Polytope
Geometric object encoding population interactions:
- Vertices = agents
- Expansion = population improvement
- Works even in nontransitive games

## Rectified Nash Response (PSRO-rN)

### Algorithm

```
At iteration t:
1. Evaluate population → payoff matrix U
2. Rectify: U_rect[i,j] = max(U[i,j], 0)    # Zero out losses
3. Compute Nash of rectified game: σ_rect
4. Train new agent against mixture with σ_rect
5. Add to population
```

### Key Insight: Game-Theoretic Niching

**Rectification effect**: "Play to your strengths, ignore your weaknesses"
- Agents specialize against specific opponents
- Creates protected strategic niches
- Population collectively covers more strategy space

### Comparison with Nash Response

| Method | Meta-strategy | Effect |
|--------|---------------|--------|
| **PSRO-N** (Nash) | Nash equilibrium | Safety, exploitation |
| **PSRO-rN** (Rectified Nash) | Rectified Nash | Diversity, niching |

## Effective Behavioral Diversity

**Measure**: $$d(\mathfrak{P}) = \text{volume}(\text{gamescape polytope})$$

**Properties**:
- Measures diversity of **effective behaviors** (not just policy differences)
- Ignores variations that don't affect outcomes
- Captures strategic variety

## Results

**Resource allocation games** (highly nontransitive):
1. **PSRO-rN**: Best population strength
2. PSRO-N: Second
3. Uniform sampling: Third
4. Self-play: Worst

**Key finding**: Rectification promotes strategic diversity → stronger populations

## When to Use

- **Self-play**: Transitive games (Chess, Go)
- **PSRO-N**: General games, need safety guarantees
- **PSRO-rN**: Nontransitive games, open-ended learning

## Related
- [[PSRO]] — Base algorithm
- [[PRD_Rectified_Nash]] — Different topic (meta-strategy solvers)
- [[NeuPL]] — Neural population learning
