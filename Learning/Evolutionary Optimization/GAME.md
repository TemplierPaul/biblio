# Generational Adversarial MAP-Elites (GAME)

## Overview

**GAME (Generational Adversarial MAP-Elites)** is a coevolutionary Quality-Diversity algorithm that illuminates **both sides** of adversarial problems by alternating between generations using Multi-Task Multi-Behavior MAP-Elites (MTMB-ME) with tournament-informed task selection.

**Authors**: Anne, Syrkis, Elhosni, Turati, Manai, Legendre, Jaquier, Risi (IT University of Copenhagen & armasuisse)
**Paper**: "Tournament Informed Adversarial Quality Diversity"
**Venue**: GECCO 2026

## Core Problem

### Adversarial QD Challenges

Traditional QD algorithms (MAP-Elites, etc.) illuminate **one side** of a problem while fixing the other:
- Red teaming: Generate attacks, fix defender
- Game balancing: Optimize strategies, fix opponent set
- **Limitation**: Partial illumination against arbitrary opponent choice

**GAME solution**: Illuminate **both sides simultaneously** through coevolution

### Adversarial QD Problem Definition

Tuple $(\mathcal{S}_{\text{Red}}, \mathcal{S}_{\text{Blue}}, \mathcal{F}, \mathcal{B})$ where:

**Search spaces**: $\mathcal{S}_{\text{Red}}$, $\mathcal{S}_{\text{Blue}}$ (can be identical or different)

**Fitness function**: $\mathcal{F}: \mathcal{S}_{\text{Red}} \times \mathcal{S}_{\text{Blue}} \to [0,1]^2$
- $(s_{\text{red}}, s_{\text{blue}}) \mapsto (f_{\text{red}}, f_{\text{blue}})$
- **Adversarial constraint**: $f_{\text{red}} + f_{\text{blue}} = 1$ (zero-sum)

**Behavior descriptor**: $\mathcal{B}: \mathcal{S}_{\text{Red}} \times \mathcal{S}_{\text{Blue}} \to \mathbb{R}^m$
- Depends on **both** sides (not intrinsic to single solution)

**Key challenges**:
1. Optimizing both spaces simultaneously (fitness tradeoff)
2. No intrinsic behavior descriptor for a solution
3. Measuring quality and diversity in adversarial setting

## GAME Algorithm

### Core Idea

Alternate illumination of each side using MTMB-ME against **tasks** (fixed opponents from previous generation):

```
1. Initialize N_task random Blue solutions as tasks

2. For gen = 1 to N_gen:

   a. Determine side: Red if odd, Blue if even

   b. Initialize multi-task archive (N_cell per task)

   c. Run MTMB-ME for N_budget evaluations:
      - Select task at random
      - Generate offspring (variation from archive)
      - Evaluate against task
      - Update task's growing archive

   d. Select N_task new tasks from all elites (task selection)

   e. Bootstrap next generation with tournament evaluations
```

### Multi-Task Multi-Behavior MAP-Elites (MTMB-ME)

**Extension of MAP-Elites** for multi-task QD:
- Maintains separate archive for each task
- Each archive: N_cell growing CVT grid
- Solution evaluated against one task at a time
- Encourages finding diverse solutions per task

**Growing Archive** (CVT-like):
- Starts empty, grows to max N_cell
- New behavior creates new cell if far enough from existing
- Otherwise competes on fitness within nearest cell
- Drift repair: Backup elites restore holes

### Task Selection: The Key Innovation

**Critical question**: Which opponents to train against next generation?

#### Behavior Task Selection (Original)

```
1. Aggregate all elites' behaviors from all tasks
2. Run K-means clustering (k = N_task)
3. Select elite of each cluster as new task
4. Tournament: New tasks vs old tasks
```

**Issues**:
- Ignores adversarial aspect (behavior is task-dependent)
- May favor elites from "easier" tasks
- No exploitation information used

#### Ranking Task Selection (Proposed) ⭐

**Inspiration**: PATA-EC from Enhanced POET

```
1. Tournament: All elites vs all previous tasks
   → fitness matrix U[elite, task]

2. For each elite e:
   - Compute fitness vector: f_e = U[e, :]
   - Compute ranking vector: r_e = argsort(argsort(f_e))
   - Normalize: r_e = 2·r_e/(N_task-1) - 1  ∈ [-1, 1]^(N_task)

3. Cluster elites by ranking vectors (K-means, k = N_task)

4. Select elite of each cluster (using avg fitness as quality)
```

**Intuition**:
- Different challenges create different rankings
- Ranking vector = adversarial behavior descriptor
- Promotes diversity of challenges

#### Pareto Task Selection (Proposed)

**Multi-objective view**: Each task = one objective

```
1. Tournament: All elites vs all previous tasks
   → fitness vector per elite

2. Apply NSGA-III to select N_task elites from Pareto fronts
```

**Intuition**: Diverse trade-offs in multi-objective space

#### Random Baseline

Select N_task elites uniformly at random (no tournament needed)

### Comparison of Task Selection Methods

| Method | Tournament Cost | Uses Adversarial Info | Promotes |
|--------|----------------|----------------------|----------|
| **Behavior** | $N_{\text{task}}^2$ | No | Behavioral diversity |
| **Random** | $N_{\text{task}}^2$ | No | Exploration |
| **Ranking** | $N_{\text{task}}^2 \cdot N_{\text{cell}}$ | Yes ⭐ | Challenge diversity |
| **Pareto** | $N_{\text{task}}^2 \cdot N_{\text{cell}}$ | Yes ⭐ | Trade-off coverage |

## Adversarial QD Measures

Traditional QD measures (max fitness, coverage, QD-score) fail in adversarial settings. GAME proposes 6 measures computed from **inter-variant tournament**:

### Setup

Tournament between variants:
- Each variant's final generation of tasks
- Evaluate all vs all: $(N_{\text{task}} \cdot N_{\text{rep}} \cdot N_{\text{variant}})^2$ matchups
- Ensures fair comparison (same opponents)

### 1. Win Rate (Quality)

*"Did it find a solution that wins against most opponents?"*

$$\text{Win Rate}(S_{\text{red}}) = \max_{s_{\text{red}} \in S_{\text{red}}} \frac{1}{|\underline{S}_{\text{blue}}|} \sum_{s_{\text{blue}} \in \underline{S}_{\text{blue}}} \mathbb{1}_{f_{s_{\text{red}}, s_{\text{blue}}} > 0.5}$$

**Interpretation**: Best agent's win percentage

### 2. ELO Score (Robust Quality)

*"Did it find a solution that wins against strong opponents?"*

- Compute ELO score for all solutions from tournament
- Rank solutions by ELO
- Normalize ranks to [0%, 100%]
- Return highest rank

$$\text{ELO Score}(S_{\text{red}}) = \max_{s_{\text{red}} \in S_{\text{red}}} \text{ELO\_rank}(s_{\text{red}})$$

**Advantage**: Values beating strong opponents > beating weak opponents

### 3. Robustness (Worst-Case Quality)

*"Did it find a solution with no weakness?"*

$$\text{Robustness}(S_{\text{red}}) = \max_{s_{\text{red}} \in S_{\text{red}}} \min_{s_{\text{blue}} \in \underline{S}_{\text{blue}}} f_{s_{\text{red}}, s_{\text{blue}}}$$

**Interpretation**: Best worst-case performance (maximin)

### 4. Coverage (Adversarial Diversity)

*"Does the set propose a diverse set of challenges?"*

```
1. For each solution, compute normalized ranking vector (vs all opponents)
2. Cluster all solutions (K-means, k = N_task)
3. Count % of clusters filled by this variant's solutions
```

$$\text{Coverage}(S_{\text{red}}) = \frac{|\{C_{s_{\text{red}}}\}_{s_{\text{red}} \in S_{\text{red}}}|}{N_{\text{task}}}$$

**Interpretation**: Diversity of challenges posed

### 5. Expertise (Counter-Solution Quality)

*"Does the set contain counter-solutions?"*

$$\text{Expertise}(S_{\text{red}}) = \min_{s_{\text{blue}} \in \underline{S}_{\text{blue}}} \max_{s_{\text{red}} \in S_{\text{red}}} f_{s_{\text{red}}, s_{\text{blue}}}$$

**Interpretation**: Worst counter-solution quality (for each opponent, find best counter)

### 6. Adversarial QD-Score (Strict Diversity)

*"Does the set propose strictly different challenges?"*

Smallest set of counter-solutions needed to make each solution lose at least once:

$$\text{AQD-Score}(S_{\text{red}}) = \min_{S_{\text{blue}} \subseteq \underline{S}_{\text{blue}}} |S_{\text{blue}}|$$

subject to: $\forall s_{\text{red}} \in S_{\text{red}}, \exists s_{\text{blue}} \in S_{\text{blue}} : f_{s_{\text{red}}, s_{\text{blue}}} < 0.5$

**Interpretation**: Minimum opponents needed to beat all strategies

## Experimental Results

### Tested Domains

1. **Pong** (symmetric):
   - Two paddles, ball speed increases
   - MLP controllers (input: ball pos/vel, paddle pos)
   - Ties common (neither scores or both score same)

2. **Cat-and-Mouse** (asymmetric):
   - Cat: fast ($v=2$) but low turning radius
   - Mouse: slow ($v=1$) but agile
   - MLP controllers
   - Asymmetric win rates (~56% cat, ~87% mouse for Ranking)

3. **Pursuers-and-Evaders** (asymmetric):
   - 2 pursuers vs 2 evaders (shared MLPs)
   - Bounded arena with central obstacle
   - Asymmetric win rates (~48% pursuers, ~92% evaders for Ranking)

### Results Summary

**Ranking consistently best** across all measures:

| Domain | Method | Win Rate | ELO Score | Robustness | Coverage | Expertise | AQD-Score |
|--------|--------|----------|-----------|------------|----------|-----------|-----------|
| **Pong** | Ranking | **59.8%** | **99.3%** | **0.00** | 40% | **0.50** | **3** |
| | Pareto | **59.9%** | **99.3%** | **0.00** | **46%** | **0.50** | **3** |
| | Behavior | 53.8% | 92.9% | **0.00** | **54%** | **0.50** | 2 |
| | Random | 54.0% | 93.3% | **0.00** | **52%** | **0.50** | 2 |
| **Cat-Mouse** | Ranking | **56.7%** | **99.3%** | **0.40** | **58%** | **0.81** | **2** |
| | Pareto | **52.6%** | **98.7%** | 0.38 | 56% | 0.60 | **2** |
| | Behavior | 42.0% | 94.1% | 0.35 | 48% | 0.47 | 1 |
| | Random | 39.5% | 92.1% | 0.35 | 55% | 0.48 | 1 |
| **Pursuers-Evaders** | Ranking | **48.0%** | **99.5%** | **0.14** | 52% | **0.58** | **1.5** |
| | Pareto | 45.2% | 98.5% | 0.13 | 44% | 0.48 | 1 |
| | Behavior | 37.5% | 89.1% | **0.13** | **54%** | 0.49 | 1 |
| | Random | 41.5% | 95.7% | **0.13** | **55%** | 0.49 | 1 |

**Key findings**:
- **Ranking** dominates on quality metrics (Win Rate, ELO, Expertise, AQD-Score)
- **Behavior/Random** sometimes higher Coverage but lower quality
- Tournament-informed methods (Ranking, Pareto) >> non-tournament methods

### Statistical Significance

- Wilcoxon test with Holm-Bonferroni correction ($p < 0.05$)
- Bold values: not significantly different from best
- Ranking significantly better on most quality measures

## Implementation Details

### Hyperparameters

- **Generations**: $N_{\text{gen}} = 10$
- **Tasks per generation**: $N_{\text{task}} = 50$
- **Cells per task**: $N_{\text{cell}} = 20$
- **Evaluation budget**: $N_{\text{budget}} = 100,000$ per generation
- **Total evaluations**: $1.5$M (with tournament)

### Controllers

**Neural networks**:
- Architecture: MLP with hidden layers [32, 16]
- Activation: tanh
- Mutation: Gaussian noise (σ=0.1) to 30% of weights

### Behavior Space

**Visual Embedding Models (VEM)**:
- CLIP embeddings of game visualizations
- Automatically captures behavioral differences
- No manual feature engineering needed

### JAX Implementation

All environments implemented in JAX for:
- Parallel evaluation on GPU
- Fast simulation
- Vectorized operations

## Key Insights

### 1. Tournament Information is Critical

**Behavior selection** (no tournament info):
- Uses behavioral diversity only
- Ignores adversarial relationships
- Performs similar to **Random**

**Ranking/Pareto** (tournament-informed):
- Uses adversarial relationships
- Promotes challenge diversity
- Significantly outperforms

### 2. Measures Complement Each Other

Different measures capture different desiderata:
- **Win Rate**: Best overall
- **ELO Score**: Best against strong
- **Robustness**: No weaknesses
- **Coverage**: Challenge diversity
- **Expertise**: Counter-solutions exist
- **AQD-Score**: Strictly different

### 3. Domain Characteristics

**Pong**: Not open-ended
- Expertise = 0.5, Robustness = 0 for all
- Easy to create ties, hard to dominate

**Cat-Mouse**: Moderately open-ended
- Mouse advantages evident (87% win rate)
- Expertise > 0.5 for both sides

**Pursuers-Evaders**: Evader-favored
- Strong asymmetry (92% evader win rate)
- Lower expertise and robustness overall

## Limitations

1. **Evaluation cost**: Tournament requires $(N_{\text{task}} \cdot N_{\text{variant}})^2$ matchups
2. **Two-sided only**: Currently restricted to 2-player adversarial
3. **Zero-sum assumption**: $f_{\text{red}} + f_{\text{blue}} = 1$
4. **Openendedness**: Tested games not highly open-ended (low AQD-Scores)

## Future Directions

1. **Sample-efficient measures**: Approximate tournament metrics
2. **Multi-player**: Extend beyond 2 sides
3. **General-sum**: Relax zero-sum constraint
4. **Larger populations**: Scale to more tasks and cells
5. **Open-ended domains**: Test on highly non-transitive games

## Comparison to Related Work

| Method | Sides | Equilibrium | QD | Coevolution |
|--------|-------|-------------|-----|-------------|
| **PSRO/JPSRO** | 1-n | Nash/CE | No | Population |
| **Self-play** | 1 | Robust | No | Temporal |
| **POET/Enhanced POET** | 2 | None | Yes (agent) | Agent-Environment |
| **GAME** | 2 | None | Yes (both) | Adversarial |

**Unique contribution**: Only method that does QD on **both sides** of adversarial problem

## Key Takeaways

1. **Tournament-informed task selection** >> behavioral selection for adversarial QD
2. **Ranking method** consistently best across quality and diversity measures
3. **Six complementary measures** needed to fully evaluate adversarial QD
4. **PATA-EC concept** from POET generalizes to adversarial QD (ranking vectors)
5. **Inter-variant tournaments** ensure fair comparison despite side dependencies

## Related

- [[GAME_detailed]] — Implementation details
- [[MAP-Elites]] — Base QD algorithm
- [[Enhanced_POET]] — PATA-EC inspiration
- [[PSRO]] / [[JPSRO]] — Game-theoretic population learning
- [[Adversarial_Coevolution]] — Broader coevolution context

## References

- Paper: Anne et al., "Tournament Informed Adversarial Quality Diversity", GECCO 2026
- Code: https://github.com/Timothee-ANNE/GAME_tournament_informed
- Related: Enhanced POET (PATA-EC), MADRID (adversarial scenarios), Rainbow Teaming (red teaming)
