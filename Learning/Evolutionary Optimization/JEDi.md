# JEDi: Quality with Just Enough Diversity

## Overview

**JEDi** (Quality with Just Enough Diversity) bridges Quality-Diversity (QD) and Evolution Strategies (ES) by using Gaussian Processes to learn the behavior-fitness relationship, then intelligently targeting promising behaviors with parallel ES emitters. This provides QD's exploration benefits without wasting evaluation budget on exhaustive illumination.

**Core Innovation**: Instead of exploring all behaviors equally (MAP-Elites) or ignoring behaviors entirely (pure ES), JEDi learns which behaviors lead to high fitness and focuses exploration there.

---

## The Problem JEDi Solves

### Quality-Diversity Dilemma

**QD Methods** (e.g., MAP-Elites):
- ✅ Extensive behavior space coverage
- ✅ Discover stepping stones through diversity
- ❌ Waste evaluation budget on non-best individuals
- ❌ Slower convergence to high fitness

**Evolution Strategies** (e.g., CMA-ES):
- ✅ Fast convergence to local optimum
- ✅ Efficient use of evaluation budget
- ❌ Get stuck in local optima (no diversity)
- ❌ Fail on hard exploration problems (deceptive fitness landscapes)

**Example**: In maze navigation, ES finds nearest wall and stops (local optimum). MAP-Elites explores all paths but slowly. JEDi learns "paths near exits have high fitness" and focuses ES there.

---

## Key Concepts

### 1. Behavior-Fitness Relationship

**Definition**: The mapping from behavior descriptors (how a policy acts) to fitness values (how well it performs).

**Example** (Robot locomotion):
- Behavior: (forward_distance, sideways_distance)
- Fitness: energy_efficiency
- Relationship: Straight-line forward locomotion often has highest efficiency

**Why It Matters**: If certain behaviors correlate with high fitness, we should focus exploration there rather than uniformly sampling all behaviors.

### 2. Gaussian Process (GP) Surrogate Model

**Purpose**: Learn smooth approximation of fitness landscape from sparse archive data.

**GP Prediction**:
```
At unexplored behavior b:
  Predicted fitness: μ(b) (mean)
  Prediction uncertainty: σ(b) (standard deviation)
```

**Benefits**:
- Interpolates between known points
- Provides principled uncertainty estimates
- Enables intelligent exploration-exploitation tradeoff

**Weighted GP Innovation**: Accounts for evaluation budget per cell. Cells with fewer evaluations have higher uncertainty, even with full archive coverage.

### 3. Just Enough Diversity

**Philosophy**: Don't seek diversity for its own sake. Use diversity only to improve maximum fitness.

**Contrast**:
- **MAP-Elites**: Goal = uniformly fill behavior space
- **JEDi**: Goal = maximize fitness, use behaviors as stepping stones

**Spectrum**:
```
Pure Optimization (ES) ← JEDi → Pure Diversity (MAP-Elites)
```

**α Parameter**: Controls position on spectrum
- `α = 0`: Pure fitness maximization
- `α = 0.5`: Balanced
- `α = 1`: Pure behavior targeting

---

## Algorithm Overview

### Main Loop

```
1. Initialize repertoire with random policies
2. FOR each JEDi loop:
    a. Train Weighted GP on repertoire (learn behavior→fitness)
    b. Select target behaviors from Pareto front
    c. FOR each target:
        - Initialize ES at nearest policy in repertoire
        - Run ES with WTFS scoring (balances fitness + target)
        - Add all ES samples to repertoire
    d. Decay α (shift from exploration to exploitation)
```

### Key Components

**1. Target Selection (Pareto Front)**

Maximize two objectives:
- **Mean fitness** μ(b): Expected fitness at behavior b
- **Uncertainty** σ(b): Exploration value of behavior b

Select targets from Pareto front of (μ, σ).

**Intuition**: Some targets have high predicted fitness (exploit), others have high uncertainty (explore). Pareto front balances both.

**2. WTFS Scoring (Weighted Target-Fitness Score)**

```
Score = α · target_score + (1-α) · fitness_score

Where:
  target_score = 1 - distance_to_target (normalized)
  fitness_score = fitness (normalized)
```

**α Annealing**:
- Early: High α → focus on reaching targets (exploration)
- Late: Low α → focus on maximizing fitness (exploitation)

**3. Parallel ES Emitters**

Run multiple ES instances simultaneously toward different targets (typically 4).

**Benefits**:
- Parallelization (4× speedup)
- Diverse exploration across multiple promising regions
- Redundancy (if one gets stuck, others may succeed)

---

## Experimental Results

### Hard Exploration (Maze Navigation)

**Task**: Navigate robot through maze to goal.

**Results (Final Max Fitness)**:
| Method | Maze A | Maze B |
|--------|--------|--------|
| JEDi | -117 | -176 |
| ES | -280 | -320 |
| MAP-Elites | -136 | -202 |
| CMA-ME | -138 | -190 |

**Key Finding**: JEDi outperforms pure ES by 58% (Maze A) due to behavior-guided exploration escaping local optima.

### Complex Robot Control (Brax)

**Task**: Locomotion policies for simulated robots.

**Results (Final Max Fitness)**:
| Method | Walker2D | HalfCheetah |
|--------|----------|-------------|
| JEDi | 4353 | 3512 |
| ES | 3967 | 3420 |
| CMA-ME | 2159 | 2840 |

**Key Finding**: JEDi reaches 2× higher fitness than CMA-ME on Walker2D, showing efficiency gains on complex control tasks.

### Coverage Trade-off

**Observation**: JEDi has lower behavior coverage than MAP-Elites (by design).

**Interpretation**: JEDi focuses on promising behaviors, not exhaustive illumination. Appropriate for optimization-focused applications where diversity is a means, not an end.

---

## When to Use JEDi

### ✅ Good Fit

**1. Hard Exploration Problems**
- Deceptive fitness landscapes
- Local optima far from global optimum
- Example: Maze navigation, sparse reward RL

**2. High-Dimensional Policy Spaces**
- Large neural network policies
- Where MAP-Elites is slow to converge
- Example: Robot control with deep networks

**3. Optimization-Focused Applications**
- Goal: Maximize single objective
- Diversity as means, not end
- Example: Game AI, engineering design optimization

### ❌ Poor Fit

**1. When Diversity Itself is Valuable**
- Generative design (want many solutions)
- Portfolio of strategies
- Better suited for MAP-Elites

**2. When Behavior-Fitness Relationship is Weak**
- No correlation between behaviors and fitness
- Better suited for pure ES

**3. When Evaluation Budget is Unlimited**
- MAP-Elites eventually reaches same fitness
- JEDi's advantage is sample efficiency

---

## Key Innovations

### 1. Weighted Gaussian Process

**Problem**: Standard GP treats all cells equally. When archive is full, uncertainty is uniformly low everywhere.

**Solution**: Weight by inverse of evaluation count per cell:
```
Weight_i = 1 / num_evaluations_i
```

**Benefit**: Reflects true confidence in each region. Under-explored cells maintain high uncertainty.

### 2. Pareto Front Target Selection

**Problem**: GP-UCB (μ + β·σ) clusters targets in high-UCB regions.

**Solution**: Explicitly optimize μ and σ as separate objectives, sample from Pareto front.

**Benefit**: Ensures diversity of targets without clustering, simpler than GP-BUBC hallucination.

### 3. WTFS with α Annealing

**Problem**: Hard constraints (ARIA) are brittle and don't allow smooth exploration-exploitation transitions.

**Solution**: Soft weighted score with annealing schedule:
```
α: 1.0 → 0.0 over JEDi loops
```

**Benefit**: Early phase explores via behavior targets, late phase exploits for fitness. Smooth, robust transitions.

---

## Comparison to Related Methods

| Method | Behavior-Aware? | Target Selection | Budget Allocation |
|--------|-----------------|------------------|-------------------|
| **MAP-Elites** | Yes | Random mutations | Uniform across behaviors |
| **CMA-ES** | No | N/A | Focused (single mode) |
| **CMA-ME** | Yes | Fitness-based | Gradient-based (all cells) |
| **PGA-ME** | Yes | Fitness + diversity | Prediction-guided |
| **JEDi** | Yes | GP-informed Pareto | Intelligent (promising behaviors) |

**JEDi's Unique Position**: Uses learned behavior-fitness relationship to guide ES, not just to organize archive.

---

## Interview Relevance

### For Research Scientists

**Common Questions**:
- "How does JEDi differ from CMA-ME?"
  - CMA-ME uses ES to improve entire archive; JEDi targets specific behaviors predicted to have high fitness
- "Why Gaussian Processes for the surrogate?"
  - Provides both prediction (mean) and uncertainty (variance) for exploration-exploitation
- "What is the α parameter's purpose?"
  - Balances behavior-targeting vs. fitness-maximization; annealed for dynamic phase transitions

**Discussion Topics**:
- QQ (quality-quality) vs. QD (quality-diversity) spectrum
- Surrogate modeling for evolutionary optimization
- Exploration-exploitation tradeoffs in high-dimensional spaces
- When diversity is instrumental vs. intrinsic

### For ML Engineers

**Practical Considerations**:
- Framework: Implemented in QDax (JAX-based)
- ES: Sep-CMA-ES (small genomes) or LM-MA-ES (large genomes)
- Parallelization: 4 ES emitters, batch size = num_ES × λ
- Tuning: α parameter most critical (task-dependent)

**Applications**:
- Neural architecture search (NAS) with behavioral constraints
- Robotics with safety or style requirements
- Game AI with diverse play styles but winning focus
- Engineering optimization with multiple performance metrics

---

## Connection to Other Topics

- **[[Quality_Diversity]]**: JEDi uses QD principles but focuses on optimization
- **[[MAP_Elites]]**: Foundation algorithm JEDi extends with intelligent targeting
- **[[Evolution_Strategies]]**: JEDi uses ES for local optimization around targets
- **[[Gaussian_Processes]]**: Surrogate model for behavior-fitness landscape
- **[[Pareto_Optimization]]**: Used for target selection (multi-objective)
- **[[CMA_ME]]**: Related QD+ES hybrid, but different target selection strategy

---

## Quick Summary

**What**: QD-ES hybrid that learns behavior-fitness relationship with GP, targets promising behaviors with parallel ES

**Why**: Combines QD's exploration benefits with ES's optimization efficiency

**How**:
1. Learn fitness landscape with Weighted GP
2. Select targets from Pareto front (mean, variance)
3. Run parallel ES toward targets with WTFS scoring
4. Anneal α from exploration to exploitation

**Key Result**: Outperforms pure ES on hard exploration (58% in mazes), outperforms CMA-ME on complex control (2× on Walker2D)

---

**See [[JEDi_detailed]] for implementation details, GP formulation, WTFS equations, and hyperparameters.**
