# Extract-QD Framework (EQD Framework)

## Overview

The **Extract-QD Framework (EQD Framework)** is a modular framework that unifies existing Uncertain Quality-Diversity (UQD) approaches and enables the design of new UQD algorithms for noisy, stochastic, or uncertain domains.

**Authors**: Flageat, Huber, Helenon, Doncieux, Cully (Imperial College London, Sorbonne Université)
**Paper**: "Extract-QD Framework: A Generic Approach for Quality-Diversity in Noisy, Stochastic or Uncertain Domains"
**Venue**: GECCO 2025

## The Uncertain QD Problem

### Standard QD Assumption

Traditional QD algorithms (MAP-Elites, POET, etc.) assume **deterministic evaluations**:
- Single evaluation reliably estimates fitness $f_i$ and descriptor $d_i$
- Elitism works: Keep best in each cell

### Reality: Uncertain Domains

Many real-world tasks involve **uncertainty**:
- **Noisy sensors**: Robotics with sensor errors
- **Stochastic dynamics**: Random initialization, environment variability
- **Inherent task noise**: Multi-agent interactions, procedural generation

**Mathematical formulation**:
- Fitness is a distribution: $f_i \sim \mathcal{D}_{f_i}$
- Descriptor is a distribution: $d_i \sim \mathcal{D}_{d_i}$

### Problems in UQD

Three main challenges identified:

#### 1. Performance Estimation
*"How to accurately estimate expected fitness/descriptors with limited samples?"*

**Issue**: Elitism favors "lucky" solutions
- Solution evaluated once, gets outlier high fitness
- Replaces truly good solution
- Archive fills with overestimated agents

**Naive solutions fail**:
- Single sample: Keeps lucky solutions
- Many samples per solution: Computationally intractable, hinders exploration

#### 2. Reproducibility Maximization
*"How to prioritize consistent solutions over non-reproducible ones?"*

**Reproducibility**: Ability to consistently reproduce descriptor across evaluations
- Tight descriptor distribution = high reproducibility
- Wide descriptor distribution = low reproducibility

**Why it matters**:
- Users need guarantees on solution behavior
- Reproducible solution that reaches target 90% of time > high-performance but unreliable

**Example**: Robot controller
- High fitness, low reproducibility: Fast but often falls
- Medium fitness, high reproducibility: Slower but reliable

#### 3. Performance-Reproducibility Trade-off
*"How to balance high fitness vs high reproducibility when they conflict?"*

**Conflict common in practice**:
- Aggressive strategies: High fitness, low reproducibility
- Conservative strategies: Lower fitness, higher reproducibility

**User preference problem**: Different applications need different trade-offs

## Extract-QD Framework

### Extension of QD Framework

**Original QD Framework** (Mouret & Clune, 2015):
- Container: Organizes solutions (e.g., grid in MAP-Elites)
- Selection Operator: Chooses parents for variation

**EQD Framework adds**:
- Extraction Operator: Re-evaluates elites from archive
- Container with Depth: Maintains $d$ promising elites per cell
- Depth-Ordering Operator: Ranks elites within depth
- Samples: Number of first evaluations per solution

### Framework Modules

#### 1. Selection Operator
Selects parents to mutate offspring.

**Options**:
- Uniform: Random selection from archive
- Fitness-proportional: Weighted by fitness
- Tournament selection
- Biased wheel

#### 2. Variation Operator
Applies mutations to generate offspring.

**Options**:
- Random mutations: Gaussian, polynomial
- Gradient-based mutations: PG-ME, PGA-ME
- Evolution strategies
- Crossover + mutation

#### 3. Extraction Operator (NEW)
Re-evaluates elites from archive for better estimation.

**Extraction types**:
- **Random**: Uniform sampling from archive
- **Deterministic**: Fixed pattern (e.g., all elites)
- **Adaptive**: Based on performance/uncertainty

**Key innovation**: "Extracts" solutions (removes from archive) to allow drift to correct cell

#### 4. Container with Depth (NEW)
Maintains $d$ promising elites per cell instead of just 1.

**Depth $d$ functions**:
- **Memory buffer**: Keep backup elites when top elite drifts
- **Richer selection**: More parent diversity
- **Better estimation**: Track multiple candidates per cell

**Special cases**:
- $d=1$: Standard QD (no depth)
- $d>1$: UQD with memory

#### 5. Depth-Ordering Operator (NEW)
Ranks the $d$ elites within each cell.

**Ordering criteria**:
- **Fitness only**: Keep $d$ highest-fitness solutions
- **Reproducibility only**: Keep $d$ most reproducible
- **Weighted combination**: $\alpha \cdot \text{fitness} + (1-\alpha) \cdot \text{reproducibility}$
- **Pareto dominance**: Non-dominated set
- **Seniority + fitness**: Latest solutions, ordered by fitness

**Top layer**: Elite returned in final archive (but full depth used during optimization)

#### 6. Samples Parameter
Number of evaluations for first assessment of solution.

**Trade-off**:
- More samples: Better initial estimate, slower exploration
- Fewer samples: Faster exploration, may need re-evaluation

### How Modules Combine

```
Generation loop:
  1. Selection Operator → choose parents from Container
  2. Variation Operator → mutate to create offspring
  3. Evaluate offspring with N Samples
  4. Extraction Operator → extract elites for re-evaluation
  5. Re-evaluate extracted elites (accumulate in buffers)
  6. Add offspring + re-evaluated elites to Container (with Depth)
  7. Depth-Ordering Operator → rank elites within each cell
```

## Extract-ME (EME)

**Extract-ME** is the flagship instantiation of the framework, demonstrating strong performance across diverse UQD tasks.

### Algorithm

```
Parameters:
  - Depth d = 8
  - Extraction budget: 25% of evaluations
  - Initial samples: N = 2
  - Extraction: Exponentially-weighted random

For generation = 1 to max_gen:

  # Determine evaluation budget split
  n_offspring = 0.75 * budget_per_gen
  n_extract = 0.25 * budget_per_gen

  # Generate and evaluate offspring
  for i = 1 to n_offspring:
    parent = SelectRandom(archive)
    offspring = Mutate(parent)
    f, d = Evaluate(offspring, n_samples=2)
    offspring.buffer = [(f, d)]  # Initialize buffer
    archive.Add(offspring, f, d)

  # Extract and re-evaluate elites
  for i = 1 to n_extract:
    # Sample elite with exponential probability by rank
    elite = ExtractExponential(archive)
    archive.Remove(elite)

    # Re-evaluate
    f, d = Evaluate(elite, n_samples=1)
    elite.buffer.append((f, d))  # Accumulate samples

    # Estimate from all samples
    f_est = median(elite.buffer.f)
    d_est = median(elite.buffer.d)

    # Add back to archive (may drift to different cell)
    archive.Add(elite, f_est, d_est)
```

### Key Features

#### Extraction Budget: 25%
**Rationale**:
- Balance exploration (new offspring) vs exploitation (better estimates)
- 75% new offspring: Maintain exploration rate
- 25% re-evaluation: Improve archive quality

**Adaptive**: Can adjust based on task characteristics

#### Depth d=8
**Why 8 elites per cell?**
- Enough backup when elites drift
- Not too large (memory, computation)
- Works well empirically across tasks

#### Exponential Extraction Probability
**Formula**: $P(\text{extract elite } i) \propto \exp(-\lambda \cdot \text{rank}_i)$

Where $\text{rank}_i \in [0, d-1]$ is elite's rank in its cell (0 = best).

**Effect**:
- Higher-ranked elites more likely to be extracted
- Exponential (not linear) allows larger depth while maintaining pressure
- Still gives lower-ranked elites some chance

**Example** ($d=8$, $\lambda=1$):
- Rank 0 (best): 54% probability
- Rank 1: 20% probability
- Rank 2: 7% probability
- Rank 7 (worst): 0.05% probability

#### Performance Estimation via Buffers
**Buffer**: Store all evaluations of a solution (initial + re-evaluations)

**Estimation**:
- Fitness: $\hat{f} = \text{median}(\text{buffer}_f)$
- Descriptor: $\hat{d} = \text{median}(\text{buffer}_d)$

**Advantages**:
- Robust to outliers (median vs mean)
- Improves over time (more samples)
- Detects drift (descriptor changes)

### Advantages over Existing Methods

| Method | Issue | EME Solution |
|--------|-------|--------------|
| **ME-Sampling** | Fixed 32 samples: High cost, hinders exploration | Adaptive re-evaluation (25% budget), 2 initial samples |
| **Archive-Sampling (AS)** | Must re-evaluate entire archive every generation | Extract only subset, scales to large archives |
| **Adapt-ME** | Sequential evaluation (not parallelizable) | Fully parallelizable, fixed budget per generation |
| **Deep-Grid** | Struggles with descriptor noise, assumes fitness fitness | Extraction + depth handle both noise types |

### Comparison Table

| Method | Depth | Extraction | Samples | Parallelizable | Scales to Large Archives |
|--------|-------|------------|---------|----------------|--------------------------|
| **ME** | 1 | None | 1 | ✓ | ✓ |
| **ME-Sampling** | 1 | None | 32 | ✓ | ✓ |
| **Deep-Grid** | 32 | None | 1 | ✓ | ✓ |
| **AS** | 2 | All | 2 | ✓ | ✗ (requires small archive) |
| **Adapt-ME** | 8 | Challenged elites | 1+ | ✗ (sequential) | △ |
| **EME** | 8 | Random 25% | 2 | ✓ | ✓ |

## Experimental Results

### Task Suite

Four robotics domains with different uncertainty types:

#### 1. Arm (8D control)
**Two variants**:
- **Arm Fitness Noise**: $\mathcal{N}(0, 0.1)$ on fitness
- **Arm Desc Noise**: $\mathcal{N}(0, 0.01)$ on descriptor

**Controller**: Direct joint positions
**Fitness**: Negative joint variance (stability)
**Descriptor**: End-effector position (2D)
**Grid**: 16×16 = 256 cells

#### 2. Hexapod (24D control)
**Uncertainty**: $\mathcal{N}(0, 0.05)$ on both fitness and descriptor

**Controller**: Periodic functions (amplitude + phase)
**Fitness**: Direction error
**Descriptor**: Final position (2D)
**Grid**: 32×32 = 1024 cells

#### 3. Walker (270D neural network)
**Uncertainty**: Random initial joint positions/velocities $\mathcal{U}(-0.05, 0.05)$

**Controller**: Neural network (hidden layer size 8)
**Fitness**: Speed - energy + survival
**Descriptor**: Feet contact pattern (2D)
**Grid**: 16×16 = 256 cells

#### 4. Ant (296D neural network)
**Uncertainty**: Random initial joint positions/velocities $\mathcal{U}(-0.1, 0.1)$

**Controller**: Neural network (hidden layer size 8)
**Fitness**: -Energy + survival
**Descriptor**: Final position (2D)
**Grid**: 32×32 = 1024 cells

### Baseline Methods

1. **Vanilla-ME**: Single evaluation, no uncertainty handling
2. **ME-Sampling**: 32 samples per solution
3. **Adapt-ME**: Adaptive re-evaluation of challenged elites
4. **AS**: Re-evaluate entire archive each generation
5. **Deep-Grid**: Depth 32, fitness-proportional selection, seniority ordering

### Key Results

**Corrected QD-Score** (sum of fitness in corrected archive):

| Task | EME | AS | Adapt-ME | Deep-Grid | ME-Sampling | Vanilla-ME |
|------|-----|----|----|-----------|-------------|------------|
| **Arm Fitness** | **Best** | Best | Med | Low | Med | Low |
| **Arm Desc** | **Best** | Best | Low | Low | Med | Low |
| **Hexapod** | **Best** | N/A | Med | Med | Med | Low |
| **Walker** | **Best** | Best | Low | Med | Med | Low |
| **Ant** | **Best** (p<0.001) | N/A | Low | Low | Med | Low |

**Key findings**:
1. **EME** matches or beats best method on every task
2. **AS** strong when applicable, but undefined for Hexapod/Ant (archive too large)
3. **Adapt-ME** struggles with high-dimensional tasks (too many re-evaluations)
4. **Deep-Grid** fails with descriptor noise
5. **ME-Sampling** decent but exploration hampered by high sampling cost

### Average Samples

Number of samples used by top-layer elites:

| Task | EME | AS | Interpretation |
|------|-----|----|----------------|
| Arm Desc | **12.3** | 8.1 | EME uses MORE samples on critical elites |
| Walker | **8.7** | 6.2 | Better allocation than uniform re-evaluation |

**Insight**: Exponential extraction wisely allocates samples to promising elites

## Extension: Extract-PGA (EPGA)

### QD-RL Integration

**PGA-ME** (Policy-Gradient-Assisted MAP-Elites):
- Variation operator: RL-based (TD3) instead of random mutation
- Strong performance in RL environments
- But doesn't handle uncertainty

**EPGA = EME + PGA**:
- Swap variation operator to use PGA's TD3-based evolution
- Keep EME's extraction, depth, re-evaluation
- Apply to uncertain RL tasks

### Algorithm

```
Same as EME, but:
  Variation Operator = TD3-based policy gradient instead of Gaussian mutation
```

### QD-Gym Results

**Tasks**:
1. Walker2D (feet contact descriptor)
2. HalfCheetah (feet contact descriptor)
3. Ant (4D feet contact descriptor)

**Baselines**:
- ME: Standard MAP-Elites
- PGA: PGA-ME without uncertainty handling
- EPGA: Extract-PGA (proposed)

**Results** (Corrected QD-Score):

All tasks: **EPGA > PGA > ME** (p < 0.005)

**Key insight**: Accounting for uncertainty improves PGA at **no additional evaluation cost**
- Same total evaluations (1.5M)
- Same per-generation budget (128)
- EPGA: 96 new + 32 re-eval
- PGA: 128 new + 0 re-eval

## Framework Generality

### Encompasses Existing Methods

The framework can express **all prior UQD methods** by choosing appropriate modules:

| Method | Container | Depth | Extraction | Ordering | Samples |
|--------|-----------|-------|------------|----------|---------|
| **ME** | Grid | 1 | None (0) | Fitness | 1 |
| **ME-Sampling** | Grid | 1 | None | Fitness | 32 |
| **Deep-Grid** | Grid | 32 | None | Seniority+Fitness | 1 |
| **AS** | Grid | 2 | Full ($Cd$) | Fitness | 2 |
| **Adapt-ME** | Grid+Buffer | 8 | Challenged ($\leq b+bd$) | Fitness | 1+ |
| **ME-Weighted** | Grid | 1 | None | Fitness+Reprod weighted | 32 |
| **MOME-X** | Grid of Pareto | 1 | None | Pareto (f, reprod) | 32 |
| **EME** | Grid | 8 | Random (25%) | Fitness | 2 |

### Designing New Methods

**Framework as toolbox**: Swap modules for task-specific algorithms

**Example design decisions**:

**If distribution of noise is known**:
- Custom depth-ordering: Weight by confidence intervals
- Extraction: Prioritize high-uncertainty solutions

**If genotype→phenotype mapping is smooth**:
- Use neighbor approximations for fitness/descriptor
- Reduce evaluation budget

**If want reproducibility**:
- Depth-ordering: Weighted (fitness, reproducibility)
- Examples: ME-Weighted, AS-Weighted, MOME-X

**If RL domain**:
- Variation: Gradient-based (PGA, CMA-ES)
- Example: EPGA

## Practical Recommendations

### When to Use EME

**Good fit**:
- Moderate-to-large archives (AS doesn't scale)
- Need parallelization (Adapt-ME too slow)
- Unknown noise characteristics (ME-Sampling wasteful)
- Evaluation budget matters

**EME as "first guess"**:
- Solid performance across diverse tasks
- Few hyperparameters to tune
- Computational efficient

### When to Customize

**Use framework to build tailored method**:
- Domain knowledge about noise structure
- Specific reproducibility requirements
- Unusual evaluation constraints

### Hyperparameter Tuning

**Extraction budget**:
- Start with 25%
- Increase if archive quality matters more than exploration
- Decrease if exploration is bottleneck

**Depth**:
- $d=8$ works well generally
- Increase if high drift rates
- Decrease if memory/computation limited

**Initial samples**:
- $N=2$ balances cost and quality
- Increase if noise very high
- $N=1$ if evaluations very expensive

## Limitations

### Current Scope

**Focused on Performance Estimation**:
- Main results for Problem 1 (performance estimation)
- Framework handles Problems 2 & 3 (reproducibility, trade-off)
- But detailed study of reproducibility methods left to future work

**Proof-of-concept for all three**:
- Table shows framework encompasses reproducibility-aware methods
- MOME-X, ME-Weighted, AS-Weighted all expressible
- But no new contributions for these problems

### Computational Cost

**Tournament metric computation**:
- Corrected archive: 512 re-evaluations per solution
- Expensive for large archives
- Necessary for ground-truth evaluation

### Task Limitations

**All tasks are robotic control**:
- Would benefit from testing on other UQD domains
- E.g., procedural generation, multi-agent, protein design

## Future Directions

1. **Reproducibility-focused variants**: Detailed study of Problems 2 & 3
2. **Theoretical analysis**: Convergence properties, sample complexity
3. **Adaptive extraction**: Learn which elites to re-evaluate
4. **Online budget allocation**: Dynamically split between offspring and extraction
5. **Multi-objective UQD**: Pareto fronts in uncertain domains

## Key Takeaways

1. **Modular framework** unifies all existing UQD methods under common view
2. **Three UQD problems**: Performance estimation, reproducibility, trade-off
3. **EME** as strong general-purpose method (25% extraction, depth 8, exponential sampling)
4. **Consistently outperforms** or matches best method on each task
5. **Extraction** enables adaptive re-evaluation while maintaining parallelizability
6. **Framework as toolbox** for designing task-specific UQD algorithms
7. **RL integration** (EPGA) shows framework extends beyond random mutations

## References

- Paper: Flageat et al., "Extract-QD Framework: A Generic Approach for Quality-Diversity in Noisy, Stochastic or Uncertain Domains", GECCO 2025
- Implementation: QDax library
- Related: [[MAP-Elites]], [[DNS]], [[POET]]

## Related

- [[Extract_QD_detailed]] — Implementation details
- [[MAP-Elites]] — Base QD algorithm
- [[Quality_Diversity]] — General QD concepts
- [[Uncertain_QD]] — UQD problem formulation
