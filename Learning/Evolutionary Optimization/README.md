# Evolutionary Optimization - Algorithm Reference

Quick reference for evolutionary algorithms, quality-diversity methods, and open-ended learning.

---

## Quality-Diversity (QD) Concept

**What**: Optimization paradigm seeking diverse collection of high-performing solutions (not just single optimum)
**Why**: Illuminates solution space, discovers stepping stones, enables transfer
**Key metrics**: QD-Score (sum of fitness), Coverage (niches filled)

---

## MAP-Elites
**Paper**: "Illuminating the Search Space by Mapping Elites" (Mouret & Clune, 2015)
**What**: Grid-based QD algorithm maintaining best solution per behavior cell
**How**: Discretize behavior space, keep elite per cell, mutate and compete locally
**When to use**: Low-D descriptor spaces (2-5D), known bounds, interpretable niches

---

## PT-ME (Parametric-Task MAP-Elites)
**Paper**: "Parametric-Task MAP-Elites" (Anne & Mouret, 2024)
**What**: Black-box algorithm for continuous multi-task optimization (parametric-task problem)
**How**: Sample new task each iteration, local linear regression operator, distill to neural network
**Key innovation**: Continuous task coverage + exploits local task structure via regression, outperforms PPO
**When to use**: Continuous task space (not finite), need solution for any task parameter
**Difference from MT-ME**: Continuous task sampling vs fixed discretization, adds linear regression operator

---

## MTMB-ME (Multi-Task Multi-Behavior MAP-Elites)
**Paper**: "Multi-Task Multi-Behavior MAP-Elites" (Anne & Mouret, 2023)
**What**: MAP-Elites variant finding many diverse solutions for many tasks simultaneously
**How**: Crossover between elites from different tasks, evaluate on random task, maintain task-specific archives
**Key innovation**: Leverages task similarity via crossover, 2× more solutions than baselines
**When to use**: Need diverse solutions for multiple related tasks, limited budget per task
**Difference from MAP-Elites**: Multiple tasks with shared optimization vs single task

---

## POET (Paired Open-Ended Trailblazer)
**Paper**: "Paired Open-Ended Trailblazer (POET)" (Wang et al., 2019)
**What**: Coevolutionary algorithm generating environments AND agents simultaneously
**How**: Mutate environments, optimize agents with ES, bidirectional transfer between pairs
**Key innovation**: Stepping stones emerge from parallel exploration, no predefined curriculum
**When to use**: Need automatic curriculum, domain allows environment parameterization

---

## Enhanced POET
**Paper**: "Enhanced POET: Open-Ended Reinforcement Learning" (Wang et al., 2020)
**What**: POET with domain-general metrics and unbounded environment generation
**How**: PATA-EC (performance-based novelty), CPPN environments, improved transfer
**Key innovation**: Domain-independent, theoretically unbounded via CPPNs
**Difference from POET**: No hand-designed features, richer environments, sustained innovation (60k+ iterations)

---

## JEDi (Quality with Just Enough Diversity)
**What**: QD-ES hybrid using learned behavior-fitness relationship
**How**: Gaussian Process predicts fitness landscape, ES optimizes toward promising behaviors
**Key innovation**: Weighted GP + Pareto target selection + WTFS scoring
**When to use**: Optimization-focused (not illumination), hard exploration, high-D policies
**Difference from MAP-Elites**: Intelligent targeting vs. uniform exploration

---

## OMNI-EPIC
**What**: LLM-driven open-ended learning generating executable environment code
**How**: Foundation models create task descriptions → Python code → RL training
**Key innovation**: Darwin Completeness goal (any computable environment), dual MoI, code synthesis
**Difference from POET**: Generates arbitrary code (not just parameters), uses LLMs
**When to use**: Need unlimited task generation, interpretable natural language tasks

---

## Digital Red Queen (DRQ)
**Paper**: "Digital Red Queen: Adversarial Program Evolution in Core War with LLMs" (Kumar et al., 2025)
**What**: LLM-guided adversarial self-play evolving assembly programs through Red Queen dynamics
**How**: Multi-round MAP-Elites, each round evolves warrior to defeat all previous opponents
**Key innovation**: Historical self-play produces convergent evolution toward general-purpose behavior
**When to use**: Adversarial domains needing robust generalists (not brittle specialists), LLM code evolution
**Difference from POET**: Adversarial self-play (not environment-agent), convergent evolution in behavior

---

## DNS (Dominated Novelty Search)
**What**: Parameter-free QD using dominated novelty score (distance to fitter solutions)
**How**: Competition fitness = mean distance to k-nearest better-performing solutions
**Key innovation**: No bounds/grids/thresholds needed, reformulates QD as fitness transformation
**When to use**: High-D (5D+), discontinuous spaces, unsupervised descriptors
**Difference from MAP-Elites**: Dynamic adaptation vs. fixed grid, scales to 1000D

---

## Soft-QD (SQUAD)
**Paper**: "Soft Quality-Diversity Optimization" (Hedayatian & Nikolaidis, USC)
**What**: Continuous QD formulation eliminating discrete archive discretization
**How**: Soft QD Score (integral of behavior values), SQUAD optimizes lower bound via attractive/repulsive forces
**Key innovation**: End-to-end differentiable, no curse of dimensionality from tessellation
**When to use**: High-D behavior spaces, differentiable domains (DQD), gradient-based optimization
**Difference from MAP-Elites**: Continuous formulation vs. discrete cells, gradient descent vs. mutations

---

## GAME (Generational Adversarial MAP-Elites)
**Paper**: "Tournament Informed Adversarial Quality Diversity" (Anne et al., 2026)
**What**: Coevolutionary QD illuminating BOTH sides of adversarial problems simultaneously
**How**: Alternating MTMB-ME generations with tournament-informed task selection (Ranking method)
**Key innovation**: Six adversarial QD measures, ranking vectors as adversarial behavior descriptors
**When to use**: Adversarial domains (red teaming, game balancing), need diverse challenges
**Difference from POET**: Adversarial coevolution vs. agent-environment, tournament-based selection

---

## Extract-QD Framework
**Paper**: "Extract-QD: A Generic Approach for QD in Noisy, Stochastic or Uncertain Domains" (Flageat et al., 2023)
**What**: Modular framework for Quality-Diversity under uncertainty (UQD)
**How**: Adaptive sampling via extraction budget (25%), archive depth tracking, exponential sampling
**Key innovation**: Unifies all UQD methods, 3 problem formulations, 6 algorithmic modules
**When to use**: Noisy evaluations, stochastic environments, reproducibility critical
**Difference from MAP-Elites**: Handles uncertainty via resampling, depth-based archives, Extract-ME algorithm

---

## QD-RL: Quality-Diversity with Reinforcement Learning

Folder: `QD-RL/` — Three progressive algorithms combining QD with Deep RL for efficient neuroevolution.

### PGA-ME (Policy Gradient Assisted MAP-Elites)
**Paper**: Nilsson & Cully (2021)
**What**: Hybrid variation operator (50% GA + 50% Policy Gradient) for neuroevolution
**How**: Evolutionary mutations + deterministic policy gradient using trained critic, actor injection
**Key innovation**: First successful hybrid QD-RL, 10× speedup over MAP-Elites
**When to use**: Unidirectional tasks, high-dimensional policies (10k+ parameters)
**Limitation**: Fails on omnidirectional (diversity collapses to single point)

### DCG-ME (Descriptor-Conditioned Gradients MAP-Elites)
**Paper**: Faldor et al. (2023)
**What**: Descriptor-conditioned critic Q(s,a|d) that guides mutations toward specific descriptors
**How**: Reward scaling by descriptor similarity S(d,d'), actor evaluation for negative samples, archive distillation
**Key innovation**: Fixes PGA-ME omnidirectional failure (+82% improvement), distills archive into single policy
**When to use**: Omnidirectional tasks, deceptive fitness landscapes, need archive distillation
**Advantage over PGA-ME**: Works on all task types, produces versatile policy

### DCRL-ME (Descriptor-Conditioned RL MAP-Elites)
**Paper**: Faldor et al. (2024)
**What**: Actor Injection mechanism transforms descriptor-conditioned actor into diverse specialized policies
**How**: Weight extraction (descriptor baked into bias), no actor evaluation needed, three variation operators
**Key innovation**: Eliminates actor evaluation overhead, 2.75× sample efficiency improvement over DCG-ME
**When to use**: Omnidirectional tasks with tight evaluation budget
**Advantage over DCG-ME**: Same benefits, 50% fewer evaluations

→ **See `QD-RL/README.md`** for detailed comparison, progression path, and guidance on choosing the right algorithm.

---

## Summary Table

| Algorithm | Type | Best For | Key Strength |
|-----------|------|----------|--------------|
| **MAP-Elites** | Grid-based QD | Low-D, known bounds | Interpretable illumination |
| **PT-ME** | Continuous multi-task | Continuous task space | Local regression, NN distillation |
| **MTMB-ME** | Multi-task QD | Many related tasks | Crossover across tasks, 2× solutions |
| **POET** | Coevolution | Automatic curriculum | Environment-agent coevolution |
| **Enhanced POET** | Coevolution | Unbounded generation | Domain-general, CPPNs |
| **JEDi** | QD-ES hybrid | Optimization focus | Intelligent behavior targeting |
| **OMNI-EPIC** | LLM-driven | Unlimited tasks | Code generation, natural language |
| **Digital Red Queen** | Adversarial self-play | LLM code evolution, cybersecurity | Convergent evolution, robust generalists |
| **DNS** | Parameter-free QD | High-D, unsupervised | No tuning, scales naturally |
| **Soft-QD (SQUAD)** | Continuous QD | High-D, differentiable | No discretization, gradient-based |
| **GAME** | Adversarial QD | Red teaming, game balancing | Both-sided illumination, tournament metrics |
| **Extract-QD** | Uncertain QD | Noisy/stochastic domains | Adaptive sampling, reproducibility |
| **PGA-ME** | QD-RL | Neuroevolution (unidirectional) | Hybrid GA+PG, 10× speedup |
| **DCG-ME** | QD-RL | Neuroevolution (omnidirectional) | Descriptor conditioning, archive distillation |
| **DCRL-ME** | QD-RL | Efficient neuroevolution | Actor injection, 2.75× sample efficiency |

---

## Relationships

### Multi-Task QD Evolution
- **MAP-Elites** → **Multi-Task MAP-Elites** (MT-ME): Extend to finite multiple tasks
- **MT-ME** → **PT-ME**: Continuous task space + local linear regression
- **MT-ME** → **MTMB-ME**: Multiple diverse solutions per task (not just 1)

### Open-Ended Learning
- **POET** → **Enhanced POET**: Domain-general extension with CPPNs
- **Enhanced POET** → **OMNI-EPIC**: Replace CPPNs with LLM code generation
- **MAP-Elites** → **Digital Red Queen**: Adversarial self-play with LLM mutations in Core War

### QD Variants
- **MAP-Elites** → **JEDi**: Add intelligent targeting with GP
- **MAP-Elites** → **DNS**: Remove grid, use dominated novelty
- **MAP-Elites** → **Soft-QD**: Remove discretization entirely, continuous formulation with gradients
- **MAP-Elites** → **GAME**: Extend to adversarial coevolution (uses MTMB-ME internally)
- **Enhanced POET** (PATA-EC) → **GAME** (Ranking): Ranking vectors for task selection
- **MAP-Elites** → **Extract-QD**: Add uncertainty handling via resampling

### QD-RL (Deep RL Integration) ✨ NEW
- **MAP-Elites** → **PGA-ME**: Add policy gradients for neuroevolution (10× speedup)
- **PGA-ME** → **DCG-ME**: Add descriptor conditioning to fix omnidirectional tasks (+82%)
- **DCG-ME** → **DCRL-ME**: Replace actor evaluation with actor injection (2.75× efficiency)

---

**See individual files for detailed implementations, hyperparameters, and code examples.**
