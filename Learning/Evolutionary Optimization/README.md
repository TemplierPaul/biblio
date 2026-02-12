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

## DNS (Dominated Novelty Search)
**What**: Parameter-free QD using dominated novelty score (distance to fitter solutions)
**How**: Competition fitness = mean distance to k-nearest better-performing solutions
**Key innovation**: No bounds/grids/thresholds needed, reformulates QD as fitness transformation
**When to use**: High-D (5D+), discontinuous spaces, unsupervised descriptors
**Difference from MAP-Elites**: Dynamic adaptation vs. fixed grid, scales to 1000D

---

## Summary Table

| Algorithm | Type | Best For | Key Strength |
|-----------|------|----------|--------------|
| **MAP-Elites** | Grid-based QD | Low-D, known bounds | Interpretable illumination |
| **POET** | Coevolution | Automatic curriculum | Environment-agent coevolution |
| **Enhanced POET** | Coevolution | Unbounded generation | Domain-general, CPPNs |
| **JEDi** | QD-ES hybrid | Optimization focus | Intelligent behavior targeting |
| **OMNI-EPIC** | LLM-driven | Unlimited tasks | Code generation, natural language |
| **DNS** | Parameter-free QD | High-D, unsupervised | No tuning, scales naturally |

---

## Relationships

- **POET** → **Enhanced POET**: Domain-general extension with CPPNs
- **MAP-Elites** → **JEDi**: Add intelligent targeting with GP
- **MAP-Elites** → **DNS**: Remove grid, use dominated novelty
- **Enhanced POET** → **OMNI-EPIC**: Replace CPPNs with LLM code generation

---

**See individual files for detailed implementations, hyperparameters, and code examples.**
