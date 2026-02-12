# Policy Gradient Assisted MAP-Elites (PGA-ME)

**Quick reference**: [[PGA-ME_detailed]]

---

## Overview

**PGA-MAP-Elites** combines **MAP-Elites** with **Deep Reinforcement Learning** (TD3) to efficiently evolve large neural network policies. It uses hybrid variation operators: 50% genetic algorithm for exploration, 50% policy gradient for exploitation.

**Authors**: Nilsson & Cully  
**Year**: 2021  
**Domain**: Neuroevolution for continuous control  
**Key Contribution**: First successful hybrid QD-RL algorithm, 10× speedup over MAP-Elites

---

## Core Problem

**Traditional MAP-Elites limitation**: Random mutations (GA) are inefficient in high-dimensional spaces → very slow convergence when evolving neural networks with 10k+ parameters.

**Solution**: Add directed search using policy gradients from Deep RL to accelerate fitness improvement.

---

## Key Innovation

### Two Complementary Variation Operators

1. **GA Variation** (50%): Random Gaussian mutations with bimodal distribution
   - σ₁ = 0.005 (small): Local exploitation
   - σ₂ = 0.05 (large): Exploration

2. **PG Variation** (50%): Deterministic policy gradient using critic
   - Uses trained Q-function to guide mutations
   - Exploits fitness landscape structure
   - Improves fitness while preserving descriptor

3. **Actor Injection** (1): Add trained RL actor as high-quality solution

### Synergy

GA provides **exploration** and **stepping stones** → PG exploitation finds local optima → Fast convergence while maintaining diversity.

---

## Algorithm Structure

```
Initialize archive (1024 CVT cells)
Initialize replay buffer + TD3 actor-critic

Loop:
  1. Train critic on replay buffer (TD3)
  2. Select b parents uniformly from archive
  3. GA variation (50%) + PG variation (50%) + Actor injection (1)
  4. Evaluate all offspring + store transitions
  5. Update archive (best per cell)
```

---

## When to Use PGA-ME

✅ **Use when**:
- Evolving **high-dimensional** policies (10k+ parameters)
- **Continuous** state and action spaces
- Want **diverse solutions** (not single optimum)
- **Limited budget** (expensive evaluations)

❌ **Use alternatives when**:
- Omnidirectional tasks (PG optimizes global fitness, ignores descriptors) → DCG-ME
- Want archive distillation → DCG-ME or DCRL-ME
- Discrete actions → Different RL algorithm

---

## Performance Summary

| Task | PGA-ME | MAP-Elites | Notes |
|------|--------|------------|-------|
| **Walker-Uni** | **980k** ✓ | 850k | State-of-the-art |
| **Ant-Omni** | 430k ✗ | 380k | **Fails** (collapses) |
| **Speedup** | **10×** | baseline | vs GA mutations |

**Problem**: PG operator optimizes global fitness → converges to "don't move" → diversity collapses on omnidirectional tasks.

---

## Comparison to Related Methods

| Method | Variation | Descriptor-Cond | Distillation |
|--------|-----------|-----------------|--------------|
| **MAP-Elites** | GA | ❌ | ❌ |
| **ME-ES** | ES | ❌ | ❌ |
| **PGA-ME** | GA + PG | ❌ | ❌ |
| **DCG-ME** | GA + DC-PG | ✅ | ✅ |
| **DCRL-ME** | GA + DC-PG + AI | ✅ | ✅ |

---

## Evolution Path

- **MAP-Elites** → **PGA-ME**: Add policy gradients for neuroevolution
- **PGA-ME** → **DCG-ME**: Add descriptor conditioning to fix omnidirectional tasks + distill to single policy
- **DCG-ME** → **DCRL-ME**: Add actor injection, eliminate actor evaluation

---

## Key Takeaways

1. **Hybrid approach**: Exploration (GA) + Exploitation (PG) synergize well
2. **Sample efficient**: 10× faster than MAP-Elites, 5× faster than ME-ES
3. **Scalable**: Works with policies of 20k+ parameters
4. **Task-dependent**: Excellent on unidirectional, fails on omnidirectional
5. **Foundation**: Base architecture for DCG-ME and DCRL-ME

---

## References

- **Paper**: Nilsson & Cully, "Policy Gradient Assisted MAP-Elites" (GECCO 2021)
- **Comparison**: TD3 (Fujimoto et al. 2018), MAP-Elites (Mouret & Clune 2015)
- **Extensions**: DCG-MAP-Elites (Faldor et al. 2023), DCRL-MAP-Elites (Faldor et al. 2024)
