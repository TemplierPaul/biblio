# Quality-Diversity with Reinforcement Learning

Algorithms combining Quality-Diversity with Deep RL for efficient neuroevolution.

---

## Overview

This folder contains QD-RL algorithms combining evolutionary and reinforcement learning approaches:

### Evolutionary QD-RL (Population-Based)

#### 1. **PGA-ME** (Policy Gradient Assisted MAP-Elites)
- **Key idea**: Hybrid variation operator (50% GA + 50% PG)
- **Paper**: Nilsson & Cully (2021)
- **Status**: [[PGA-ME]] overview | [[PGA-ME_detailed]] implementation
- **Best for**: Unidirectional tasks with high-dimensional policies
- **Limitation**: Fails on omnidirectional tasks (diversity collapses)

#### 2. **DCG-ME** (Descriptor-Conditioned Gradients MAP-Elites)
- **Key idea**: Descriptor-conditioned critic Q(s,a|d) + archive distillation
- **Paper**: Faldor et al. (2023)
- **Status**: [[DCG-ME]] overview | [[DCG-ME_detailed]] implementation
- **Solves**: PGA-ME's omnidirectional failure (+82% improvement)
- **Bonus**: Distills entire archive into single versatile policy
- **Limitation**: Actor evaluation overhead (~20% sample efficiency cost)

#### 3. **DCRL-ME** (Descriptor-Conditioned RL MAP-Elites)
- **Key idea**: Use descriptor-conditioned actor as generative model (Actor Injection)
- **Paper**: Faldor et al. (2024)
- **Status**: [[DCRL-ME]] overview | [[DCRL-ME_detailed]] implementation
- **Solves**: DCG-ME's sample efficiency problem (2× better)
- **Innovation**: Weight extraction technique for actor specialization
- **Best**: Combines all benefits with maximum efficiency

### Pure RL QD (Single Policy)

#### 4. **QDAC** (Quality-Diversity Actor-Critic)
- **Key idea**: Dual critics (value + successor features) with Lagrangian optimization
- **Paper**: Grillotti et al. (2024)
- **Status**: [[QDAC]] overview | [[QDAC_detailed]] implementation
- **Advantage**: Single policy (no population), explicit skill execution
- **Performance**: 15% more diverse, 38% higher performance vs baselines
- **Innovation**: Constrained optimization balances quality-diversity automatically

---

## Algorithm Evolution

### Population-Based Branch (MAP-Elites variants)

```
MAP-Elites (2015): Grid-based QD with GA mutations
    ↓
    Problem: Inefficient in high-D spaces
    ↓
PGA-ME (2021): Add policy gradients for neuroevolution
    ↓ 10× speedup on 20k-parameter policies
    ↓
    Problem: Fails on omnidirectional tasks (diversity collapses)
    ↓
DCG-ME (2023): Add descriptor conditioning to critic + actor
    ↓ +82% improvement on omnidirectional tasks
    ↓
    Problem: Actor evaluation is expensive (256 evals/iter)
    ↓
DCRL-ME (2024): Replace actor evaluation with actor injection
    ↓ 2.75× fewer evaluations per iteration
    ↓
    Final result: Efficient neuroevolution with descriptor control
```

### Pure RL Branch (Single Policy)

```
GCRL / URL: Goal-conditioned or unsupervised RL
    ↓
    Problem: No quality-diversity trade-off
    ↓
SMERL (2020): MI-based diversity + near-optimality threshold
    ↓
    Problem: Reward engineering, can't execute specific skills
    ↓
DOMiNO (2022): Successor features for diversity
    ↓
    Problem: Discovers diverse behaviors but can't target specific skills
    ↓
QDAC (2024): Dual critics (V + ψ) + Lagrangian optimization
    ↓ Explicit skill execution + adaptive quality-diversity trade-off
    ↓
    Final result: Single versatile policy with constrained optimization
```

---

## Quick Comparison

### Population-Based Algorithms

| Feature | PGA-ME | DCG-ME | DCRL-ME |
|---------|--------|--------|---------|
| **Variation** | GA + PG | GA + DC-PG | GA + DC-PG + AI |
| **Descriptor-conditioned** | ❌ | ✅ | ✅ |
| **Archive distillation** | ❌ | ✅ | ✅ |
| **Actor evaluation** | ❌ | ✅ (expensive) | ❌ (eliminated) |
| **Works omnidirectional** | ✗ Fails | ✓ Works | ✓ Works |
| **Evaluations/iter** | 256 | 704 | 256 |
| **Speedup vs MAP-Elites** | 10× | 5× | 10× |
| **Output** | Population | Population + Policy | Population + Policy |

### Population vs Pure RL

| Feature | DCRL-ME | QDAC |
|---------|---------|------|
| **Approach** | Evolutionary + RL | Pure RL |
| **Output** | Population + Policy | Single Policy |
| **Skill execution** | Via distilled policy | Native |
| **Quality-diversity trade-off** | Archive selection | Lagrangian (adaptive λ) |
| **Critics** | Q(s,a\|d) | V(s,z) + ψ(s,z) |
| **Successor features** | ❌ | ✅ |
| **Conflicting skills** | Limited | ✅ Excellent |
| **Performance vs baselines** | +82% (vs PGA-ME) | +38% (vs all) |
| **Diversity vs baselines** | +82% (vs PGA-ME) | +15% (vs all) |
| **Best for** | Population diversity | Single versatile policy |

---

## When to Use Each

### Population-Based (MAP-Elites variants)

#### PGA-ME
- ✅ Unidirectional locomotion tasks
- ✅ Maximization problems (not minimization)
- ✅ Simplest algorithm, fewest dependencies
- ❌ Omnidirectional motion
- ❌ Energy minimization

#### DCG-ME
- ✅ Omnidirectional tasks
- ✅ Want archive distillation
- ✅ Evaluation budget available
- ✅ Deceptive fitness landscapes
- ❌ Very tight evaluation budget

#### DCRL-ME
- ✅ Omnidirectional tasks with tight budget
- ✅ Maximum sample efficiency needed
- ✅ Want descriptor conditioning + distillation
- ✅ Suitable for expensive evaluations (robot sim, etc)
- ✅ Need population for diversity analysis
- ❌ Evaluation is bottleneck (not computation)

### Pure RL (Single Policy)

#### QDAC
- ✅ Need single versatile policy (not population)
- ✅ Skills defined by trajectory statistics (e.g., feet contact proportions)
- ✅ Want explicit skill execution (target specific z)
- ✅ Skills may conflict with reward (e.g., negative velocity)
- ✅ Need adaptation/transfer/hierarchical RL capabilities
- ✅ Off-policy RL infrastructure available (SAC/DreamerV2)
- ✅ Automatic quality-diversity balancing (adaptive λ)
- ❌ Need population for evolutionary analysis
- ❌ Want archive of discrete solutions

---

## File Structure

### Overview Files (High-Level)
- **PGA-ME.md** (~4KB): Concept, key innovation, when to use
- **DCG-ME.md** (~5.5KB): Descriptor conditioning idea, why it works
- **DCRL-ME.md** (~7KB): Actor injection mechanism, sample efficiency gains
- **QDAC.md** (~13KB): Dual critics architecture, Lagrangian optimization, applications

### Detailed Files (Full Implementation)
- **PGA-ME_detailed.md** (~7KB): Complete algorithm, all operators, hyperparameters
- **DCG-ME_detailed.md** (~10KB): Descriptor-conditioned critic/actor, training procedure
- **DCRL-ME_detailed.md** (~11KB): Weight extraction, actor injection, worked examples
- **QDAC_detailed.md** (~26KB): Complete implementation, theoretical proofs, evaluation metrics, ablations

---

## Key Concepts

### 1. Descriptor Conditioning
**Problem**: Global fitness optimization ignores diversity → PG converges to single optimum.

**Solution**: Condition critic on target descriptor → Q(s,a|d) estimates return when achieving descriptor d.

**Implementation**: Reward scaling by similarity S(d,d') = exp(-||d-d'||/L).

### 2. Actor Injection
**Problem**: Descriptor-conditioned actor π_φ(s|d) has different architecture than archive policies π(s).

**Solution**: Extract state weights, bake descriptor into bias → get unconditioned specialized policy.

**Formula**:
```
New bias = d^T W_descriptor + b_original
New policy = π_d(s) = existing_layers(W_state @ s + new_bias)
```

### 3. Implicit Descriptor Sampling
**Automatic curriculum** without handcrafted descriptor selection:
- GA offspring: Target descriptor = observed descriptor
- PG offspring: Target descriptor = parent descriptor
- AI offspring: Target descriptor = sampled descriptor
- Result: Natural positive/negative sample mix

### 4. Successor Features for Skills (QDAC)
**Problem**: Can't execute skills defined by trajectory statistics using naive approach.

**Example**: Feet contact z = [0.1, 0.6] means "use foot 1 10% of time, foot 2 60% of time" over entire trajectory.

**Naive approach fails**: Minimize Σ γ^t ||φ_t - z|| requires φ_t = z at every timestep (impossible for proportions).

**Solution**: Successor features ψ(s,z) = E[Σ γ^i φ_{t+i}] captures expected features over trajectory.

**Result**: Constraint ||(1-γ)ψ(s,z) - z|| ≤ ε ensures average features match skill.

### 5. Lagrangian Quality-Diversity Trade-off (QDAC)
**Objective**: Maximize (1 - λ(s,z))·V(s,z) - λ(s,z)·||(1-γ)ψ(s,z) - z||

**Components**:
- Red term: Maximize return (quality)
- Blue term: Execute skill (diversity)
- λ ∈ [0,1]: Adaptive multiplier

**Adaptation**:
- λ↑ when skill constraint violated → focus on diversity
- λ↓ when skill achieved → focus on quality

**Advantage**: Automatic balancing without manual tuning.

---

## Experimental Results

### Omnidirectional Tasks (Where DCG-ME Shines)

| Task | PGA-ME | DCG-ME | DCRL-ME |
|------|--------|--------|---------|
| Ant-Omni | 430k | **870k** | **900k** |
| AntTrap-Omni | 420k | **800k** | **850k** |
| Hexapod-Omni | 380k | **680k** | **700k** |

### Unidirectional Tasks (Simpler)

| Task | PGA-ME | DCG-ME | Notes |
|------|--------|--------|-------|
| Walker-Uni | **980k** | **980k** | No improvement needed |

### Sample Efficiency (DCRL-ME Wins)

| Metric | DCG-ME | DCRL-ME | Improvement |
|--------|--------|---------|-------------|
| Evals/iteration | 704 | 256 | 2.75× |
| Total budget for 1M evals | 1M | ~360k | 2.75× |

---

## Cross-References

- [[MAP-Elites]] — Base QD algorithm
- [[PT-ME]] — Continuous multi-task variant
- [[MTMB-ME]] — Multi-task multi-behavior
- [[DNS]] — Parameter-free QD
- [[GAME]] — Adversarial QD

---

## Papers & Code

### Population-Based
- **PGA-ME**: Nilsson & Cully, GECCO 2021 | [[PGA-ME_detailed]]
- **DCG-ME**: Faldor et al., GECCO 2023 | [[DCG-ME_detailed]]
- **DCRL-ME**: Faldor et al., TELO 2024 | [[DCRL-ME_detailed]]

### Pure RL
- **QDAC**: Grillotti, Faldor, González León, Cully, ICML 2024 | [[QDAC_detailed]] | [Code](https://github.com/adaptive-intelligent-robotics/QDAC)

---

## How to Use These Files

### For Quick Understanding
1. Read **overview** file (PGA-ME.md, DCG-ME.md, or DCRL-ME.md)
2. Review quick comparison table above
3. Decide which algorithm fits your problem

### For Implementation
1. Read **overview** file for conceptual understanding
2. Read **_detailed** file for complete algorithm
3. Review worked examples in **_detailed** file
4. Implement from hyperparameters + pseudocode provided

### For Troubleshooting
- Check architecture diagrams in DCG-ME_detailed and DCRL-ME_detailed
- Review worked examples showing step-by-step execution
- Compare to reference implementations in QDax (JAX)

---

## Summary

**QD-RL combines Quality-Diversity with Deep RL** in two paradigms:

### Population-Based (Evolutionary + RL)
- **PGA-ME**: 10× speedup over MAP-Elites for neuroevolution
- **DCG-ME**: Fixes omnidirectional tasks, adds archive distillation
- **DCRL-ME**: Improves sample efficiency 2.75× via actor injection
- **Best for**: Maintaining diverse population, evolutionary analysis

### Pure RL (Single Policy)
- **QDAC**: Dual critics (V + ψ) with Lagrangian optimization
- **Performance**: 38% higher quality, 15% more diversity vs baselines
- **Innovation**: Successor features enable trajectory-level skill execution
- **Best for**: Single versatile policy, adaptation, hierarchical RL

**Choose based on**:
- **Population needed?** → DCRL-ME
- **Single policy preferred?** → QDAC
- **Skills = trajectory statistics?** → QDAC (successor features crucial)
- **Maximum sample efficiency?** → DCRL-ME (if population OK) or QDAC (if single policy)
