# Descriptor-Conditioned Gradients MAP-Elites (DCG-ME)

**Quick reference**: [[DCG-ME_detailed]]

---

## Overview

**DCG-MAP-Elites** extends PGA-ME with **descriptor-conditioned critic and actor** to solve the omnidirectional task failure problem. It achieves 82% improvement over PGA-ME while **simultaneously distilling** the entire archive into a single versatile policy.

**Authors**: Faldor, Chalumeau, Flageat, Cully  
**Year**: 2023  
**Key Innovation**: Descriptor-conditioned actor-critic Q(s,a|d) that guides mutations toward specific descriptors while improving fitness

---

## The Problem PGA-ME Fails On

**Omnidirectional locomotion**: Robot should reach all (x,y) positions while minimizing energy.

What happens with PGA-ME:
1. Critic learns: "Best action = don't move" (minimizes energy globally)
2. PG operator pushes ALL solutions toward zero movement
3. All behaviors collapse to origin (0,0)
4. **Diversity destroyed!**

**Root cause**: Critic Q(s,a) only optimizes global fitness, has no concept of descriptors.

---

## Solution: Descriptor-Conditioned Critic

### The Innovation

Instead of Q(s,a), use **Q(s,a|d)** that estimates return **when achieving target descriptor d**.

```
S(d, d') = exp(-||d - d'|| / L)  [similarity function]

Q(s,a|d') = S(d,d') · Q(s,a)
          ≡ reward scaled by descriptor similarity
```

**Effect**:
- If d ≈ d': S ≈ 1 → learns to achieve d'
- If d ≠ d': S ≈ 0 → learns this action doesn't achieve d'

**Result**: PG mutations improve fitness **while maintaining** target descriptor → preserves diversity!

---

## Key Components

### 1. Descriptor-Conditioned Critic

```python
Q(s, a | d) takes:
  - State s
  - Action a
  - Target descriptor d
Outputs: Expected return when achieving d
```

Trained with similarity scaling:
```
target = S(d_actual, d_target) · reward + ...
```

### 2. Descriptor-Conditioned Actor

```python
π(s | d) takes:
  - State s
  - Target descriptor d
Outputs: Action to achieve d while maximizing fitness
```

**Bonus**: By training on all archive descriptors, the actor **distills entire archive** into single policy!

### 3. Implicit Descriptor Sampling

No handcrafted descriptor sampling needed:

**PG offspring**: Target = parent's descriptor (PG mutates toward parent descriptor)
**GA offspring**: Target = observed descriptor (GA produces whatever behavior)
**Actor evaluation**: Sample d' from archive, observe actual d → natural negative samples

---

## Algorithm Overview

```
Same as PGA-ME + descriptor conditioning:

1. Train descriptor-conditioned actor-critic on transitions (s,a,r,s',d,d')
2. Variation: GA + DC-PG (conditioned on parent descriptor)
3. Actor evaluation: Generate π(·|d') for batch of d' from archive
   → Produces transitions with d ≠ d' for robust training
4. Implicit curriculum: Early training has many d ≠ d' (negative examples),
   late training has fewer as actor improves
```

---

## When to Use DCG-ME

✅ **Use when**:
- PGA-ME fails (omnidirectional tasks with deceptive fitness)
- Want **archive distillation** into single policy
- Need robust descriptor-preserving mutations
- Sufficient budget for actor evaluation (small overhead vs PGA-ME)

❌ **Consider alternatives when**:
- PGA-ME works well (unidirectional) → simpler algorithm
- Very tight evaluation budget → DCRL-ME (no actor evaluation)

---

## Performance Gains

| Task | PGA-ME | DCG-ME | Improvement |
|------|--------|--------|-------------|
| Ant-Omni | 430k | 870k | **+102%** |
| AntTrap-Omni | 420k | 800k | **+90%** |
| Hexapod-Omni | 380k | 680k | **+79%** |
| **Average** | — | — | **+82%** |

On unidirectional tasks (Walker-Uni): Equal or slightly better than PGA-ME.

### Archive Distillation Results

On 2 tasks (Ant-Omni, AntTrap-Omni):
- Single descriptor-conditioned policy achieves **95%+ of archive QD-score**
- Eliminates need to store/execute thousands of policies
- Still 6-8% descriptor error (acceptable in practice)

---

## Comparison: PGA-ME vs DCG-ME vs DCRL-ME

| Feature | PGA-ME | DCG-ME | DCRL-ME |
|---------|--------|--------|---------|
| **Variation** | GA + PG | GA + DC-PG | GA + DC-PG + AI |
| **Descriptor-cond** | ❌ | ✅ | ✅ |
| **Archive distillation** | ❌ | ✅ | ✅ |
| **Actor evaluation** | ❌ | ✅ (b_AE=256) | ❌ (uses AI) |
| **Omnidirectional** | ✗ Fails | ✓ Works | ✓ Works |
| **Sample efficiency** | Baseline | -20% | +50% |

---

## Key Insights

1. **Descriptor conditioning ≡ reward scaling**: Q(s,a|d) is mathematically equivalent to scaling reward by S(d,d')
2. **Implicit sampling**: Variation operators naturally generate positive/negative descriptor samples
3. **Active learning**: Actor evaluation generates negative samples where actor fails
4. **Archive distillation**: Bonus benefit from multi-descriptor training
5. **Robust to deceptive fitness**: Works where global optimization fails

---

## When Descriptor Conditioning Saves You

**Problem scenario**: Fitness function has global optimum that conflicts with diversity

Examples:
- Energy minimization (optimum = motionless)
- Minimize distance to origin (optimum = origin)
- Maximize safety (optimum = no risk = no behavior)

Solution: Descriptor conditioning forces mutations toward specific descriptors despite deceptive fitness.

---

## References

- **Paper**: Faldor et al., "MAP-Elites with Descriptor-Conditioned Gradients..." (GECCO 2023)
- **Building on**: PGA-ME (Nilsson & Cully 2021), TD3 (Fujimoto et al. 2018)
- **Extended by**: DCRL-MAP-Elites (Faldor et al. 2024)
