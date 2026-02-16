# Quality-Diversity Actor-Critic (QDAC)

**Quick reference**: [[QDAC_detailed]]

---

## Overview

**QDAC** is an off-policy actor-critic deep RL algorithm that learns **high-performing and diverse behaviors** by seamlessly unifying two critics: a **value function critic** (for quality) and a **successor features critic** (for diversity). It uses constrained optimization to (1) maximize return while (2) executing diverse skills.

**Authors**: Grillotti, Faldor, González León, Cully (Imperial College London, Iconic AI)
**Year**: 2024
**Key Innovation**: Dual-critic architecture with Lagrangian optimization that balances quality-diversity trade-off via λ(s,z)

**Performance**: 15% more diverse behaviors and 38% higher performance than baselines on 6 locomotion tasks.

---

## The Quality-Diversity Problem

**Goal**: Learn a skill-conditioned policy π(a|s,z) that:
1. **Quality**: Maximizes expected return
2. **Diversity**: Executes different skills z ∈ Z

**Problem Formulation**:
```
For all skills z ∈ Z:
  maximize E[Σ γ^t r_t]
  subject to: φ̄ = z

where φ̄ = lim(T→∞) (1/T)Σ φ_t  (average features = desired skill)
```

**Key insight**: Skills are defined by **expected features** φ(s,a) over trajectories, not just momentary states.

---

## Core Architecture

### Dual Critics

1. **Value Function V(s,z)**: Evaluates quality
   - Standard RL critic: E[Σ γ^i r_{t+i} | s_t = s]
   - Learned via Bellman equation

2. **Successor Features ψ(s,z)**: Evaluates diversity
   - Expected discounted sum of features: E[Σ γ^i φ_{t+i} | s_t = s]
   - Captures agent's future behavior
   - Learned via Bellman equation (φ plays role of reward)

### The Unified Objective

**Key innovation**: Policy Skill Improvement via successor features.

The actor maximizes a **Lagrangian** that unifies both critics:

```
L(s,z) = (1 - λ(s,z)) · V(s,z)
         - λ(s,z) · ||(1-γ)ψ(s,z) - z||

where λ(s,z) ∈ [0,1] is an adaptive Lagrange multiplier
```

**Two terms**:
- **Red term**: Maximize return (quality)
- **Blue term**: Minimize distance between expected features and desired skill (diversity)

**Lagrange multiplier λ(s,z)**:
- Increases when skill constraint is violated → focus on diversity
- Decreases when skill is achieved → focus on performance

---

## How It Works

### Training Loop

1. **Sample skill** z ~ Uniform(Z)
2. **Collect episode** with π(·|s,z) for T steps
3. **Store transitions** (s,a,r,φ,s',z) in replay buffer
4. **Update Lagrange multiplier** λ(s,z):
   - Increase λ if ||(1-γ)ψ(s,z) - z|| > threshold
   - Decrease λ if constraint satisfied
5. **Update critics** V(s,z) and ψ(s,z) via Bellman errors
6. **Update actor** π(s,z) to maximize Lagrangian

### Why Successor Features?

**Problem with naive approach**: Minimizing Σ γ^t ||φ_t - z|| only works when φ_t = z at all times.

**Example - Feet Contact**:
- Skill z = [0.1, 0.6] means "use foot 1 10% of time, foot 2 60% of time"
- Naive approach fails: can't have foot contacts of 0.1 and 0.6 at single timestep!
- **Successor features** reason about **average over trajectory**: (1-γ)ψ(s,z) ≈ φ̄

**Theoretical justification**: QDAC proves upper bound on ||φ̄ - z|| using successor features (Proposition 1).

---

## Key Features

### Advantages

1. **Single versatile policy**: Unlike MAP-Elites methods that output populations
2. **Explicit skill execution**: Can achieve specific target skills z
3. **Adaptive trade-off**: λ(s,z) balances quality-diversity automatically
4. **Works on conflicting skills**: Can learn skills contrary to reward (e.g., negative velocity)
5. **Sample efficient**: Off-policy RL with replay buffer + skill re-labeling

### Comparison with Other Approaches

**vs. MAP-Elites methods (DCG-ME, PGA-ME, QD-PG)**:
- ✅ Single policy instead of population
- ✅ Better performance (38% higher)
- ✅ More diverse (15% more)
- ✅ Can execute skills contrary to reward

**vs. Pure RL QD methods (SMERL, DOMiNO)**:
- ✅ Can achieve specific target skills (not just discover diverse behaviors)
- ✅ Better adaptation performance
- ✅ Constrained optimization instead of reward engineering

**vs. Goal-Conditioned RL (GCRL)**:
- ✅ Balances task reward with skill execution (not just skill)
- ✅ Optimizes for both quality and diversity

---

## Applications

### 1. Diverse Locomotion

**6 challenging tasks** (Walker, Ant, Humanoid × 4 feature types):

**Feature Types**:
- **Feet Contact**: Proportion of time each foot touches ground
  - Useful for damage recovery
- **Velocity**: (v_x, v_y) in xy-plane
  - Classic GCRL task
- **Jump**: Height of lowest foot
  - Requires oscillating around target due to gravity
- **Angle**: Body orientation (cos α, sin α)
  - Forces moonwalking/sidestepping

### 2. Few-Shot Adaptation

**5 perturbed environments** (no retraining allowed):
- Humanoid - Hurdles: Jump over varying heights
- Humanoid - Motor Failure: Adapt to knee motor damage
- Ant - Gravity: Adapt to different gravity
- Walker - Friction: Adapt to ground friction
- Ant - Wall: Hierarchical RL with meta-controller

**Results**: Competitive or better than baselines, especially on hurdles and hierarchical tasks.

---

## Implementation Details

### Model-Free Variant (QDAC)

Built on **Soft Actor-Critic (SAC)**:
- Off-policy actor-critic
- Maximum entropy RL
- Dual Q-networks for stability

### Model-Based Variant (QDAC-MB)

Built on **DreamerV2**:
- World model learns dynamics
- Better on complex features (e.g., Jump with min operator)
- Higher sample efficiency

### Key Tricks

1. **Skill re-labeling**: Duplicate transitions with new random skills → sample efficiency
2. **Cross-entropy loss for λ**: Binary classification (constraint satisfied or not)
3. **Universal function approximators**: V(s,z) and ψ(s,z) share architecture benefits
4. **Threshold ε**: Relaxed constraint ||φ̄ - z|| ≤ ε instead of exact equality

---

## Ablations

**QDAC-SepSkill**: Replace successor features with naive Σ γ^t ||φ_t - z||
- ❌ Fails on Feet Contact (can only execute corner skills)
- Shows importance of successor features critic

**QDAC-FixedLambda**: Use fixed λ instead of adaptive
- ❌ Worse diversity (can't reach skill space edges)
- ❌ Fails on hard tasks (Jump)
- Shows importance of constrained optimization

**UVFA**: Combines both ablations (naive distance + fixed trade-off)
- ❌ Worst performance
- Essentially skill-conditioned value function without diversity mechanism

---

## Mathematical Insight

### From Constraint to Lagrangian

**Problem 1 (P1)**: Intractable constraint φ̄ = z

↓ Relax with L2 norm and threshold

**Problem 2 (P2)**: ||φ̄ - z|| ≤ ε

↓ Upper bound via successor features (Proposition 1)

**Problem 3 (P3)**:
```
maximize E[V(s,z)]
subject to: E[||(1-γ)ψ(s,z) - z||] ≤ ε
```

↓ Lagrangian method

**Final Objective**:
```
(1 - λ(s,z)) V(s,z) - λ(s,z) ||(1-γ)ψ(s,z) - z||
```

---

## Related Concepts

- **Quality-Diversity (QD)**: Generate diverse high-performing solutions
- **Successor Features**: Expected discounted sum of features, enables transfer learning
- **Constrained Optimization**: Lagrange multipliers for quality-diversity trade-off
- **Universal Value Functions**: V(s,g) and ψ(s,z) conditioned on goals/skills
- **Mutual Information**: URL methods maximize I(τ,z) ≈ Σ I(s_t,z)

---

## Limitations & Future Work

### Current Limitations

1. **Hand-defined features**: Requires manual φ(s,a) design (like most QD algorithms)
2. **Ergodic environments**: Assumes stationary state distribution exists
3. **Continuous control**: Only evaluated on locomotion tasks

### Future Directions

1. **Unsupervised feature learning**: Learn φ automatically (like AURORA, SMERL)
2. **Non-ergodic tasks**: Extend to Atari games, sparse reward tasks
3. **Hierarchical composition**: Use skills for hierarchical planning
4. **Transfer learning**: Leverage successor features for zero-shot transfer

---

## Key Takeaways

1. **Dual critics** (value + successor features) enable quality-diversity optimization in RL
2. **Successor features** are crucial for executing skills defined by average features over trajectories
3. **Adaptive Lagrange multiplier** balances quality-diversity trade-off automatically
4. **Single policy** more practical than population-based QD methods for downstream tasks
5. **State-of-the-art** on skill execution and adaptation benchmarks

**One-sentence summary**: QDAC learns a single skill-conditioned policy that can execute diverse high-performing behaviors by optimizing a Lagrangian that unifies value function and successor features critics.
