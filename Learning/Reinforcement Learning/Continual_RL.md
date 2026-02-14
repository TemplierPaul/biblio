# Continual Reinforcement Learning (CRL)

**Quick reference**: [[Continual_RL_detailed]]

---

## Overview

**Continual Reinforcement Learning (CRL)** enables agents to learn from a sequence of tasks in non-stationary environments while avoiding catastrophic forgetting. Unlike traditional RL which assumes stationary environments, CRL handles dynamic environments where task distributions change over time.

**Authors**: Pan et al. (2025), Khetarpal et al. (2022)  
**Key Innovation**: Balances stability-plasticity-scalability triangle (unique to CRL)

**Why it matters**: Real-world agents — robots, autonomous vehicles, dialogue systems — face environments that change. Standard RL breaks when the MDP is non-stationary.

---

## The Problem

### Sequential Task Learning

**Setting**:
- Agent faces tasks sequentially: $\mathcal{T} = \{T_1, T_2, ..., T_N\}$
- Each task has different dynamics/rewards/goals
- Limited/no access to previous task data
- Must maintain performance on all tasks

**Example**: Warehouse robot → hospital navigation → outdoor delivery (must remember all three)

**Failure mode**: Standard RL fine-tuning causes **catastrophic forgetting** — achieving 90% on task 1, then dropping to 30% after learning task 2.

---

## The Triangular Challenge

CRL uniquely balances **three** critical dimensions (not just two like supervised CL):

### 1. Stability
**Definition**: Retain old knowledge; maintain performance on previously learned tasks

**Catastrophic Forgetting**:
$$CF_i = \max(p_{i,i} - p_{N,i}, 0)$$
- $p_{i,i}$: Performance on task $i$ after training on it
- $p_{N,i}$: Performance on task $i$ after training on all $N$ tasks

**Goal**: Minimize $CF$ → Keep old knowledge intact

### 2. Plasticity
**Definition**: Adaptability to learn new tasks effectively

**Two sub-components**:
- **Forward Transfer (FT)**: Old knowledge accelerates new learning ("tasks 1-2 make task 3 faster")
- **Backward Transfer (BT)**: New knowledge improves old tasks ("learning task 3 makes us better at task 1")

**Goal**: Maximize FT and BT

**Plasticity Loss**: Networks progressively lose ability to learn *anything new* (distinct from forgetting). Causes: dead neurons, feature rank collapse, weight magnitude growth.

### 3. Scalability
**Definition**: Learn many tasks with bounded compute and memory resources

**Why critical in CRL** (not in supervised CL):
- RL requires millions of environment interactions per task
- Replay buffers grow linearly with tasks
- Inference latency matters for real-time control

**Goal**: Minimize resources while maximizing task count

### The Trilemma

**Trade-off**:
- **High Stability** → Protect old weights → **Low Plasticity** (can't learn new tasks)
- **High Plasticity** → Update weights freely → **Low Stability** (catastrophic forgetting)
- **High Scalability** → Limit memory/compute → **Low Stability & Plasticity** (insufficient capacity)

**Every CRL method is a different point in this triangle.**

---

## Taxonomy: 4 Types of CRL Methods

### 1. Policy-Focused Methods
**What to store/transfer**: Policy $\pi(a|s)$ or value function $Q(s,a)$

**Three sub-categories**:

#### A. Policy Reuse
Store complete policies, reuse for initialization.
- **Algorithms**: MAXQINIT, ClonEx-SAC, Boolean Task Algebra, CSP

#### B. Policy Decomposition
Shared components + task-specific components.
- **Factor decomposition**: $\theta_k = L \cdot s_k$ (shared basis $L$, task coefficients $s_k$)
- **Multi-head**: Shared trunk + task-specific heads (PNN)
- **Hierarchical**: Options/skills reused (HLifeRL)

#### C. Policy Merging
Merge old and new policies while protecting old knowledge.
- **EWC** (Elastic Weight Consolidation): Fisher Information-weighted regularization
- **Distillation**: P&C (Progress & Compress)
- **Hypernetworks**: HN-PPO

**When to use**:
- Policy Reuse → Very similar tasks
- Decomposition → Clear shared structure
- Merging (EWC) → Cannot store data, limited memory

---

### 2. Experience-Focused Methods
**What to store/transfer**: Past experiences (transitions)

#### A. Direct Replay
Store actual transitions $(s, a, r, s')$ in replay buffer.
- **Algorithms**: CLEAR, SLER, 3RL, CoMPS
- **Pros**: Simple, effective, exact fidelity
- **Cons**: Memory intensive, privacy concerns

#### B. Generative Replay
Generate synthetic past experiences instead of storing real ones.
- **Algorithms**: RePR (GAN-based), S-TRIGGER (VAE-based), Continual-Dreamer
- **Pros**: Memory efficient, privacy-preserving
- **Cons**: Generation fidelity degrades, feature drift

**When to use**:
- Direct replay → Memory available, privacy not critical
- Generative → Privacy critical, memory limited

---

### 3. Dynamic-Focused Methods
**What to store/transfer**: Environment dynamics $T(s'|s,a)$

#### A. Direct Modeling
Explicitly learn transition functions.
- **Algorithms**: HyperCRL, LLIRL, VBLRL
- **Mechanism**: Mixture models, Chinese Restaurant Process, hypernetworks

#### B. Indirect/Latent Modeling
Use latent representations without explicit dynamics.
- **Algorithms**: LILAC, LiSP, Continual-Dreamer
- **Mechanism**: World models, RNN context encoding

**When to use**: Sample efficiency critical, planning needed, good model class available

**Pros**: Sample efficient, enables planning
**Cons**: Model errors compound, harder to scale to high-dimensional observations

---

### 4. Reward-Focused Methods
**What to store/transfer**: Reward shaping functions or intrinsic motivation

**General form**: 
$$R^M_t = R_t + h(s,a,s') + \alpha \cdot R^I_t$$

**Two approaches**:
1. **Reward Shaping**: Potential-based $h(s,s') = \gamma \Phi(s') - \Phi(s)$ (SR-LLRL, ELIRL)
2. **Intrinsic Motivation**: Curiosity bonuses (IML, MT-Core)

**When to use**: Have domain knowledge, sparse rewards problem

---

## Special Methods Often Overlooked

### Successor Features (SF) & GPI
**Key idea**: Decompose value function as $Q^\pi(s,a) = \psi^\pi(s,a)^\top \mathbf{w}$
- $\psi^\pi$: Successor features (dynamics-dependent, reward-independent)
- $\mathbf{w}$: Reward weights (task-specific)

**For CRL**: If tasks share dynamics but differ in rewards → Learn $\psi$ once, only update $\mathbf{w}$ per task. **Zero-shot transfer** to new reward functions!

**Generalized Policy Improvement (GPI)**: Select best action from best previous policy for each state — no retraining.

### Meta-RL as CRL Alternative
**Idea**: Learn to adapt quickly instead of preventing forgetting.

**Key methods**:
- **MAML**: Gradient-based meta-initialization
- **RL²**: RNN-based in-context learning
- **AdA**: Transformer-scale open-ended adaptation
- **Algorithm Distillation**: Distill learning histories into transformer

**When better than CRL**: Tasks from known distribution, adaptation more important than retention

---

## Reset Baselines (Embarrassingly Strong)

**Surprising finding**: Simple periodic reset + replay buffer often outperforms complex CRL methods.

**Why it works**: Addresses plasticity loss (dead neurons) without complex mechanisms.

**Key approaches**:
- **Periodic reset**: Reset weights, keep buffer
- **Shrink & perturb**: $\theta \leftarrow \alpha \theta + (1-\alpha)\theta_{init}$
- **Continual Backpropagation**: Reinitialize dormant neurons
- **UPGD**: Perturb low-utility parameters

**Implication**: Any new CRL method should be compared against "periodic reset + replay"

---

## Foundation Models for CRL

**Frozen Backbone Approach**: Pre-trained representation + lightweight per-task head

**Why it helps**:
- Frozen representation → catastrophic forgetting mostly disappears
- Small head → can be reset (plasticity maintained)
- Representation implicitly transfers

**Key models**: R3M, VIP, RT-2, Octo

**When it fails**: New tasks require genuinely novel features not in pre-training

---

## When to Use What

| Use Case | Recommended Method | Why |
|----------|-------------------|-----|
| **Memory available** | Direct replay (CLEAR) | Simple, effective, exact fidelity |
| **Memory limited** | EWC or generative replay | No data storage needed |
| **Privacy critical** | Generative replay | No raw data stored |
| **Related tasks** | Policy decomposition | Shared structure exploitation |
| **Unrelated tasks** | PNN or reset + replay | Avoid negative transfer |
| **Sample efficiency** | Model-based (HyperCRL) | Planning reduces interactions |
| **Zero-shot transfer** | Successor Features + GPI | Reward changes only |
| **Known task distribution** | Meta-RL (MAML, RL²) | Fast adaptation > retention |
| **Foundation model available** | Frozen backbone + heads | Sidesteps forgetting entirely |

---

## CRL vs Related Paradigms

| Aspect | CRL | Multi-Task RL | Meta-RL | Transfer RL |
|--------|-----|---------------|---------|-------------|
| **Data access** | Sequential, limited past | All tasks always | i.i.d. task samples | Source → target (one-shot) |
| **Goal** | No forgetting + transfer | Optimize all tasks jointly | Fast adaptation | Target task only |
| **Forgetting** | Major challenge | Not applicable | Can occur | Acceptable |
| **Evaluation** | All tasks + transfer metrics | Average performance | Adaptation speed | Target performance |

---

## Evaluation

### Core Metrics

1. **Average Performance**: $A_N = \frac{1}{N} \sum_{i=1}^{N} p_{N,i}$
2. **Forgetting**: $FG = \frac{1}{N-1} \sum_{i=1}^{N-1} \max(p_{i,i} - p_{N,i}, 0)$
3. **Forward Transfer**: Impact of old knowledge on new learning
4. **Backward Transfer**: New knowledge improving old tasks

### Comparison Baselines
- **MTL** (upper bound): Access to all tasks simultaneously
- **Single-Task Expert**: Per-task ceiling
- **Fine-tuning** (lower bound): Naive sequential training
- **Periodic Reset + Replay**: Strong simple baseline

### What to Report
- Performance matrix $P_{ij}$ (task $i$ after training on task $j$)
- All three dimensions: Forgetting, transfer, resources
- Multiple task orderings (random + adversarial)
- Multiple seeds (≥5 for RL)

---

## Benchmarks

| Benchmark | Environment | Tasks | Key Feature |
|-----------|-------------|-------|-------------|
| **Continual World** | Meta-World (MuJoCo) | 10 | Robotic manipulation |
| **CORA** | Atari, Procgen, MiniHack | Variable | Multi-environment |
| **COOM** | ViZDoom | Multiple | Embodied perception |
| **Atari Sequences** | Atari games | Variable | Classic benchmark |
| **LIBERO** | Robotic tasks | Multiple | Long-horizon manipulation |

**Frameworks**: CORA, TELLA, Avalanche-RL

---

## CRL vs Supervised CL

| Aspect | Supervised CL | Continual RL |
|--------|---------------|--------------|
| **Core Challenge** | Stability-Plasticity | Stability-Plasticity-**Scalability** |
| **Data** | i.i.d. within task | Non-i.i.d. trajectories |
| **Feedback** | Labels (deterministic) | Rewards (stochastic, sparse) |
| **Exploration** | N/A | Critical component |
| **Compute per task** | Moderate | Massive (millions of interactions) |

**Why scalability matters**: RL requires orders of magnitude more compute than supervised learning.

---

## Practical Recommendations

### For Researchers
1. Report all three dimensions (forgetting, transfer, resources)
2. Include strong baselines (periodic reset + replay, frozen backbone)
3. Test multiple task orderings
4. Use ≥5 random seeds (RL is noisy)

### For Practitioners
1. **Start simple**: Try frozen backbone → EWC → CLEAR
2. **Match method to constraint**:
   - Memory limited → EWC
   - Compute limited → Policy reuse
   - Privacy critical → Generative replay
3. **Monitor plasticity**: Track learning speed on new tasks (not just forgetting)
4. **Consider meta-RL**: If tasks from known distribution

---

## Key Insights

1. **CRL ≠ CL + RL**: Unique challenges (scalability, exploration, non-stationary rewards)
2. **Plasticity loss ≠ Forgetting**: Networks lose ability to learn new tasks (distinct problem)
3. **No silver bullet**: Every method trades off stability, plasticity, scalability
4. **Simple baselines strong**: Periodic reset + replay often competitive
5. **Foundation models help**: Frozen representations sidestep forgetting
6. **Task order matters**: Performance highly sensitive to sequence
7. **Theory lags practice**: Empirical methods ahead of formal understanding

---

## Key Algorithms Quick Reference

### Regularization
- **EWC**: Fisher Information-weighted L2 penalty
- **SI**: Online parameter importance
- **P&C**: Distillation-based knowledge compression

### Architecture
- **PNN**: Frozen columns + lateral connections
- **PackNet**: Iterative pruning for capacity
- **Supermask**: Binary masks for subnetworks

### Replay
- **CLEAR**: Dual-buffer complementary learning
- **CoMPS**: Meta-policy search + replay
- **RePR**: GAN-based generative replay

### Transfer
- **SF/GPI**: Successor features for zero-shot
- **UVFA**: Goal-conditioned value functions
- **Boolean Task Algebra**: Policy composition

### Plasticity
- **Periodic Reset**: Reset weights, keep buffer
- **Continual Backpropagation**: Reinitialize dormant neurons
- **Shrink & Perturb**: Interpolate toward initialization

### Meta-RL
- **MAML**: Gradient-based meta-learning
- **RL²**: RNN in-context learning
- **AdA**: Transformer-scale adaptation

---

## Open Problems

1. **Task boundary detection**: Detecting distributional shifts without supervision
2. **True backward transfer**: Actively improving old tasks from new knowledge
3. **Scaling to 1000+ tasks**: Current benchmarks test 10-100 tasks
4. **Theory**: Formal analysis of stability-plasticity-scalability trade-offs
5. **LLM integration**: Continual RLHF, LLM-driven reward design
6. **Multi-agent CRL**: Continual learning with co-evolving agents
7. **Safety constraints**: Exploration under safety constraints during transitions

---

## References

**Surveys**:
- Pan et al. (2025), *A Survey of Continual Reinforcement Learning*
- Khetarpal et al. (2022), *Towards Continual Reinforcement Learning*

**Catastrophic Forgetting**:
- Kirkpatrick et al. (2017), *Overcoming Catastrophic Forgetting* (EWC)
- Rusu et al. (2016), *Progressive Neural Networks*
- Rolnick et al. (2019), *Experience Replay for Continual Learning* (CLEAR)

**Plasticity Loss**:
- Lyle et al. (2023, 2024), *Understanding and Disentangling Plasticity Loss*
- Dohare et al. (2024), *Loss of Plasticity in Deep Continual Learning* (Nature)
- Nikishin et al. (2022), *The Primacy Bias in Deep RL*

**Successor Features**:
- Barreto et al. (2017, 2018), *Successor Features for Transfer*
- Schaul et al. (2015), *Universal Value Function Approximators*

**Meta-RL**:
- Finn et al. (2017), *Model-Agnostic Meta-Learning* (MAML)
- Team et al. (2023), *Human-Timescale Adaptation* (AdA)
- Laskin et al. (2022), *Algorithm Distillation*

**Benchmarks**:
- Wolczyk et al. (2021), *Continual World*
- Powers et al. (2022), *CORA*

**See `Continual_RL_detailed.md` for complete algorithm implementations, mathematical formulations, and technical details.**
