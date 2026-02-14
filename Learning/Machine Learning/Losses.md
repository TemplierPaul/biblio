# Losses in Deep Learning — Overview

A loss function defines what the model optimizes. Choosing the right loss is as important as choosing the right architecture — it encodes your assumptions about the problem, the data, and what "good" means. This document maps the landscape from basic supervised losses to frontier RL, LLM alignment, and open-endedness objectives.

---

## 1. The Big Picture

Every loss answers one question: **what should the model do better?**

| Family | What It Optimizes | Core Setting |
|--------|-------------------|-------------|
| **Regression losses** | Predict continuous values accurately | Supervised, continuous targets |
| **Classification losses** | Assign correct discrete labels | Supervised, discrete targets |
| **Probabilistic / likelihood losses** | Model the data distribution | Generative, density estimation |
| **Contrastive / metric losses** | Learn similarity structure | Representation learning, few-shot |
| **Reconstruction losses** | Reproduce inputs from compressed codes | Autoencoders, compression |
| **Adversarial losses** | Fool a discriminator / be unfoolable | GANs, adversarial training |
| **Diffusion / score losses** | Estimate score function / denoise | Diffusion models |
| **RL policy losses** | Maximize expected return | Sequential decision-making |
| **RL value losses** | Estimate future return accurately | Critic training, planning |
| **Alignment losses** | Match model behavior to human preferences | RLHF, DPO, LLM fine-tuning |
| **Diversity / quality-diversity losses** | Explore behavior space while maintaining quality | Open-endedness, MAP-Elites, QD |
| **Regularization losses** | Constrain model complexity or behavior | Added to any primary loss |
| **Auxiliary / self-supervised losses** | Extract signal from unlabeled data | Pre-training, representation learning |

---

## 2. Taxonomy by Problem Type

### Supervised Learning

```
Regression:
  MSE → L1 / Huber → Quantile → Log-Cosh → Tukey

Classification:
  Cross-Entropy → Focal Loss → Label Smoothing CE → Poly Loss
  Hinge → Squared Hinge
  Binary CE → Weighted BCE → Asymmetric Loss
```

### Representation Learning

```
Contrastive:
  Triplet → N-pair → InfoNCE / NT-Xent → SupCon
  
Self-supervised:
  SimCLR (NT-Xent) → BYOL (prediction) → VICReg (variance-invariance-covariance)
  → Barlow Twins (redundancy reduction) → DINO (self-distillation)

Metric learning:
  Contrastive → Triplet → ArcFace / CosFace → ProxyNCA
```

### Generative Models

```
VAE:             ELBO = Reconstruction + KL divergence
GAN:             Minimax → Wasserstein → Hinge → Non-saturating
Normalizing Flow: Exact negative log-likelihood
Diffusion:       Denoising score matching / simplified L_simple
Autoregressive:  Cross-entropy (next-token prediction)
```

### Reinforcement Learning

```
Policy optimization:
  REINFORCE → Actor-Critic → PPO (clipped surrogate) → V-MPO → AWR

Value learning:
  TD(0) → TD(λ) → n-step → Retrace → Distributional (QR-DQN, IQN, C51)

Model-based:
  Dynamics MSE → ELBO (world models) → Contrastive (CURL, SPR)

Exploration:
  Curiosity (prediction error) → RND → Count-based → MaxEnt

Multi-agent:
  Self-play → PSRO regret → NeuPL mixture loss → Population diversity
```

### LLM Alignment

```
RLHF:            PPO on reward model score
DPO:             Direct preference optimization (no reward model)
KTO:             Kahneman-Tversky optimization (unpaired preferences)
ORPO:            Odds Ratio Preference Optimization
SimPO:           Simple Preference Optimization (reference-free)
GRPO:            Group Relative Policy Optimization (DeepSeek)
```

### Open-Endedness and Quality-Diversity

```
MAP-Elites:      Fitness within behavior niches
QD-score:        Σ fitness across filled niches
Novelty search:  Distance to k-nearest archive members
AURORA:          Learned behavior characterization + QD
```

---

## 3. Decision Flowchart

```
What's your target?
├── Continuous value → Regression losses (MSE / Huber / Quantile)
├── Discrete class → Classification losses (CE / Focal / Hinge)
├── Probability distribution → Likelihood (NLL / KL / ELBO)
├── Ranking / similarity → Metric losses (Triplet / InfoNCE)
├── Generate data → Generative losses (GAN / VAE / Diffusion)
├── Sequential decisions → RL losses (PPO / SAC / TD)
├── Human preferences → Alignment losses (DPO / RLHF / GRPO)
└── Diverse solutions → QD losses (MAP-Elites / Novelty + Fitness)
```

---

## 4. Key Principles

**1. Match loss to output distribution**: MSE assumes Gaussian noise. CE assumes categorical. Quantile loss makes no distributional assumptions.

**2. Gradient behavior matters more than loss value**: Huber exists because MSE gradients explode on outliers. Focal loss exists because CE gradients vanish on easy examples. PPO clip exists because policy gradients have high variance.

**3. Losses encode inductive biases**: L1 → sparse solutions. KL divergence → mode-covering. Reverse KL → mode-seeking. Contrastive → push/pull in embedding space.

**4. Composite losses are the norm**: Real systems combine multiple losses. VAE = reconstruction + KL. PPO = policy + value + entropy. RLHF = reward + KL penalty. Always understand what each term does and how they trade off.

**5. The RL connection**: Every loss can be viewed through the RL lens — supervised learning is RL with immediate deterministic rewards. This connection deepens with RLHF, where LLM training literally becomes RL.

---

## 5. Document Map

- **losses.md** (this file): High-level overview and taxonomy
- **losses_detailed.md**: Detailed treatment of each loss — formulas, gradients, failure modes, practical guidance
- **losses_qa.md**: Interview Q&A — concise answers for research scientist/engineer interviews
