# Losses in Deep Learning

Short, precise answers for research scientist / engineer interviews. Organized by domain to help you build a structured mental model.

---

## Table of Contents

- [[#Part 1: Core Supervised & Representation Learning]]
  - [[#Regression (MSE, MAE, Huber, Quantile)]]
  - [[#Classification (Cross-Entropy, Focal, Hinge, Label Smoothing)]]
  - [[#Contrastive & Metric Learning (InfoNCE, Triplet, VICReg)]]
  - [[#Regularization (L1, L2)]]
- [[#Part 2: Generative & Probabilistic Objectives]]
  - [[#Probabilistic (KL Divergence, ELBO)]]
  - [[#GANs (WGAN, Adversarial)]]
  - [[#Diffusion (Score Matching, Flow Matching)]]
- [[#Part 3: Reinforcement Learning & Control]]
  - [[#Policy Losses (REINFORCE, PPO, SAC)]]
  - [[#Value Losses (TD, Distributional)]]
  - [[#Multi-Agent & Quality-Diversity (PSRO, NeuPL, MAP-Elites)]]
- [[#Part 4: LLM Alignment]]
  - [[#RLHF Pipeline]]
  - [[#DPO & Variants (KTO, ORPO)]]
  - [[#GRPO (DeepSeek)]]
- [[#Part 5: Synthesis & Quick Reference]]
  - [[#Cross-Cutting Connections]]
  - [[#Quick-Fire Round]]

---

## Part 1: Core Supervised & Representation Learning

**Goal:** Map inputs to targets or learn useful embeddings.
**Key Trade-offs:** Robustness vs. Efficiency, Hard vs. Soft decisions.

### Regression (MSE, MAE, Huber, Quantile)

#### Q: What are MSE, MAE, and Huber Loss?
*   **MSE (Mean Squared Error)**: $\mathcal{L} = \frac{1}{N}\sum (y - \hat{y})^2$. Penalizes large errors **quadratically**. Differentiable everywhere.
*   **MAE (Mean Absolute Error)**: $\mathcal{L} = \frac{1}{N}\sum |y - \hat{y}|$. Penalizes errors **linearly**. Robust to outliers but gradient is undefined at 0.
*   **Huber Loss**: A hybrid. **Quadratic** for small errors ($|\delta| < 1$) to be differentiable, and **Linear** for large errors to be robust to outliers.

#### Q: When do you use MSE vs MAE vs Huber?
**MSE (L2)**: Default for clean data. Assumes Gaussian noise. Penalizes large errors accurately (large gradient). **Pitfall**: Sensitive to outliers (gradient blows up).
**MAE (L1)**: Robust to outliers. Assumes Laplace noise. Constant gradient. **Pitfall**: Gradient undefined at 0 (instability).
**Huber**: Best of both. Quadratic near 0 (smooth), linear far from 0 (robust). **Use case**: Reinforcement Learning value functions (DQN) where targets are unstable.

#### Q: What is quantile loss?
Asymmetric L1 loss. Penalizes over/under-estimation differently.
**Use case**: Predicting intervals (uncertainty) or distributional RL (QR-DQN), where you learn the full distribution of returns, not just the mean.

### Classification (Cross-Entropy, Focal, Hinge, Label Smoothing)

#### Q: What is Cross-Entropy Loss?
**Formula**: $\mathcal{L} = -\sum y \log(\hat{y})$.
Measures the difference between two probability distributions: the true labels $y$ and predicted probabilities $\hat{y}$.
For **Binary** classification (Log Loss): $- (y \log(p) + (1-y) \log(1-p))$.
**Key Property**: Heavily penalizes **confident wrong predictions**.

#### Q: Wait, isn't Binary Classification just Regression on probabilities?
**Yes, technically.**
*   **Mathematically**: You are regressing the probability $p \in [0,1]$ via a sigmoid function. "Logistic Regression" is literally in the name.
*   **Why we separate them**: We care about the **decision boundary** (thresholding probability to getting discrete classes) and use discrete metrics (Accuracy, F1).
*   **Why not use MSE?**: Regressing probabilities with MSE + Sigmoid is **non-convex** and has vanishing gradients. Cross-Entropy is convex for this problem.

#### Q: Why Cross-Entropy (CE) and not MSE?
CE ($\hat{y}-y$) provides strong gradients even when the model is wrong. MSE gradients vanish when sigmoid/softmax saturates ("vanishing gradient problem"). CE minimizes KL divergence to the true distribution.

#### Q: What is Focal Loss?
$\text{FL} = -(1-p_t)^\gamma \log(p_t)$.
**Intuition**: Down-weights "easy" examples (where $p_t \approx 1$). Forces model to focus on hard, misclassified examples.
**Use case**: Extreme class imbalance (e.g., Object Detection background vs. foreground).

#### Q: Label Smoothing: Why do we lie to the model?
Replace hard target `[0, 1]` with soft target `[0.1, 0.9]`.
**Benefit**: Prevents network from becoming overconfident (logits $\to \infty$) and overfitting. implicit calibration.

#### Q: When to use Hinge Loss?
**Margin Maximization**. Used in SVMs.
**Key difference**: Hinge loss is **0** if the point is correctly classified by a margin. CE is never 0. Use Hinge via **Uncertainty Sampling** or efficient retrieval index training.

### Contrastive & Metric Learning (InfoNCE, Triplet, VICReg)

#### Q: What is Contrastive Loss?
**Goal**: Pull positive pairs (similar items) together, push negative pairs (dissimilar items) apart in the embedding space.
**Metric Learning**: Learning a distance function $d(x, y)$ such that $d(\text{Anchor}, \text{Positive}) < d(\text{Anchor}, \text{Negative})$.

#### Q: What is Triplet Loss?
**Formula**: $\mathcal{L} = \max(d(A, P) - d(A, N) + \alpha, 0)$.
**Components**: Anchor ($A$), Positive ($P$), Negative ($N$), Margin ($\alpha$).
**Intuition**: Ensure Positive is closer to Anchor than Negative by at least margin $\alpha$. Used in FaceNet (Face Recognition).

#### Q: Explain InfoNCE.
$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, k_+)/\tau)}{\sum \exp(\text{sim}(q, k_i)/\tau)}$$
**(N-1)-way classification**: "Find the positive key among negatives".
**Key Insight**: Maximizes mutual information. Requires **large batch sizes** (SimCLR) or **momentum queues** (MoCo) for good negative samples.

#### Q: SimCLR vs BYOL vs VICReg?
*   **SimCLR**: Uses negatives (InfoNCE). Needs large batches.
*   **BYOL**: No negatives. Predicts target network representation. Relies on asymmetry (stop-gradient) to prevent collapse.
*   **VICReg**: No negatives, no momentum. Explicitly regularizes **Variance** (don't collapse), **Invariance** (match views), and **Covariance** (decorrelate features).

### Regularization (L1, L2)

#### Q: L1 vs L2 Regularization?
*   **L2 (Weight Decay)**: Gaussian prior. Shrinks all weights.
*   **L1 (Lasso)**: Laplace prior. Induces **sparsity** (sets weights to 0). Feature selection.

---

## Part 2: Generative & Probabilistic Objectives

**Goal:** Learn the data distribution $p(x)$.
**Key Trade-offs:** Mode seeking (Reverse KL) vs. Mode covering (Forward KL).

### Probabilistic (KL Divergence, ELBO)

#### Q: What is KL Divergence?
**Formula**: $D_{KL}(P || Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$.
**Intuition**: Measures strictly non-negative "distance" (not symmetric) between two distributions. 0 if identical.
**Interpretation**: "How much information is lost when we approximate $P$ with $Q$?"

#### Q: Forward ($P||Q$) vs Reverse ($Q||P$) KL?
*   **Forward KL (MLE)**: **Mode-covering**. "Don't assign 0 probability to data". Model $Q$ stretches to cover all modes of $P$.
*   **Reverse KL**: **Mode-seeking**. "Don't put mass where data isn't". Model $Q$ collapses to a single mode. Used in **Variational Inference**.

#### Q: Explain the VAE Loss (ELBO).
$$\text{ELBO} = \text{Reconstruction} - \text{KL(Posterior || Prior)}$$
1.  **Reconstruct**: Make the output look like input.
2.  **Regularize**: Keep latent space $z$ close to $N(0,1)$.
**Issue**: **Posterior Collapse** (Decoder ignores $z$, just uses autoregresion). Fix with KL annealing or "free bits".

### GANs (WGAN, Adversarial)

#### Q: Why WGAN? (Wasserstein Loss)
Original GAN loss saturates if Discriminator is too good (vanishing gradients).
**WGAN**: Uses Earth Mover's Distance. Gradient flows even when distributions are disjoint.
**Requirement**: Critic must be **1-Lipschitz** (enforced by Gradient Penalty or Spectral Norm).

### Diffusion (Score Matching, Flow Matching)

#### Q: What does a Diffusion model optimize?
**Simple answer**: MSE between **predicted noise** and **added noise**.
$$\mathcal{L} = ||\epsilon - \epsilon_\theta(x_t, t)||^2$$
**Deep answer**: It learns the **Score Function** ($\nabla_x \log p(x)$) — the gradient pointing towards higher data density.

#### Q: What is Flow Matching?
Generalizes diffusion. Instead of a curved SDE path, it learns a **straight line** vector field from Noise $\to$ Data.
**Benefit**: Faster inference, simpler training (Stable Diffusion 3, Flux).

---

## Part 3: Reinforcement Learning & Control

**Goal:** Maximize expected return.
**Key Trade-offs:** Bias vs Variance, Exploration vs Exploitation.

### Policy Losses (REINFORCE, PPO, SAC)

#### Q: REINFORCE problems?
High variance. One lucky trajectory boosts all actions in it.
**Fix**: Subtract a **Baseline** (Value function).

#### Q: How does PPO work?
**Clipped Objective**:
$$\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$$
Prevents the policy from changing too much in one step. If the update is too large, gradient is cut off. Stabilizes training.

#### Q: Which alg to choose? (PPO vs SAC)
*   **PPO**: On-policy. Stable, general-purpose (Robotics, LLMs).
*   **SAC**: Off-policy. Sample efficient (reuses data). Maximum Entropy (explores better). Best for continuous control.

### Value Losses

#### Q: Temporal Difference (TD) vs Monte Carlo?
*   **TD(0)**: Bootstraps ($r + \gamma V(s')$). Low variance, High Bias.
*   **Monte Carlo**: Full return. Unbiased, High Variance.
*   **TD($\lambda$) / GAE**: Best of both. Interpolates using $\lambda$.

#### Q: Distributional RL?
Learn the **full distribution** of returns (C51, QR-DQN), not just the mean.
**Why?**: Better representation learning, captures multimodal outcomes, risk-sensitive policies.

### Multi-Agent & Quality-Diversity

#### Q: PSRO (Policy Space Response Oracles)?
Iterative game solving.
1.  Train "Best Response" to current meta-strategy.
2.  Add to population.
3.  Solve meta-game (Nash) to get new meta-strategy.
**Analogy**: Continual Learning, but the "tasks" are opponent strategies.

#### Q: MAP-Elites / Quality-Diversity?
Don't just find the *best* solution. Find a *grid* of high-performing solutions for every niche (e.g., "fast & tall", "slow & short").
**Loss**: No standard gradient loss. Evolutionary metric: `New Cell Score > Old Cell Score`.

---

## Part 4: LLM Alignment

**Goal:** Make valid next-token predictors helpful and safe.

### RLHF Pipeline
1.  **SFT**: Fine-tune on high-quality demos.
2.  **Reward Model**: Train on pairwise preferences ($A \succ B$).
3.  **PPO**: Optimize policy against Reward Model - KL penalty.
    *   **Why KL?**: Prevents "Reward Hacking" (drifting too far from base model).

### DPO (Direct Preference Optimization)
**Insight**: Optimal policy can be solved in closed form.
**Result**: Directly optimize the policy on preference pairs (A > B).
$$\mathcal{L}_{DPO} = -\log \sigma \left( \beta \log \frac{\pi(y_w)}{\pi_{ref}(y_w)} - \beta \log \frac{\pi(y_l)}{\pi_{ref}(y_l)} \right)$$
**Benefit**: No Reward Model, No PPO, stable.

### GRPO (DeepSeek / Group Relative PO)
Used for Math/Reasoning.
**Idea**: Sample $G$ outputs. Rank them by ground-truth correctness (or minimal reward model).
**Key**: Normalize advantage **within the group**.
**Why?**: Removes need for a Value Critic network. Saves memory/compute.

---

## Part 5: Synthesis & Quick Reference

### Cross-Cutting Connections
*   **GANs $\approx$ Multi-Agent RL**: Min-Max game.
*   **Diffusion $\approx$ Denoising Autoencoders**: Learning to reverse corruption.
*   **Contrastive Learning $\approx$ Classification**: Is just classification where classes are dynamically created (instance discrimination).
*   **RLHF $\approx$ Control**: Shaping a stochastic process (LLM) to stay within safety bounds while maximizing utility.

### Quick-Fire Round

*   **Default Regression?** $\to$ Huber (Robust).
*   **Default Classification?** $\to$ Cross-Entropy + Label Smoothing.
*   **Default RL (Continuous)?** $\to$ SAC (Efficiency) or PPO (Stability).
*   **Default LLM Alignment?** $\to$ DPO (Ease) or PPO (Peak performance).
*   **Why Entropy Bonus?** $\to$ Exploration (don't collapse).
*   **Why L1?** $\to$ Sparsity.
*   **Why Spectral Norm?** $\to$ 1-Lipschitz (Stable GANs).
