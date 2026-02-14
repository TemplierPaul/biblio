# Losses in Deep Learning — Detailed Reference

---

## 1. Regression Losses

### 1.1 Mean Squared Error (MSE / L2 Loss)

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$$

**Gradient**: $\nabla_{\hat{y}} = \frac{2}{N}(\hat{y} - y)$ — linear in error, proportional to magnitude.

**Properties**:
- Equivalent to maximum likelihood under Gaussian noise: $p(y|x) = \mathcal{N}(\hat{y}, \sigma^2)$
- Penalizes large errors quadratically → sensitive to outliers
- Smooth, differentiable everywhere → stable optimization
- Unique global minimum (convex in output)

**When to use**: Clean data, Gaussian-like errors, need smooth gradients.
**When to avoid**: Heavy-tailed noise, outliers, sparse targets.

**RL connection**: TD error in value learning is MSE between predicted and target value: $\mathcal{L} = (V(s) - (r + \gamma V(s')))^2$. Every critic in actor-critic methods optimizes this.

---

### 1.2 Mean Absolute Error (MAE / L1 Loss)

$$\mathcal{L}_{\text{MAE}} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$

**Gradient**: $\nabla_{\hat{y}} = \text{sign}(\hat{y} - y)$ — constant magnitude regardless of error size.

**Properties**:
- Equivalent to maximum likelihood under Laplace noise
- Robust to outliers (constant gradient)
- Median regression (predicts conditional median, not mean)
- Non-differentiable at zero → can cause optimization instability

**When to use**: Outlier-heavy data, want median prediction, robust estimation.
**When to avoid**: Need smooth gradients, data is clean Gaussian.

---

### 1.3 Huber Loss (Smooth L1)

$$\mathcal{L}_{\text{Huber}} = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}\end{cases}$$

**Properties**:
- MSE for small errors, L1 for large errors
- Best of both: smooth near zero, robust to outliers
- $\delta$ controls transition (typically 1.0)
- Differentiable everywhere

**When to use**: Default robust regression. Used heavily in RL (DQN uses Huber loss for TD error).
**When to avoid**: When you specifically need L2 (Gaussian assumption) or L1 (median).

**RL connection**: DQN and many value-based methods use Huber instead of MSE for TD error because value targets can be noisy/non-stationary, making outlier robustness important.

---

### 1.4 Quantile Loss

$$\mathcal{L}_\tau(\hat{y}, y) = \begin{cases} \tau(y - \hat{y}) & \text{if } y \geq \hat{y} \\ (1-\tau)(\hat{y} - y) & \text{if } y < \hat{y}\end{cases}$$

**Properties**:
- Asymmetric L1: penalizes over/under-prediction differently
- $\tau = 0.5$ → median (standard L1)
- $\tau = 0.9$ → 90th percentile
- Multiple quantiles → full predictive distribution without distributional assumptions

**When to use**: Uncertainty quantification, risk-sensitive prediction, financial applications.

**RL connection**: Quantile Regression DQN (QR-DQN) and Implicit Quantile Networks (IQN) use quantile loss to learn the full distribution of returns, not just the mean. This enables risk-sensitive RL and better handling of stochastic environments.

---

### 1.5 Log-Cosh Loss

$$\mathcal{L} = \frac{1}{N}\sum_i \log(\cosh(\hat{y}_i - y_i))$$

**Properties**: Approximately MSE for small errors, L1 for large errors (like Huber but smoother). Twice differentiable everywhere — useful for methods requiring Hessians (e.g., Newton's method, Fisher information in EWC).

---

## 2. Classification Losses

### 2.1 Cross-Entropy (CE) / Negative Log-Likelihood

**Binary**:
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_i \left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

**Multi-class** (with softmax):
$$\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_i \sum_c y_{i,c} \log \hat{y}_{i,c} = -\frac{1}{N}\sum_i \log \hat{y}_{i, c^*}$$

where $c^*$ is the true class.

**Gradient** (softmax + CE): $\nabla_{z_c} = \hat{y}_c - y_c$ — elegantly simple. Gradient is just the difference between predicted probability and target.

**Properties**:
- Equivalent to MLE for categorical distribution
- KL divergence between true and predicted distributions (up to constant)
- Well-calibrated when model is expressive enough
- Can produce overconfident predictions

**When to use**: Standard classification. Default choice.
**When to avoid**: Extreme class imbalance (use focal loss), noisy labels (use label smoothing).

**RL connection**: Cross-entropy is the policy loss in behavioral cloning: $\mathcal{L}_{BC} = -\log \pi(a^*|s)$. Also appears in REINFORCE as $-\log \pi(a|s) \cdot R$ — the log-probability weighted by return.

---

### 2.2 Focal Loss

$$\mathcal{L}_{\text{focal}} = -\alpha_t (1-\hat{y}_t)^\gamma \log(\hat{y}_t)$$

**Key idea**: Down-weight easy examples, focus on hard ones.

**Parameters**:
- $\gamma$: Focusing parameter. $\gamma=0$ → standard CE. $\gamma=2$ (typical) → hard examples dominate.
- $\alpha_t$: Class balancing weight.

**Properties**:
- $(1-\hat{y}_t)^\gamma$ → well-classified examples (high $\hat{y}_t$) contribute almost nothing
- Addresses class imbalance better than simple reweighting
- Originally from object detection (RetinaNet) where background dominates

**When to use**: Class imbalance, object detection, any setting where easy negatives swamp the loss.

---

### 2.3 Label Smoothing Cross-Entropy

$$y_{\text{smooth}} = (1-\epsilon)y_{\text{hard}} + \frac{\epsilon}{C}$$

Replace one-hot targets with soft targets. Standard value: $\epsilon = 0.1$.

**Properties**:
- Prevents overconfidence (softmax saturating)
- Acts as implicit regularization
- Better calibration
- KL divergence interpretation: penalizes deviation from uniform over non-target classes

**When to use**: Noisy labels, want calibration, transformer training (standard in NLP/vision transformers).

---

### 2.4 Hinge Loss (SVM Loss)

$$\mathcal{L}_{\text{hinge}} = \max(0, 1 - y \cdot \hat{y})$$

where $y \in \{-1, +1\}$.

**Multi-class** (Weston-Watkins):
$$\mathcal{L} = \sum_{c \neq c^*} \max(0, \hat{y}_c - \hat{y}_{c^*} + \Delta)$$

**Properties**:
- Margin-based: only penalizes if margin < 1
- Sparse gradients: correctly classified examples contribute nothing
- Maximum margin classifier
- Not probabilistic (no probability interpretation)

**When to use**: When you want margin maximization, SVMs, ranking.
**When to avoid**: When you need probability estimates.

---

### 2.5 Knowledge Distillation Loss

$$\mathcal{L}_{\text{KD}} = (1-\alpha)\mathcal{L}_{\text{CE}}(y, \hat{y}) + \alpha \cdot T^2 \cdot \text{KL}(\sigma(z_T/T) \| \sigma(z_S/T))$$

- $z_T$: Teacher logits, $z_S$: Student logits
- $T$: Temperature (softens distributions)
- $\alpha$: Balance between hard labels and soft labels

**Properties**: Transfers "dark knowledge" — relative class similarities encoded in teacher's soft predictions.

**RL connection**: Policy distillation in CRL. Progress & Compress uses distillation to consolidate knowledge. In multi-agent RL, population distillation compresses a population into a single policy.

---

## 3. Probabilistic / Likelihood Losses

### 3.1 Negative Log-Likelihood (NLL)

$$\mathcal{L}_{\text{NLL}} = -\sum_i \log p_\theta(x_i)$$

**The foundational loss**: Maximizing likelihood = minimizing NLL. Cross-entropy, MSE (Gaussian), MAE (Laplace) are all special cases under specific distributional assumptions.

---

### 3.2 KL Divergence

$$D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

**Properties**:
- Asymmetric: $D_{\text{KL}}(p\|q) \neq D_{\text{KL}}(q\|p)$
- **Forward KL** $D_{\text{KL}}(p_{\text{data}}\|q_\theta)$: Mode-covering. $q$ must be nonzero wherever $p$ is nonzero → spread out.
- **Reverse KL** $D_{\text{KL}}(q_\theta\|p_{\text{data}})$: Mode-seeking. $q$ can ignore modes of $p$ → concentrated.
- Non-negative, zero iff $p=q$

**RL connection**: 
- KL penalty in RLHF: $\mathcal{L} = -R(x) + \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ prevents policy from diverging too far from reference.
- Trust region in TRPO: constrain $D_{\text{KL}}(\pi_{\text{old}} \| \pi_{\text{new}}) \leq \delta$.
- PPO's clip is a simpler approximation to KL-constrained optimization.
- In VAE: $D_{\text{KL}}(q(z|x) \| p(z))$ regularizes the latent space.

---

### 3.3 Evidence Lower Bound (ELBO) — VAE Loss

$$\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q(z|x)}[\log p(x|z)] + D_{\text{KL}}(q(z|x) \| p(z))$$

= Reconstruction loss + KL regularization

**Components**:
- **Reconstruction**: How well can we rebuild $x$ from $z$? (MSE for continuous, BCE for binary)
- **KL term**: How close is the encoder $q(z|x)$ to the prior $p(z)$? Encourages structured latent space.

**$\beta$-VAE**: $\mathcal{L} = \text{Recon} + \beta \cdot D_{\text{KL}}$. $\beta > 1$ → more disentangled latents. $\beta < 1$ → better reconstruction.

**Failure modes**:
- **Posterior collapse**: KL term dominates → encoder ignores input, decoder ignores $z$. Fix: KL annealing, free bits.
- **Blurry reconstructions**: MSE reconstruction → average over modes. Fix: perceptual loss, adversarial loss.

**RL connection**: World models (Dreamer, RSSM) use ELBO-like objectives to learn latent dynamics for planning. The reconstruction term ensures the latent captures enough state information; the KL term keeps the latent structured.

---

## 4. Contrastive and Metric Learning Losses

### 4.1 Contrastive Loss (Siamese)

$$\mathcal{L} = y \cdot d(a, b)^2 + (1-y) \cdot \max(0, m - d(a, b))^2$$

$y=1$ if same class, $y=0$ if different. $m$ is margin. Pull similar pairs together, push dissimilar pairs apart.

---

### 4.2 Triplet Loss

$$\mathcal{L} = \max(0, d(a, p) - d(a, n) + m)$$

- $a$: Anchor, $p$: Positive (same class), $n$: Negative (different class)
- Margin $m$: Minimum desired separation

**Key challenge**: Mining hard negatives. Easy negatives (already far) contribute zero gradient. Must find negatives that are close to the anchor.

**Mining strategies**: Batch-hard (hardest pos/neg per anchor), semi-hard (negative further than positive but within margin), online mining.

---

### 4.3 InfoNCE / NT-Xent (SimCLR)

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i}\exp(\text{sim}(z_i, z_k)/\tau)}$$

- $z_i, z_j$: Positive pair (two augmentations of same sample)
- Denominator: All other samples in batch (negatives)
- $\tau$: Temperature (lower → sharper, harder negatives matter more)

**Properties**:
- Softmax over similarities → log-softmax classification problem
- Lower bound on mutual information between views
- Needs large batch sizes (more negatives = tighter bound)
- Foundation of SimCLR, CLIP, MoCo

**Variants**: SupCon (supervised contrastive — uses labels), CLIP (image-text pairs), MoCo (momentum-updated negatives).

**RL connection**: CURL (Contrastive Unsupervised RL) uses InfoNCE to learn state representations for RL from pixels. SPR (Self-Predictive Representations) uses contrastive prediction of future states.

---

### 4.4 VICReg (Variance-Invariance-Covariance)

$$\mathcal{L} = \lambda \cdot v(Z) + \mu \cdot s(Z, Z') + \nu \cdot c(Z)$$

- **Variance** $v$: Prevent embedding collapse (maintain variance along each dimension)
- **Invariance** $s$: MSE between positive pair embeddings
- **Covariance** $c$: Decorrelate embedding dimensions (prevent redundancy)

**Properties**: No negatives needed (unlike InfoNCE). No momentum encoder (unlike BYOL). Explicit anti-collapse mechanisms. Simpler to understand and tune.

---

### 4.5 ArcFace / CosFace (Angular Margin)

$$\mathcal{L} = -\log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos\theta_j}}$$

Adds angular margin $m$ to the angle between feature and class center. Produces highly discriminative embeddings. Standard in face recognition and verification.

---

## 5. Adversarial / GAN Losses

### 5.1 Original GAN (Minimax)

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**Problem**: Generator gradient vanishes when discriminator is too good ($D(G(z)) \approx 0$).

**Non-saturating variant**: Generator maximizes $\mathbb{E}[\log D(G(z))]$ instead. Stronger gradients early in training.

---

### 5.2 Wasserstein GAN (WGAN)

$$\mathcal{L}_D = \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$
$$\mathcal{L}_G = -\mathbb{E}_{z \sim p_z}[D(G(z))]$$

With Lipschitz constraint on $D$ (gradient penalty or spectral normalization).

**Properties**:
- Minimizes Earth Mover's distance → meaningful loss value (correlates with sample quality)
- No mode collapse from vanishing gradients
- More stable training
- Gradient penalty: $\lambda \mathbb{E}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$

---

### 5.3 Hinge GAN Loss

$$\mathcal{L}_D = \mathbb{E}_{x \sim p_{\text{data}}}[\max(0, 1-D(x))] + \mathbb{E}_{z}[\max(0, 1+D(G(z)))]$$
$$\mathcal{L}_G = -\mathbb{E}_{z}[D(G(z))]$$

Standard in modern GANs (BigGAN, StyleGAN). Combines stability of WGAN with simplicity.

**RL connection**: GANs have a game-theoretic structure — two players (G and D) in a zero-sum game. The equilibrium is a Nash equilibrium. This connects to multi-agent RL and game theory. GAIL (Generative Adversarial Imitation Learning) directly uses GAN-style losses for inverse RL.

---

## 6. Diffusion and Score Matching Losses

### 6.1 Denoising Score Matching

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

where $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$, $\epsilon \sim \mathcal{N}(0, I)$.

**Simplified DDPM loss** ($L_{\text{simple}}$): Equally weight all timesteps. In practice, this works better than the full variational bound.

**Properties**:
- Equivalent to learning the score function $\nabla_x \log p(x)$
- Also equivalent to predicting noise, predicting $x_0$, or predicting velocity ($v$-prediction)
- Noise prediction is standard; $v$-prediction improves stability at low noise levels

**Conditioning**: Classifier-free guidance adds conditional and unconditional score estimates: $\tilde\epsilon = \epsilon_\theta(x_t, \varnothing) + w(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing))$

### 6.2 Flow Matching (Rectified Flows)

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

where $x_t = (1-t)x_0 + tx_1$ is a linear interpolation.

**Properties**: Simpler than diffusion — straight-line paths, fewer steps at inference. Foundation of Stable Diffusion 3, Flux.

---

## 7. Reinforcement Learning Losses

### 7.1 REINFORCE (Vanilla Policy Gradient)

$$\mathcal{L}_{\text{PG}} = -\mathbb{E}_{\pi_\theta}\left[\sum_t \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

where $G_t = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$ is the return.

**Gradient**: $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot G_t]$

**Properties**:
- Unbiased but extremely high variance
- On-policy only (must re-collect data after each update)
- Foundation of all policy gradient methods

**Variance reduction**:
- **Baseline**: $G_t \to G_t - b(s_t)$ where $b$ is a baseline (typically $V(s)$)
- **Advantage**: $A(s,a) = Q(s,a) - V(s)$ — how much better was action $a$ than average?
- **GAE**: $\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$ where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

---

### 7.2 Actor-Critic Loss

**Actor** (policy):
$$\mathcal{L}_{\text{actor}} = -\mathbb{E}\left[\log \pi_\theta(a|s) \cdot \hat{A}(s,a)\right]$$

**Critic** (value):
$$\mathcal{L}_{\text{critic}} = \mathbb{E}\left[(V_\phi(s) - V^{\text{target}})^2\right]$$

**Combined** (e.g., A2C):
$$\mathcal{L} = \mathcal{L}_{\text{actor}} + c_1 \mathcal{L}_{\text{critic}} - c_2 H(\pi_\theta)$$

where $H(\pi)$ is the entropy bonus encouraging exploration.

---

### 7.3 PPO (Proximal Policy Optimization)

$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \;\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the importance ratio.

**Full PPO loss**:
$$\mathcal{L} = \mathcal{L}_{\text{PPO}}^{\text{clip}} + c_1 \mathcal{L}_{\text{value}} - c_2 H(\pi)$$

**Why clipping?**: Prevents large policy updates that destabilize training. The clip creates a "trust region" — if the ratio goes outside $[1-\epsilon, 1+\epsilon]$, no further gradient signal.

**Properties**:
- Practical approximation to TRPO's KL constraint
- Robust, widely used (OpenAI Five, ChatGPT RLHF, robotics)
- $\epsilon$ typically 0.1–0.2
- Requires multiple epochs on same data (unlike REINFORCE)

**Compared to TRPO**: PPO is first-order (no Hessian), simpler to implement, similar performance. TRPO has stronger theoretical guarantees but is impractical at scale.

---

### 7.4 SAC (Soft Actor-Critic) — Maximum Entropy RL

**Actor**:
$$\mathcal{L}_\pi = \mathbb{E}_{s \sim \mathcal{B}}\left[\alpha \log \pi_\theta(a|s) - Q_\phi(s, a)\right]$$

**Critic**:
$$\mathcal{L}_Q = \mathbb{E}\left[(Q_\phi(s,a) - (r + \gamma(Q_{\bar\phi}(s',a') - \alpha \log \pi(a'|s'))))^2\right]$$

**Temperature** $\alpha$ (automatically tuned):
$$\mathcal{L}_\alpha = \mathbb{E}_{a \sim \pi}\left[-\alpha(\log \pi(a|s) + \bar{\mathcal{H}})\right]$$

where $\bar{\mathcal{H}}$ is target entropy.

**Properties**:
- Off-policy → sample efficient
- Entropy regularization → exploration + robustness
- Automatic temperature tuning → no $\alpha$ hyperparameter
- Stochastic policy → better exploration than deterministic (DDPG/TD3)

**When to use**: Continuous control, sample efficiency matters, off-policy acceptable.

---

### 7.5 TD Learning Losses

**TD(0)**:
$$\mathcal{L} = (V(s) - (r + \gamma V(s')))^2$$

**n-step TD**:
$$\mathcal{L} = \left(V(s) - \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n V(s_{t+n})\right)^2$$

**TD($\lambda$)**: Weighted average of all n-step returns: $G_t^\lambda = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}$

**Distributional RL**:
Instead of learning $\mathbb{E}[G]$, learn the full distribution of returns.

| Method | Representation | Loss |
|--------|---------------|------|
| **C51** | Fixed atoms, learned probabilities | Cross-entropy (projected Bellman) |
| **QR-DQN** | Fixed quantiles, learned atoms | Quantile Huber loss |
| **IQN** | Sampled quantiles | Quantile regression |

**Why distributional?**: Better representations, risk-sensitive decision-making, more stable training. The distribution "knows more" than the mean.

---

### 7.6 Model-Based RL Losses

**Dynamics prediction** (Dreamer, MBPO):
$$\mathcal{L}_{\text{dynamics}} = \|s_{t+1} - \hat{s}_{t+1}\|^2 + \|r_t - \hat{r}_t\|^2$$

**RSSM** (Recurrent State Space Model — Dreamer):
$$\mathcal{L}_{\text{world}} = \mathcal{L}_{\text{recon}} + \beta_1 D_{\text{KL}}(q \| p) + \beta_2 \mathcal{L}_{\text{reward}} + \beta_3 \mathcal{L}_{\text{continue}}$$

ELBO-like: the world model is a VAE over latent state sequences.

**Contrastive model learning** (SPR, CURL):
$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_{t+k}, \hat{z}_{t+k})/\tau)}{\sum_j \exp(\text{sim}(z_j, \hat{z}_{t+k})/\tau)}$$

Predict future representations contrastively rather than reconstructing pixels. More robust than pixel-level prediction.

---

### 7.7 Exploration Losses

**Curiosity / ICM** (Intrinsic Curiosity Module):
$$R^{\text{intrinsic}}_t = \|\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\|^2$$

Prediction error in learned feature space (not raw pixels, to avoid the noisy TV problem).

**RND** (Random Network Distillation):
$$R^{\text{intrinsic}}_t = \|f_\theta(s_t) - f_{\text{fixed}}(s_t)\|^2$$

$f_{\text{fixed}}$: Random, frozen network. $f_\theta$: Trained to match it. Novel states have high prediction error because $f_\theta$ hasn't seen them.

**Maximum Entropy exploration**: $\mathcal{L} = -H(\pi) = \mathbb{E}[\log \pi(a|s)]$. Encourage high-entropy policies. SAC does this implicitly.

**Count-based** (pseudo-counts): $R^I = 1/\sqrt{N(s)}$ or learned density models to estimate visitation counts in continuous spaces.

---

## 8. Multi-Agent and Game-Theoretic Losses

### 8.1 Self-Play Losses

In self-play, the "loss" is the game outcome against yourself or a population. The gradient comes from treating the opponent as the environment.

**Fictitious play**: Best respond to average opponent strategy.
**PSRO regret**: Minimize exploitability of the meta-strategy: $\epsilon(\sigma) = \max_{\pi} J(\pi, \sigma) - J(\sigma, \sigma)$.

### 8.2 NeuPL / Population-Conditioned Losses

$$\mathcal{L}_{\text{NeuPL}} = -\mathbb{E}_{\sigma \sim \Delta^K}\left[\mathbb{E}_{\pi_\theta(\cdot|\sigma)}[R(\pi_\theta, \sigma_{\text{opp}})]\right]$$

The policy is conditioned on a mixture vector $\sigma$ over the opponent population. It must simultaneously be a good response to every mixture — this is a multi-objective loss over the simplex.

**Connection to CRL**: Each mixture $\sigma$ is analogous to a "task." The network must maintain performance across all mixtures (stability) while expanding to new ones (plasticity).

### 8.3 Diversity Losses in Multi-Agent

**Behavioral diversity**: Maximize pairwise distances between policies in behavior space:
$$\mathcal{L}_{\text{div}} = -\sum_{i \neq j} d(\tau_i, \tau_j)$$

**Determinantal Point Processes (DPP)**: $\log \det(K)$ where $K_{ij} = k(\pi_i, \pi_j)$ — maximizes volume in strategy space, encouraging diverse and spread-out policies.

---

## 9. LLM Alignment Losses

### 9.1 RLHF (Reinforcement Learning from Human Feedback)

**Three stages**:
1. **SFT**: Supervised fine-tuning on demonstrations → cross-entropy loss
2. **Reward model**: Train on preference pairs $(y_w \succ y_l)$:
   $$\mathcal{L}_{\text{RM}} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$
   (Bradley-Terry model)
3. **PPO**: Optimize policy against reward model with KL penalty:
   $$\mathcal{L}_{\text{RLHF}} = -\mathbb{E}_{x \sim D, y \sim \pi_\theta}\left[r_\phi(x,y) - \beta D_{\text{KL}}(\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x))\right]$$

**KL penalty** $\beta$: Prevents "reward hacking" — the policy exploiting quirks of the reward model by diverging too far from the pre-trained distribution.

---

### 9.2 DPO (Direct Preference Optimization)

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

**Key insight**: Reparameterize the RLHF objective to eliminate the reward model entirely. The optimal policy under RLHF satisfies $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \text{const}$. Substituting this into the reward model loss gives DPO directly.

**Properties**:
- No reward model needed (simpler pipeline)
- No RL (pure supervised optimization on preference pairs)
- Equivalent to RLHF under certain conditions
- More stable training, less compute

**Failure mode**: Can overfit to preference pairs, especially with limited data. The implicit reward model may be less robust than an explicit one.

---

### 9.3 KTO (Kahneman-Tversky Optimization)

$$\mathcal{L}_{\text{KTO}} = \mathbb{E}_{(x,y) \sim D}\left[\lambda_y (1 - v_{\text{KTO}}(x, y; \beta))\right]$$

where $v_{\text{KTO}}$ applies a sigmoid to the reward difference weighted by loss aversion.

**Key insight**: Doesn't need paired preferences (win/lose on same prompt). Works with unpaired binary feedback: "this output is good" or "this output is bad."

**Properties**: More data-efficient (don't need same prompt with two outputs), handles asymmetric preferences (loss aversion).

---

### 9.4 GRPO (Group Relative Policy Optimization)

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_{x}\left[\frac{1}{G}\sum_{i=1}^G \min\left(r_i(\theta)\hat{A}_i, \;\text{clip}(r_i, 1-\epsilon, 1+\epsilon)\hat{A}_i\right)\right] + \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

where advantages are computed relative to the group: $\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$.

**Key insight** (DeepSeek-R1): No critic network needed. Sample $G$ outputs per prompt, score them, normalize scores within the group. The group provides its own baseline.

**Properties**: Simpler than PPO+critic (no value network), scales well, used in DeepSeek-R1 for math/reasoning.

---

### 9.5 SimPO (Simple Preference Optimization)

$$\mathcal{L}_{\text{SimPO}} = -\log\sigma\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)$$

**Key changes from DPO**: Length-normalized log-probabilities (prevents length bias) + margin $\gamma$ (minimum quality gap). No reference model needed.

---

### 9.6 ORPO (Odds Ratio Preference Optimization)

$$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w) + \lambda \cdot \mathcal{L}_{\text{OR}}$$

Combines SFT on preferred output with an odds ratio penalty on dispreferred output. No reference model needed. Single training stage (SFT + alignment simultaneously).

---

## 10. Quality-Diversity and Open-Endedness Losses

### 10.1 MAP-Elites Fitness

$$\text{Objective: maximize } f(x) \text{ within each cell of behavior space}$$

Not a differentiable loss — operates via evolutionary selection. Each cell in the behavior archive stores the highest-fitness solution with that behavioral characterization.

**QD-score**: $\sum_{\text{filled cells}} f(x_{\text{cell}})$ — total quality across all occupied niches.

### 10.2 Differentiable Quality-Diversity (DQD)

$$\mathcal{L}_{\text{DQD}} = -f(x) + \lambda \cdot \text{archive\_improvement}(x)$$

Use gradient information to search for both high-fitness and novel solutions. CMA-MEGA combines CMA-ES with gradient-based QD.

### 10.3 Novelty Search

$$\text{Novelty}(x) = \frac{1}{k}\sum_{j=1}^k d(b(x), b(x_j))$$

Average distance to $k$ nearest neighbors in behavior space. Ignores fitness entirely — pure exploration.

**Novelty + Fitness**: $\text{Score}(x) = (1-\alpha) \cdot \text{novelty}(x) + \alpha \cdot f(x)$

### 10.4 AURORA (Autonomous Representations for Open-ended learning)

Learns behavior descriptors via autoencoder rather than hand-defining them:
$$\mathcal{L}_{\text{AURORA}} = \mathcal{L}_{\text{recon}} + \lambda \mathcal{L}_{\text{QD}}$$

The representation loss ensures meaningful behavioral characterizations; QD fills the learned archive.

### 10.5 Adversarial / Regret-Based Open-Endedness

**PAIRED** (Protagonist Antagonist Induced Regret Environment Design):
$$\mathcal{L}_{\text{env}} = V^{\text{protagonist}} - V^{\text{antagonist}}$$

The environment generator maximizes the regret between a protagonist and antagonist agent. Creates environments at the "frontier of learning" — not too easy, not impossible.

**Connection**: This is a multi-agent game where the loss is defined over the gap between agents, driving open-ended curriculum generation.

---

## 11. Regularization Losses

### 11.1 Weight Regularization

**L2 (Ridge / Weight Decay)**: $\lambda \|\theta\|_2^2$ — Gaussian prior on weights, shrinks toward zero.

**L1 (Lasso)**: $\lambda \|\theta\|_1$ — Laplace prior, induces sparsity (some weights exactly zero).

**Elastic Net**: $\lambda_1 \|\theta\|_1 + \lambda_2 \|\theta\|_2^2$ — combines both.

### 11.2 Entropy Regularization

$$\mathcal{L}_{\text{entropy}} = -H(\pi) = \mathbb{E}[\log \pi(a|s)]$$

Discourages premature policy collapse in RL. Higher entropy → more exploration. SAC makes this a first-class objective. PPO adds it as an auxiliary loss.

### 11.3 Spectral Normalization

Constrains the spectral norm (largest singular value) of weight matrices to 1. Enforces Lipschitz continuity. Essential for WGAN discriminator, useful for training stability generally.

### 11.4 EWC (as Regularization)

$$\mathcal{L}_{\text{EWC}} = \frac{\lambda}{2}\sum_i F_i(\theta_i - \theta^*_i)^2$$

Fisher Information-weighted L2 regularization toward old parameters. See Section 7 of CRL notes — this is a continual learning regularizer.

### 11.5 Dropout as Implicit Regularization

Not a loss term per se, but equivalent to approximate Bayesian inference / ensemble averaging. At training: randomly zero out activations. At inference: scale or use Monte Carlo dropout for uncertainty.

---

## 12. Self-Supervised and Auxiliary Losses

### 12.1 Next-Token Prediction (Autoregressive LM)

$$\mathcal{L}_{\text{LM}} = -\sum_t \log p_\theta(x_t | x_{<t})$$

The foundational LLM pre-training loss. Cross-entropy on next-token prediction. GPT, LLaMA, and all autoregressive LMs use this.

### 12.2 Masked Language Modeling (BERT)

$$\mathcal{L}_{\text{MLM}} = -\sum_{t \in \text{masked}} \log p_\theta(x_t | x_{\backslash t})$$

Predict randomly masked tokens from context. Bidirectional (unlike autoregressive). Foundation of BERT, RoBERTa.

### 12.3 Masked Image Modeling (MAE)

$$\mathcal{L}_{\text{MAE}} = \|x_{\text{masked}} - \hat{x}_{\text{masked}}\|^2$$

Mask 75% of image patches, reconstruct from remaining 25%. MSE on pixel values of masked patches. Surprisingly effective pre-training for vision.

### 12.4 CLIP (Contrastive Language-Image Pre-training)

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2}(\mathcal{L}_{\text{i2t}} + \mathcal{L}_{\text{t2i}})$$

Symmetric InfoNCE: image-to-text and text-to-image matching. Creates aligned multimodal embedding space. Foundation of text-to-image generation, zero-shot classification.

### 12.5 Auxiliary RL Losses

**Reward prediction**: Predict reward from state-action pair (extra supervision).
**Next-state prediction**: Predict $s_{t+1}$ from $(s_t, a_t)$ (learns dynamics implicitly).
**Inverse dynamics**: Predict $a_t$ from $(s_t, s_{t+1})$ (focuses representation on controllable aspects).
**Temporal contrastive**: States close in time should be close in representation.

These improve representation quality in RL and are especially useful with pixel observations.

---

## 13. Practical Guidance: Choosing Losses

### By Problem Type

| Problem | Default Loss | Upgrade When... |
|---------|-------------|----------------|
| Regression | MSE | Outliers → Huber. Need uncertainty → Quantile. Sparse targets → L1. |
| Binary classification | BCE | Imbalanced → Focal. Noisy labels → Label Smoothing. |
| Multi-class | Cross-entropy | Imbalanced → Focal. Calibration → Label Smoothing. |
| Embedding / retrieval | Triplet | Scale → InfoNCE. Labels → SupCon. |
| Generation (images) | Diffusion (DDPM) | Speed → Flow Matching. Specific control → Conditional. |
| Generation (text) | Autoregressive CE | Alignment → DPO/RLHF. |
| RL (discrete) | DQN (Huber TD) | Distributional → QR-DQN. Multi-agent → PSRO. |
| RL (continuous) | SAC | On-policy needed → PPO. Simple → TD3. |
| RL (alignment) | PPO + reward model | Simplify → DPO. No critic → GRPO. Unpaired → KTO. |
| Continual learning | Replay + current loss | Regularize → EWC. Architecture → PNN. Plasticity → Reset. |
| Open-endedness | QD-score | Learned descriptors → AURORA. Curriculum → PAIRED. |

### Common Pitfalls

1. **MSE for classification**: Gradients are wrong — use CE.
2. **CE without class weights for imbalanced data**: Majority class dominates — use focal loss or balanced sampling.
3. **Triplet loss without hard mining**: Most triplets are trivial → zero gradient → no learning.
4. **Forgetting the KL penalty in RLHF**: Reward hacking — policy exploits reward model.
5. **Ignoring entropy in RL**: Premature convergence to suboptimal deterministic policy.
6. **MSE for value function with non-stationary targets**: Use Huber loss, target networks, or clipped value loss.
7. **$\beta$-VAE with $\beta$ too high**: Posterior collapse — KL term overwhelms reconstruction.

### Composing Losses

Most real systems combine multiple losses. Guidelines:

- **Scale terms to similar magnitudes** before weighting (losses on different scales → one dominates).
- **Monitor each term independently** — if one term drops to zero, it's not contributing.
- **Anneal weights** when terms have different convergence rates (e.g., KL annealing in VAE).
- **Use stop-gradient** when one term shouldn't backprop through part of the computation (e.g., target networks in TD learning, EMA in self-supervised learning).
