# Losses in Deep Learning

Short, precise answers for research scientist / engineer interviews in ML, RL, open-endedness, and LLMs.

---

## Table of Contents

- [[#Regression]]
  - [[#Q: When do you use MSE vs MAE vs Huber?]]
  - [[#Q: What is quantile loss and when would you use it?]]
  - [[#Q: What's the probabilistic interpretation of MSE and MAE?]]
- [[#Classification]]
  - [[#Q: Why cross-entropy and not MSE for classification?]]
  - [[#Q: What is focal loss?]]
  - [[#Q: What does label smoothing do?]]
  - [[#Q: When would you use hinge loss over cross-entropy?]]
  - [[#Q: How does knowledge distillation loss work?]]
- [[#Probabilistic / Information-Theoretic]]
  - [[#Q: What's the difference between forward and reverse KL?]]
  - [[#Q: Explain the VAE loss (ELBO).]]
  - [[#Q: How does the ELBO relate to RL?]]
- [[#Contrastive / Metric Learning]]
  - [[#Q: Explain InfoNCE.]]
  - [[#Q: What's the triplet loss mining problem?]]
  - [[#Q: How does VICReg avoid collapse without negatives?]]
  - [[#Q: SimCLR vs BYOL vs VICReg — what's the difference?]]
- [[#GAN Losses]]
  - [[#Q: Why did WGAN improve on the original GAN loss?]]
  - [[#Q: What's the GAN-RL connection?]]
- [[#Diffusion / Score Matching]]
  - [[#Q: What loss does a diffusion model optimize?]]
  - [[#Q: What's flow matching and why does it matter?]]
- [[#RL Policy Losses]]
  - [[#Q: Explain REINFORCE and its problems.]]
  - [[#Q: How does PPO work? Why the clip?]]
  - [[#Q: PPO vs TRPO vs SAC — when to use each?]]
  - [[#Q: What's the entropy bonus in RL and why does it matter?]]
  - [[#Q: Explain GAE (Generalized Advantage Estimation).]]
- [[#RL Value Losses]]
  - [[#Q: What's the difference between TD(0), n-step, and TD(λ)?]]
  - [[#Q: What is distributional RL and why does it help?]]
  - [[#Q: Why use Huber loss instead of MSE for TD error?]]
- [[#Multi-Agent and Game Theory]]
  - [[#Q: What loss does PSRO optimize?]]
  - [[#Q: How does the NeuPL loss work?]]
  - [[#Q: How does diversity factor into multi-agent losses?]]
- [[#LLM Alignment]]
  - [[#Q: Walk me through the RLHF pipeline.]]
  - [[#Q: How does DPO eliminate the reward model?]]
  - [[#Q: What's GRPO and why does DeepSeek use it?]]
  - [[#Q: DPO vs KTO vs SimPO vs ORPO — when to use each?]]
- [[#Quality-Diversity and Open-Endedness]]
  - [[#Q: What loss does MAP-Elites optimize?]]
  - [[#Q: What's novelty search?]]
  - [[#Q: What's PAIRED and how does it connect to RL?]]
- [[#Regularization]]
  - [[#Q: L1 vs L2 regularization?]]
  - [[#Q: What's spectral normalization and why does WGAN need it?]]
- [[#Cross-Cutting / Synthesis]]
  - [[#Q: How does cross-entropy connect supervised learning to RL?]]
  - [[#Q: How does the VAE loss connect to world models in RL?]]
  - [[#Q: What's the connection between GAN training and multi-agent RL?]]
  - [[#Q: Name a loss from each major family and say when you'd use it.]]
  - [[#Q: What's the single most important principle for choosing a loss?]]
- [[#Quick-Fire Round]]

---

## Regression

### Q: When do you use MSE vs MAE vs Huber?

**MSE**: Default for clean data. Assumes Gaussian noise. Predicts conditional mean. Gradient proportional to error size — large errors get large updates. Sensitive to outliers.

**MAE**: Robust to outliers. Predicts conditional median. Constant gradient magnitude — doesn't blow up on outliers. Non-differentiable at zero can cause instability.

**Huber**: Best of both — MSE near zero (smooth gradients), MAE far from zero (robust to outliers). Default in RL value learning (DQN uses it). Hyperparameter $\delta$ controls transition.

**One-liner**: Clean data → MSE. Outliers → Huber. Need median → MAE.

### Q: What is quantile loss and when would you use it?

Asymmetric L1: penalizes over-prediction and under-prediction differently based on quantile $\tau$. At $\tau=0.5$ it's MAE (median). At $\tau=0.9$ it targets the 90th percentile.

**Use for**: Uncertainty quantification, risk-sensitive prediction, distributional RL (QR-DQN learns full return distribution via quantile regression).

### Q: What's the probabilistic interpretation of MSE and MAE?

MSE = maximum likelihood under Gaussian noise. MAE = maximum likelihood under Laplace noise. This matters because it tells you which distributional assumption you're making — and when that assumption is wrong, your loss is suboptimal.

---

## Classification

### Q: Why cross-entropy and not MSE for classification?

Cross-entropy gives gradient $\hat{y} - y$ (through softmax), which is always well-behaved. MSE gives gradients that vanish when the sigmoid/softmax saturates — learning stalls when the model is confidently wrong, which is exactly when it needs to learn most. CE also has information-theoretic justification (minimizes KL to true distribution).

### Q: What is focal loss?

$\mathcal{L}_{\text{focal}} = -(1-\hat{y}_t)^\gamma \log(\hat{y}_t)$. Down-weights easy, well-classified examples by factor $(1-\hat{y}_t)^\gamma$. With $\gamma=2$, an example classified at 0.9 confidence gets $100\times$ less weight than one at 0.5. Designed for extreme class imbalance (RetinaNet: foreground vs background in object detection).

### Q: What does label smoothing do?

Replaces one-hot target $[0,0,1,0]$ with soft target $[0.025, 0.025, 0.925, 0.025]$ (for $\epsilon=0.1$, 4 classes). Prevents overconfidence (softmax from saturating), improves calibration, acts as regularization. Standard in transformer training.

### Q: When would you use hinge loss over cross-entropy?

Hinge loss for margin maximization (SVMs, ranking). It has zero gradient once the margin is satisfied — only updates on violations. CE always provides gradient signal. Use hinge when you care about decision boundary quality, CE when you care about calibrated probabilities.

### Q: How does knowledge distillation loss work?

Train student to match teacher's soft predictions (logits / $T$-scaled probabilities), not just hard labels. Soft targets encode relative class similarities ("this 7 looks like a 1 more than a 3"). Loss = $\alpha \cdot T^2 \cdot \text{KL}(p_T \| p_S) + (1-\alpha) \cdot \text{CE}(y, p_S)$. Temperature $T$ softens distributions to reveal dark knowledge.

**RL connection**: Policy distillation in continual RL (Progress & Compress). Population distillation in multi-agent RL.

---

## Probabilistic / Information-Theoretic

### Q: What's the difference between forward and reverse KL?

**Forward KL** $D_{\text{KL}}(p \| q)$: Mode-covering. $q$ must put mass everywhere $p$ does. Leads to spread-out approximations that overestimate uncertainty. Used in variational inference (ELBO).

**Reverse KL** $D_{\text{KL}}(q \| p)$: Mode-seeking. $q$ can collapse to a single mode of $p$. Leads to concentrated approximations. Used in policy optimization, expectation propagation.

**Interview tip**: If asked which KL to use — forward KL for safety (don't miss any modes), reverse KL for precision (commit to one good mode).

### Q: Explain the VAE loss (ELBO).

$\mathcal{L} = -\mathbb{E}_{q(z|x)}[\log p(x|z)] + D_{\text{KL}}(q(z|x) \| p(z))$

First term: reconstruction quality (how well decoder rebuilds input from latent). Second term: latent regularization (push encoder toward prior, typically standard normal).

**Posterior collapse**: KL term dominates → encoder outputs the prior regardless of input → decoder ignores $z$. Fix with KL annealing (start $\beta=0$, increase to 1), free bits (minimum KL per dimension), or more expressive decoders.

**$\beta$-VAE**: Weight KL by $\beta$. $\beta > 1$ → more disentangled but blurrier. $\beta < 1$ → sharper but tangled latents.

### Q: How does the ELBO relate to RL?

World models (Dreamer) use ELBO to learn latent dynamics. The reconstruction term ensures the latent state captures enough info for prediction. The KL term keeps latent dynamics structured and compressible. The agent then plans in latent space, avoiding expensive real interactions.

---

## Contrastive / Metric Learning

### Q: Explain InfoNCE.

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i}\exp(\text{sim}(z_i, z_k)/\tau)}$$

Treat it as $(N-1)$-way classification: "which of these is my positive pair?" Numerator: similarity to positive. Denominator: similarity to all (positives + negatives). Temperature $\tau$ controls sharpness.

**Properties**: Lower bound on mutual information. Needs large batch sizes (more negatives → tighter bound). Foundation of SimCLR, CLIP, MoCo.

### Q: What's the triplet loss mining problem?

Most random triplets are trivial (negative already far from anchor) → zero gradient → no learning. Must mine hard negatives (close to anchor but wrong class). Strategies: batch-hard (hardest per anchor), semi-hard (violates margin but not ordering), online mining. Without proper mining, triplet loss trains very slowly.

### Q: How does VICReg avoid collapse without negatives?

Three explicit terms: **Variance** (maintain per-dimension variance above threshold — prevents collapse to a point), **Invariance** (MSE between positive pair — learn shared features), **Covariance** (decorrelate dimensions — prevents collapse to a line/plane). No negatives needed, no momentum encoder.

### Q: SimCLR vs BYOL vs VICReg — what's the difference?

**SimCLR**: InfoNCE with negatives from batch. Needs large batches.
**BYOL**: Predicts one view from another using momentum encoder. No negatives. Works via asymmetry (stop gradient on target).
**VICReg**: Explicit variance/covariance regularization. No negatives, no momentum. Most interpretable.

All learn good representations. VICReg is simplest to understand; SimCLR gives best results with enough batch size; BYOL works well with smaller batches.

---

## GAN Losses

### Q: Why did WGAN improve on the original GAN loss?

Original GAN: discriminator outputs probability → once $D$ is perfect, $\log(1 - D(G(z)))$ saturates → generator gets no gradient. Training collapses.

WGAN: discriminator outputs unbounded score (critic) + Lipschitz constraint → minimizes Earth Mover's distance → meaningful loss that correlates with generation quality → stable gradients even when critic is strong. Trade-off: need to enforce Lipschitz (weight clipping, gradient penalty, or spectral norm).

### Q: What's the GAN-RL connection?

GANs are two-player zero-sum games — same mathematical framework as competitive multi-agent RL. The Nash equilibrium of the GAN game corresponds to the generator matching the data distribution. GAIL (Generative Adversarial Imitation Learning) directly uses GAN loss for inverse RL: discriminator distinguishes expert from agent trajectories, agent tries to fool it.

---

## Diffusion / Score Matching

### Q: What loss does a diffusion model optimize?

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

Predict the noise $\epsilon$ that was added to the clean image $x_0$ to get the noisy image $x_t$. This is equivalent to learning the score function $\nabla_x \log p(x)$ (denoising score matching). The model learns to denoise at every noise level.

**Variants**: Predict noise ($\epsilon$-prediction, standard), predict clean image ($x_0$-prediction), predict velocity ($v$-prediction, better at low noise).

### Q: What's flow matching and why does it matter?

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$$

Learns velocity of a linear interpolation between noise and data. Simpler than diffusion (straight paths, no noise schedules). Faster inference (fewer steps). Foundation of Stable Diffusion 3, Flux.

---

## RL Policy Losses

### Q: Explain REINFORCE and its problems.

$$\nabla J = \mathbb{E}[\nabla \log \pi(a|s) \cdot G_t]$$

Push up probability of actions that got high return, push down those that got low return. Unbiased gradient estimate.

**Problem**: Extremely high variance. If all returns are positive (common), all actions get pushed up — just by different amounts. Very sample-inefficient.

**Fix**: Subtract baseline $b(s)$ (typically $V(s)$): $\nabla J = \mathbb{E}[\nabla \log \pi(a|s) \cdot (G_t - b(s))]$. Still unbiased, much lower variance. This gives you advantage $A(s,a) = G_t - V(s)$.

### Q: How does PPO work? Why the clip?

$$\mathcal{L} = -\mathbb{E}\left[\min\left(r(\theta)\hat{A}, \;\text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)\hat{A}\right)\right]$$

where $r(\theta) = \pi_\theta(a|s) / \pi_{\text{old}}(a|s)$.

**The clip prevents catastrophic policy updates.** If the importance ratio goes outside $[1-\epsilon, 1+\epsilon]$, no further gradient — prevents the policy from changing too much in one step. This is a first-order approximation to TRPO's KL constraint. Much simpler to implement, similarly effective.

**Full loss**: $\mathcal{L}_{\text{PPO}} + c_1 \mathcal{L}_{\text{value}} - c_2 H(\pi)$. Value loss trains critic. Entropy bonus prevents premature convergence.

### Q: PPO vs TRPO vs SAC — when to use each?

**PPO**: Default for on-policy. Simple, robust, scales well. Used in RLHF (ChatGPT), robotics, games. Hyperparameter-friendly.

**TRPO**: Stronger theoretical guarantees (monotonic improvement). But requires Hessian-vector products → complex, slow. Use when guaranteed improvement matters more than speed.

**SAC**: Off-policy, entropy-regularized. Much more sample efficient than PPO (reuses data). Default for continuous control when interactions are expensive. Automatic temperature tuning.

**One-liner**: Limited data → SAC. Scale/simplicity → PPO. Theoretical guarantees → TRPO.

### Q: What's the entropy bonus in RL and why does it matter?

$-c_2 H(\pi) = c_2 \mathbb{E}[\log \pi(a|s)]$. Penalizes low-entropy (deterministic) policies. Without it, policy collapses to a single action early → explores poorly → gets stuck in local optima. SAC makes entropy a first-class objective (maximum entropy RL). PPO adds it as auxiliary loss.

**Practical effect**: Agent maintains stochasticity → explores more → finds better policies. Temperature/coefficient $c_2$ controls exploration-exploitation.

### Q: Explain GAE (Generalized Advantage Estimation).

$$\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

$\lambda$ interpolates between TD(0) advantage ($\lambda=0$, low variance, high bias) and Monte Carlo advantage ($\lambda=1$, high variance, low bias). Typical $\lambda=0.95$.

**Why it matters**: The quality of advantage estimates directly controls the quality of policy gradient. GAE is the standard way to compute advantages in PPO and A3C.

---

## RL Value Losses

### Q: What's the difference between TD(0), n-step, and TD(λ)?

**TD(0)**: $V^{\text{target}} = r + \gamma V(s')$. Low variance (one-step bootstrap). High bias (depends on current $V$ accuracy). Fast.

**n-step**: $V^{\text{target}} = \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n V(s_{t+n})$. More real rewards → less bias. More variance. Trade-off via $n$.

**TD($\lambda$)**: Weighted average of ALL n-step returns. $\lambda=0$ → TD(0). $\lambda=1$ → Monte Carlo. Smooth bias-variance control.

### Q: What is distributional RL and why does it help?

Instead of learning $\mathbb{E}[G]$ (scalar), learn the full distribution of returns.

**C51**: Fixed atoms $z_i$, learn probabilities $p_i$. Loss: cross-entropy after projecting Bellman update onto atoms.
**QR-DQN**: Fixed quantiles $\tau_i$, learn atom locations. Loss: quantile Huber loss.
**IQN**: Sample quantiles at random, learn atom locations. Most flexible.

**Why it helps**: Better representations (distribution is richer signal than mean). Risk-sensitive decisions. More stable training. State-of-the-art on Atari.

### Q: Why use Huber loss instead of MSE for TD error?

TD targets $r + \gamma V(s')$ are non-stationary (change as $V$ improves) and noisy (stochastic rewards, bootstrap error). MSE gradient is proportional to error magnitude → large TD errors cause huge, destabilizing updates. Huber clips the gradient at $\delta$, preventing catastrophic updates from outlier transitions.

---

## Multi-Agent and Game Theory

### Q: What loss does PSRO optimize?

Each iteration: train a best response $\pi_{\text{BR}}$ against the current meta-strategy $\sigma$. The "loss" is the negated expected return against $\sigma$: $\mathcal{L} = -\mathbb{E}_\sigma[J(\pi_{\text{BR}}, \sigma)]$. Then update $\sigma$ via a meta-solver (e.g., Nash, $\alpha$-rank). The population-level objective is minimizing exploitability.

### Q: How does the NeuPL loss work?

A single policy network conditioned on mixture $\sigma \in \Delta^K$: $\pi_\theta(a|s,\sigma)$. For each sampled $\sigma$, optimize expected return against the corresponding opponent: $\mathcal{L} = -\mathbb{E}_{\sigma}[J(\pi_\theta(\cdot|\sigma), \sigma_{\text{opp}})]$. This is a multi-task RL problem over the simplex — every mixture is a "task." The CRL connection is direct: maintaining competence across all $\sigma$ IS continual learning.

### Q: How does diversity factor into multi-agent losses?

**Behavioral diversity loss**: Maximize pairwise behavioral distances. Prevents population collapse to single strategy.

**DPP-based**: Maximize $\log \det(K)$ where $K_{ij}$ measures similarity between policies $i$ and $j$. Encourages spread in strategy space.

**Quality-Diversity connection**: MAP-Elites fills behavior niches with high-fitness solutions. In MARL, "behavior niches" are strategic roles, "fitness" is win rate.

---

## LLM Alignment

### Q: Walk me through the RLHF pipeline.

**Stage 1 — SFT**: Fine-tune base LLM on demonstrations. Loss: cross-entropy (next-token prediction).

**Stage 2 — Reward Model**: Train on preference pairs $(y_w \succ y_l | x)$. Loss: $-\log\sigma(r(x,y_w) - r(x,y_l))$ (Bradley-Terry model).

**Stage 3 — PPO**: Optimize $\pi_\theta$ to maximize reward model score minus KL penalty: $\max_\theta \mathbb{E}[r_\phi(x,y) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})]$.

**KL penalty is critical**: Without it, $\pi_\theta$ "reward hacks" — exploits quirks in $r_\phi$ by generating degenerate text that scores high.

### Q: How does DPO eliminate the reward model?

**Key insight**: Under the RLHF objective, the optimal policy satisfies $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + C$. Substitute this into the Bradley-Terry preference model and you get:

$$\mathcal{L}_{\text{DPO}} = -\log\sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

No reward model, no RL — purely supervised on preference pairs. Simpler, more stable, less compute.

**Limitation**: The implicit reward model is less flexible than an explicit one. Can overfit with limited preference data.

### Q: What's GRPO and why does DeepSeek use it?

**GRPO** (Group Relative Policy Optimization): Sample $G$ outputs per prompt, score them with a reward function, normalize scores within the group to get advantages: $\hat{A}_i = (r_i - \bar{r}) / \text{std}(r)$. Then apply PPO-style clipped objective.

**Why no critic**: The group average IS the baseline. No need for a separate value network. Simpler, cheaper, scales well.

**Why DeepSeek uses it**: For math/reasoning, the reward signal is binary (correct/incorrect) and can be verified. Group normalization works well here — within a batch of attempts, some succeed and some fail, providing contrastive signal.

### Q: DPO vs KTO vs SimPO vs ORPO — when to use each?

**DPO**: Need paired preferences (win/lose on same prompt). Standard choice. Needs reference model.

**KTO**: Unpaired binary feedback ("good" or "bad" outputs, not necessarily on same prompt). More data-efficient collection. Models loss aversion.

**SimPO**: Like DPO but reference-free and length-normalized. Simpler. Prevents length gaming.

**ORPO**: Single-stage (SFT + alignment simultaneously). Fastest pipeline. Reference-free.

**One-liner**: Have paired prefs → DPO. Unpaired → KTO. Want simplicity → SimPO/ORPO. Need strongest results → RLHF with PPO (still slightly ahead on hardest benchmarks, but gap shrinking).

---

## Quality-Diversity and Open-Endedness

### Q: What loss does MAP-Elites optimize?

Not a differentiable loss — evolutionary selection. Maintain an archive of behavior cells. Each cell stores the highest-fitness solution with that behavioral characterization. Objective: maximize QD-score = $\sum_{\text{filled cells}} f(x)$.

**Differentiable QD** (CMA-MEGA): Use gradients of both fitness and behavior descriptors to search more efficiently. Still fills an archive, but uses gradient info for proposals.

### Q: What's novelty search?

$\text{novelty}(x) = \frac{1}{k}\sum_{j=1}^k d(b(x), b(x_j))$ — distance to $k$-nearest neighbors in behavior space. Ignores fitness entirely — just rewards being different. Can discover stepping stones to hard objectives that fitness-based search can't reach.

**Novelty + fitness**: Weighted combination. Quality-diversity interpolation.

### Q: What's PAIRED and how does it connect to RL?

$\mathcal{L}_{\text{env}} = V^{\text{protagonist}} - V^{\text{antagonist}}$. An environment generator creates environments that maximize regret between a protagonist and a weaker antagonist. This produces an automatic curriculum at the frontier of learning — environments that are hard for the protagonist but not impossible (because the antagonist must also be able to make progress).

**RL connection**: All three components (generator, protagonist, antagonist) are RL agents. The loss is defined over the gap between agents' value functions. Multi-agent game theory driving open-ended learning.

---

## Regularization

### Q: L1 vs L2 regularization?

**L2** ($\|\theta\|_2^2$): Gaussian prior. Shrinks all weights toward zero uniformly. Smooth. Equivalent to weight decay (with appropriate optimizer).

**L1** ($\|\theta\|_1$): Laplace prior. Induces sparsity — some weights go exactly to zero. Useful for feature selection. Non-smooth at zero.

**Elastic Net**: Both. Gets sparsity from L1 plus stability from L2.

### Q: What's spectral normalization and why does WGAN need it?

Divides weight matrices by their largest singular value, constraining spectral norm to 1. This enforces 1-Lipschitz continuity on the network. WGAN's theoretical guarantee requires a Lipschitz critic — spectral norm is the cleanest way to enforce this (better than weight clipping, which distorts the function class).

---

## Cross-Cutting / Synthesis

### Q: How does cross-entropy connect supervised learning to RL?

Behavioral cloning: $\mathcal{L}_{BC} = -\log \pi(a^*|s)$ — pure cross-entropy on expert actions. REINFORCE: $\mathcal{L} = -\log \pi(a|s) \cdot R$ — cross-entropy weighted by return. When $R = 1$ for expert actions, REINFORCE reduces to behavioral cloning. The entire spectrum from supervised learning to RL is unified through the log-probability.

### Q: How does the VAE loss connect to world models in RL?

World models (Dreamer) learn $p(s_{t+1} | s_t, a_t)$ in latent space using ELBO: reconstruction loss ensures latent captures useful state information, KL regularization keeps dynamics smooth and structured. The agent then does imagined rollouts in latent space (planning) — the loss quality directly determines planning quality.

### Q: What's the connection between GAN training and multi-agent RL?

Both are games. GAN: generator vs discriminator, zero-sum, convergence to Nash equilibrium = generator matches data. MARL: agents competing/cooperating, convergence to Nash/correlated equilibrium. GAIL makes this explicit — uses GAN loss as inverse RL objective. The training dynamics (cycling, mode collapse, instability) are shared pathologies.

### Q: Name a loss from each major family and say when you'd use it.

- **Regression**: Huber (default robust regression, RL value learning)
- **Classification**: Cross-entropy (standard), Focal (imbalanced)
- **Contrastive**: InfoNCE (representation learning, CLIP, CURL)
- **Generative**: ELBO (VAE, world models), denoising score matching (diffusion)
- **RL policy**: PPO clip (on-policy, alignment), SAC (off-policy continuous)
- **RL value**: Distributional (QR-DQN for risk-sensitive, C51 for Atari)
- **Alignment**: DPO (preference-based LLM alignment)
- **QD**: MAP-Elites QD-score (diverse solution discovery)
- **Regularization**: KL penalty (RLHF, trust regions), entropy (exploration)

### Q: What's the single most important principle for choosing a loss?

**Match the loss to the output distribution and problem structure.** MSE assumes Gaussian noise. CE assumes categorical outputs. Quantile loss makes no distributional assumptions. Huber handles heavy tails. If your loss assumes the wrong thing about your data, no amount of architecture or training tricks will fix it.

Second principle: **gradient behavior matters more than loss value.** You're not minimizing the loss directly — you're following gradients. A loss with good gradient properties (informative everywhere, bounded, smooth) will train better than one with a "better" objective but pathological gradients.

---

## Quick-Fire Round

**Q: Default regression loss?** Huber (unless you know noise is Gaussian, then MSE).

**Q: Default classification loss?** Cross-entropy with label smoothing.

**Q: Default RL continuous control?** SAC (off-policy) or PPO (on-policy).

**Q: Default LLM alignment?** DPO (simple) or PPO with reward model (strongest).

**Q: Default self-supervised vision?** MAE or DINOv2.

**Q: Default self-supervised text?** Next-token prediction (autoregressive).

**Q: Default contrastive?** InfoNCE with cosine similarity and temperature 0.07.

**Q: Default generative images?** Diffusion (DDPM) or Flow Matching.

**Q: Why entropy bonus in RL?** Prevents premature policy collapse, maintains exploration.

**Q: Why KL penalty in RLHF?** Prevents reward hacking (policy exploiting reward model artifacts).

**Q: Why Huber not MSE in DQN?** Non-stationary targets cause outlier TD errors; Huber clips gradient.

**Q: Why behavior cloning on replayed data (CLEAR)?** More stable than re-running RL loss on off-policy data.

**Q: Why distributional RL?** Richer signal → better representations → more stable training → risk-sensitive decisions.

**Q: How does PSRO relate to continual learning?** Each PSRO iteration adds a policy while maintaining competence against all previous ones — this IS continual learning over strategy space.

**Q: Can you train an LLM without RL?** DPO, SimPO, ORPO all do alignment without RL. They reparameterize the RLHF objective into supervised losses on preference data. But PPO-based RLHF still has a slight edge on hardest benchmarks.
