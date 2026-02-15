# Advanced Deep Learning - Interview Q&A

**Coverage**: Diffusion Models, Vision-Language-Action Models

---


## Table of Contents

- [[#Part 1: Generative Models]]
  - [[#Explain VAE (Variational Autoencoder)]]
  - [[#Explain GAN (Generative Adversarial Network)]]
  - [[#Diffusion vs GAN vs VAE - when to use each?]]
- [[#Part 2: Diffusion Models]]
  - [[#Explain forward diffusion process (noising)]]
  - [[#What does the model predict during training (noise, x_0, or score)?]]
  - [[#Write the training objective for DDPM]]
  - [[#What's the reparameterization trick for x_t?]]
  - [[#DDPM vs DDIM sampling - differences?]]
  - [[#What's latent diffusion and why use it?]]
  - [[#Explain classifier-free guidance]]
  - [[#Why do diffusion models achieve better sample quality than GANs?]]
- [[#Part 3: Vision-Language-Action Models]]
  - [[#What's the key innovation of VLA models?]]
  - [[#RT-1 vs RT-2 architecture differences?]]
  - [[#How are actions represented (discretization)?]]
  - [[#What's FiLM conditioning?]]
  - [[#How does RT-2 leverage pre-trained VLMs?]]
  - [[#What emergent capabilities does RT-2 show?]]
  - [[#Why co-fine-tune on robot data?]]
  - [[#What are the limitations of VLAs?]]
- [[#Part 4: Advanced Architectures]]
  - [[#What is Mixture of Experts (MoE)?]]

---

## Part 1: Generative Models

### Explain VAE (Variational Autoencoder)

**The Goal:**

Learn a generative model of data p(x) that can:
1. Generate new samples similar to training data
2. Learn meaningful latent representations
3. Handle uncertainty

**Architecture:**

**Encoder:** q(z|x) - Maps data x to latent code z
**Decoder:** p(x|z) - Reconstructs x from latent code z

**Key Idea - Variational Inference:**

We want to learn p(z|x), but it's intractable. Instead, learn an approximate distribution q(z|x) that's close to the true posterior.

**The VAE Loss (ELBO):**

Maximize Evidence Lower Bound:

ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))

**Two terms:**

1. **Reconstruction loss**: E[log p(x|z)]
   - How well can we reconstruct x from z?
   - Typically: MSE for continuous data, BCE for binary

2. **KL divergence**: KL(q(z|x) || p(z))
   - How different is q(z|x) from prior p(z)?
   - Regularizes latent space
   - Typically: KL between Gaussian q(z|x) and standard normal p(z)

**Reparameterization Trick:**

Problem: Can't backpropagate through sampling z ~ q(z|x)

Solution: Reparameterize as:
- Sample ε ~ N(0, I)
- Compute z = μ(x) + σ(x) ⊙ ε

Now gradients flow through μ and σ!

**Intuition:**

- **Encoder** compresses x into mean μ and variance σ²
- **Sample** z from N(μ, σ²) using reparameterization
- **Decoder** reconstructs x from z
- **KL term** keeps latent space organized (prevents overfitting)

**Why the KL term matters:**

Without it, encoder could map each x to a unique z with zero variance → no generalization.

With it, similar inputs get mapped to overlapping regions in latent space → smooth interpolation.

**Training Process:**

1. Input image x
2. Encode to μ(x), σ(x)
3. Sample z = μ + σ ⊙ ε
4. Decode to x̂
5. Compute loss: ||x - x̂||² + KL(N(μ,σ²) || N(0,I))
6. Backpropagate, update weights

**Generating New Samples:**

1. Sample z ~ N(0, I) from prior
2. Decode z to get x = decoder(z)
3. Since latent space is continuous, can interpolate between points

**Applications:**

- Image generation
- Data compression
- Anomaly detection (high reconstruction error)
- Semi-supervised learning
- Disentangled representations

**VAE vs Autoencoder:**

| Feature | Autoencoder | VAE |
|---------|-------------|-----|
| Latent space | Deterministic | Probabilistic |
| Loss | Reconstruction only | Reconstruction + KL |
| Generation | No (irregular latent space) | Yes (smooth prior) |
| Interpolation | Poor | Smooth |

---

### Explain GAN (Generative Adversarial Network)

**The Core Idea:**

Two neural networks compete in a game:
- **Generator (G)**: Creates fake data
- **Discriminator (D)**: Distinguishes real from fake

**The Adversarial Game:**

- **D's goal**: Maximize ability to classify real vs fake
  - D(x_real) → 1
  - D(G(z)) → 0

- **G's goal**: Fool D by generating realistic data
  - D(G(z)) → 1

**Objective Function:**

min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]

**Training Alternates:**

1. **Update D** (discriminator):
   - Sample real data x
   - Sample noise z, generate fake G(z)
   - Train D to maximize V (classify correctly)

2. **Update G** (generator):
   - Sample noise z
   - Train G to minimize V (fool D)

**Intuition - Counterfeiter vs Police:**

- Generator = counterfeiter making fake money
- Discriminator = police detecting fakes
- As police get better at detection, counterfeiter improves quality
- Eventually, counterfeits become indistinguishable from real money

**Training Dynamics:**

**Nash Equilibrium:**
When G generates perfect samples, D outputs 0.5 (can't tell real from fake).

At this point:
- D can't improve (samples are perfect)
- G can't improve (already generating real distribution)

**Practical Challenges:**

**1. Mode Collapse:**
- Generator produces limited variety (only a few types of outputs)
- **Solution**: Unrolled GAN, minibatch discrimination

**2. Training Instability:**
- D becomes too strong → G gets no gradient signal
- G becomes too strong → D gives up
- **Solution**: Careful tuning, Wasserstein GAN

**3. Vanishing Gradients:**
- When D is perfect, log(1-D(G(z))) saturates
- **Solution**: Use -log D(G(z)) instead (non-saturating loss)

**GAN Variants:**

**DCGAN (Deep Convolutional GAN):**
- Use conv layers instead of fully connected
- BatchNorm in G and D
- LeakyReLU in D, ReLU in G
- More stable training

**WGAN (Wasserstein GAN):**
- Replace JS divergence with Wasserstein distance
- More meaningful loss metric
- More stable training
- Clip weights or use gradient penalty

**Conditional GAN:**
- Condition on class label y
- G(z, y) and D(x, y)
- Can control what type of image to generate

**StyleGAN:**
- State-of-the-art image generation
- Progressive growing
- Style transfer at different resolutions
- Generates incredibly realistic faces

**Applications:**

- **Image generation**: High-quality synthetic images
- **Image-to-image translation**: Pix2Pix, CycleGAN (day→night, sketch→photo)
- **Super-resolution**: Upscale low-res images
- **Data augmentation**: Generate training data
- **Art and creativity**: Generate art, music, text

**VAE vs GAN:**

| Feature | VAE | GAN |
|---------|-----|-----|
| Training | Stable | Can be unstable |
| Sample quality | Blurry | Sharp, realistic |
| Latent space | Interpretable | Less interpretable |
| Diversity | Good | Risk of mode collapse |
| Likelihood | Tractable lower bound | Intractable |

**When to Use Each:**

- **VAE**: Need stable training, interpretable latent space, anomaly detection
- **GAN**: Need highest quality samples, image synthesis, don't care about exact likelihood

### Diffusion vs GAN vs VAE - when to use each?

**Diffusion**:
- ✅ Best sample quality (FID scores)
- ✅ Stable training (no mode collapse)
- ✅ Good mode coverage
- ❌ Slow sampling (10-1000 steps)
- **Use**: When quality matters most (text-to-image)

**GAN**:
- ✅ Fast sampling (1 step)
- ✅ Sharp images
- ❌ Training instability (mode collapse)
- ❌ Poor mode coverage
- **Use**: Real-time generation, video

**VAE**:
- ✅ Fast sampling
- ✅ Exact likelihood
- ❌ Blurry samples
- ❌ Lower quality
- **Use**: Compression, representation learning

**Trend**: Diffusion dominant for images (Stable Diffusion, DALL-E), GANs for video/real-time.

---

## Part 2: Diffusion Models

### Explain forward diffusion process (noising)

**Forward process**: Gradually add Gaussian noise over T steps

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

Where:
- $x_0$: Original image
- $x_T$: Pure noise $\sim \mathcal{N}(0, I)$
- $\beta_t$: Noise schedule (increases: 0.0001 → 0.02)

**Closed form** (key property):
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

Where $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$

**Intuition**: After T=1000 steps, image becomes pure noise (destroys information).

### What does the model predict during training (noise, x_0, or score)?

**Standard (DDPM)**: Predict **noise** $\epsilon$

**Training**:
1. Sample $x_0$ from dataset
2. Sample timestep $t$ uniformly
3. Sample noise $\epsilon \sim \mathcal{N}(0,I)$
4. Create noisy image: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
5. Predict noise: $\hat{\epsilon} = \epsilon_\theta(x_t, t)$

**Alternatives**:
- Predict $x_0$ directly
- Predict score $\nabla_{x_t} \log p(x_t)$

**Why predict noise**:
- More stable training
- Equivalent objectives (related by reparameterization)
- Empirically works best

### Write the training objective for DDPM

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t) \|^2 \right]$$

**Components**:
- $t \sim \text{Uniform}(1, T)$: Random timestep
- $x_0 \sim p_{data}$: Real image
- $\epsilon \sim \mathcal{N}(0, I)$: Random noise
- $\epsilon_\theta$: Noise prediction network

**Simplified**: Mean squared error between true noise and predicted noise.

**Key insight**: Train network to "denoise" by predicting what noise was added.

### What's the reparameterization trick for x_t?

Instead of sampling $x_t$ sequentially (apply noise T times):

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

Where $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$

**Advantages**:
1. **Direct sampling**: Jump to any timestep t in one step
2. **Efficient training**: No need to run forward process T times
3. **Parallelizable**: Sample different t's independently

**Derivation**: Apply Gaussian convolution repeatedly (product of Gaussians).

### DDPM vs DDIM sampling - differences?

**DDPM** (Denoising Diffusion Probabilistic Models):
- Stochastic sampling (adds noise at each step)
- Markovian process
- Requires ~1000 steps
- Slow (~50 seconds per image)

**DDIM** (Denoising Diffusion Implicit Models):
- **Deterministic** sampling (optional noise)
- **Non-Markovian** (can skip timesteps)
- Requires only 10-50 steps
- **10-100x faster**

**Trade-off**: DDIM slightly lower diversity, but quality nearly identical

**Why DDIM works**: Finds different (non-Markovian) process with same marginals $p(x_t)$

**Used in**: Stable Diffusion (50 DDIM steps by default)

### What's latent diffusion and why use it?

**Latent Diffusion**: Run diffusion in **compressed latent space** instead of pixel space

**Architecture**:
1. **VAE encoder**: $z = E(x)$ (512×512 → 64×64 latent)
2. **Diffusion**: Denoise in latent space
3. **VAE decoder**: $x = D(z)$ (64×64 → 512×512 image)

**Why use it**:
- **4-8x faster**: Smaller spatial dimensions ($64^2$ vs $512^2$)
- **Lower memory**: Less attention computation
- **Same quality**: VAE learned good compression

**Example** (Stable Diffusion):
- Pixel diffusion: 512×512×3 = 786K dimensions
- Latent diffusion: 64×64×4 = 16K dimensions (~50x reduction)

**Trade-off**: Slight VAE artifacts, but huge efficiency gain.

### Explain classifier-free guidance

**Goal**: Improve alignment with text prompt (make outputs more faithful to condition).

**Formula**:
$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

Where:
- $c$: Condition (text prompt)
- $\emptyset$: Unconditional (empty prompt)
- $w$: Guidance scale (typically 7-15)

**Interpretation**: Move prediction in direction away from unconditional, toward conditional

**Effect**:
- $w = 0$: Unconditional (ignore prompt)
- $w = 1$: Standard conditional
- $w > 1$: Amplify conditioning (sharper, more aligned)

**Trade-off**: Higher w → better alignment but less diversity

**Training requirement**: Randomly drop conditioning (10-20% of time) during training to learn $\epsilon_\theta(x_t, \emptyset)$





### Why do diffusion models achieve better sample quality than GANs?

1. **Stable training**: No adversarial game, just regression (predict noise)
2. **Better mode coverage**: No mode collapse (VAE-like coverage)
3. **Iterative refinement**: Multiple denoising steps improve quality
4. **Scalability**: Easier to scale to large models/datasets
5. **No GAN discriminator tricks**: Don't need careful balancing

**Evidence**: SOTA FID scores on ImageNet, realistic images (Stable Diffusion vs StyleGAN)

**Trade-off**: Slower sampling (but DDIM, distillation help)


---


## Part 3: Vision-Language-Action Models

### What's the key innovation of VLA models?

**End-to-end learning** from (vision + language) → robot actions using transformers.

**Traditional robotics**:
```
Perception → State Estimation → Planning → Control
(separate hand-engineered modules)
```

**VLA**:
```
(Image, Text Instruction) → Transformer → Action
(single learned model)
```

**Key innovations**:
1. Treat robotics as **sequence modeling** (like language)
2. Leverage **pre-trained vision-language models**
3. **Action tokenization**: Discretize actions for transformers

### RT-1 vs RT-2 architecture differences?

**RT-1** (Robotics Transformer 1):
- Trained **from scratch** on robot data
- Vision: EfficientNet (ImageNet pre-trained)
- Language: Universal Sentence Encoder
- FiLM conditioning (vision + language)
- 130K robot episodes

**RT-2** (Robotics Transformer 2):
- **Co-fine-tune** pre-trained VLM (PaLI-X, PaLM-E)
- Vision: ViT (Vision Transformer)
- Language: PaLM (LLM)
- Add action tokens to vocabulary
- 100K robot episodes + web VLM data

**Key difference**: RT-2 leverages internet-scale vision-language knowledge.

### How are actions represented (discretization)?

**Continuous actions**: 7-DoF (x, y, z, roll, pitch, yaw, gripper) ∈ ℝ⁷

**Discretization** (RT-1):
- Bin each dimension into 256 values
- Action becomes sequence of 7 tokens (each in 0-255)
- Predict via **classification** (cross-entropy)

**Why discretize**:
- Transformers designed for discrete tokens
- Classification more stable than regression
- Enables multi-modal action distributions

**Alternatives**: Continuous actions (regression head), mixture of Gaussians, diffusion.

### What's FiLM conditioning?

**FiLM** (Feature-wise Linear Modulation):

Condition visual features on language via **affine transformation**:
$$\text{FiLM}(h) = \gamma \odot h + \beta$$

Where:
- $h$: Visual features
- $\gamma, \beta$: Learned from language embedding
- $\odot$: Element-wise multiplication

**In RT-1**:
1. Language → embedding
2. Embedding → $\gamma, \beta$ via MLP
3. Apply to vision features at multiple layers

**Advantage**: Efficient fusion, conditions low-level visual processing on language.

### How does RT-2 leverage pre-trained VLMs?

**Co-fine-tuning** strategy:

1. **Start**: Pre-trained VLM (PaLI-X) on billions of image-text pairs
   - Already understands objects, scenes, language

2. **Add**: Action tokens to vocabulary (like text tokens)

3. **Fine-tune**: On robot trajectories
   - Input: Image + "pick up the apple"
   - Output: Action tokens [0.1, 0.2, -0.3, ...]

4. **Result**: Model that does vision-language tasks AND robot control

**Key**: Internet knowledge transfers to robotics (symbol grounding, reasoning).

### What emergent capabilities does RT-2 show?

**Beyond basic control** (from web pre-training):

1. **Reasoning**: "Move banana to Taylor Swift album number" → moves to position 10

2. **Symbol grounding**: "Pick up extinct animal" → grasps toy dinosaur (understands "extinct")

3. **Math**: "Move to sum of 2+2" → position 4

4. **Chain-of-thought**: "I should pick X because..." → action

5. **Visual understanding**: "Pick the fruit" → recognizes apple is fruit

**Why surprising**: Never trained on these specific tasks - emerges from VLM pre-training.

**Evidence**: 62% success on novel tasks (vs 32% for RT-1).

### Why co-fine-tune on robot data?

**Alternative**: Freeze VLM, add action head → poor performance

**Co-fine-tuning**:
- Adapt **entire model** (vision + language + new action head)
- VLM features learn to support action prediction
- Better integration of vision-language-action

**Trade-off**: Risk catastrophic forgetting of VLM capabilities
- Solution: Mix robot data with VLM tasks during fine-tuning

**Result**: Model that retains VLM knowledge while learning robot control.

### What are the limitations of VLAs?

1. **Sample efficiency**: Still need 100K+ robot episodes (expensive to collect)

2. **Action precision**: Discretization limits fine-grained control

3. **Long-horizon tasks**: Struggle with multi-step planning (typically 1-3 steps)

4. **Safety**: No safety guarantees, can produce unexpected behaviors

5. **Sim-to-real gap**: Web images ≠ robot camera views

6. **Fixed morphology**: Doesn't generalize to different robot bodies

7. **Compute**: Large models, slow inference

**Open challenges**: Scaling robot data, sim-to-real transfer, safety.

---

## Part 4: Advanced Architectures

### What is Mixture of Experts (MoE)?

**Core Concept:**

Instead of one large model, use multiple specialized "expert" networks, and learn when to use each expert.

**Architecture:**

1. **Multiple expert networks**: E₁, E₂, ..., Eₙ
2. **Gating network**: Decides which experts to use
3. **Combiner**: Aggregates expert outputs

**How It Works:**

For input x:
1. Gating network outputs weights: g(x) = [g₁, g₂, ..., gₙ]
2. Each expert produces output: yᵢ = Eᵢ(x)
3. Final output: y = Σᵢ gᵢ(x) · yᵢ

**Gating Function:**

Typically softmax over expert scores:
g(x) = softmax(Wx)

This ensures weights sum to 1 and are non-negative.

**Types of MoE:**

**Soft MoE:**
- All experts contribute (weighted)
- Smooth, differentiable
- More computation (run all experts)

**Hard MoE (Sparse MoE):**
- Only activate top-k experts
- Efficient (fewer computations)
- Requires tricks for gradients (Gumbel-softmax, straight-through estimators)

**Example - Language Model:**

- Expert 1: Handles technical text
- Expert 2: Handles creative writing
