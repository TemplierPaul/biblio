# Advanced Deep Learning - Interview Q&A

**Coverage**: Diffusion Models, Vision-Language-Action Models

---


## Table of Contents

- [[#Part 2: Diffusion Models]]
  - [[#Explain forward diffusion process (noising)]]
  - [[#What does the model predict during training (noise, x_0, or score)?]]
  - [[#Write the training objective for DDPM]]
  - [[#What's the reparameterization trick for x_t?]]
  - [[#DDPM vs DDIM sampling - differences?]]
  - [[#What's latent diffusion and why use it?]]
  - [[#Explain classifier-free guidance]]
  - [[#Diffusion vs GAN vs VAE - when to use each?]]
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
