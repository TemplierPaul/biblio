# Diffusion Models

## Definition
Diffusion models are generative models that learn to generate data by reversing a gradual noising process. They iteratively denoise random noise to create high-quality samples (images, audio, etc.).

## Core Idea

### Forward Process (Diffusion)
Gradually add Gaussian noise to data over $T$ steps:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

Where:
- $x_0$: Original data
- $x_T$: Pure noise $\sim \mathcal{N}(0, I)$
- $\beta_t$: Noise schedule (increases over time)

**Closed form** (key property):
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$$

Where $\bar{\alpha}_t = \prod_{s=1}^{t} (1-\beta_s)$

**Reparameterization**:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### Reverse Process (Denoising)
Learn to reverse the diffusion, removing noise step-by-step:
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**Goal**: Start from $x_T \sim \mathcal{N}(0, I)$, generate $x_0$

## Training

### Objective (DDPM)
Train neural network $\epsilon_\theta$ to predict noise:
$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t)\|^2\right]$$

**Algorithm**:
1. Sample $x_0$ from dataset
2. Sample timestep $t \sim \text{Uniform}(1, T)$
3. Sample noise $\epsilon \sim \mathcal{N}(0, I)$
4. Compute noisy $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
5. Predict noise: $\hat{\epsilon} = \epsilon_\theta(x_t, t)$
6. Loss: $\|\epsilon - \hat{\epsilon}\|^2$

**Key Insight**: Instead of predicting $x_0$ directly, predict the noise (more stable)

## Sampling

### DDPM Sampling (Slow)
Iteratively denoise from $x_T$ to $x_0$:
```
x_T ~ N(0, I)
for t = T down to 1:
    z ~ N(0, I) if t > 1, else z = 0
    x_{t-1} = 1/sqrt(α_t) * (x_t - (1-α_t)/sqrt(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z
```

**Problem**: Requires $T=1000$ steps (slow, ~50 seconds for one image)

### DDIM Sampling (Fast)
Deterministic, non-Markovian process:
- Can skip timesteps: use 10-50 steps instead of 1000
- 10-100x faster sampling
- Slight quality tradeoff

**Used in**: Stable Diffusion, DALL-E

## Model Architecture

### U-Net with Time Embedding
- **Backbone**: U-Net (encoder-decoder with skip connections)
- **Input**: Noisy image $x_t$ + timestep $t$
- **Output**: Predicted noise $\epsilon_\theta(x_t, t)$
- **Time encoding**: Sinusoidal positional encoding (like Transformers)
- **Typical**: ResNet blocks, attention layers at low resolution

### Conditional Generation
Add conditioning signal $c$ (text, class label):
$$\epsilon_\theta(x_t, t, c)$$

**Methods**:
- **Class conditioning**: Embed class, concatenate/add to time embedding
- **Text conditioning**: CLIP/T5 text embeddings, cross-attention (Stable Diffusion)
- **Classifier-free guidance**: Amplify conditional signal vs unconditional

## Classifier-Free Guidance
**Technique** to improve conditional generation:
$$\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))$$

Where:
- $w > 1$: Guidance scale (higher = more faithful to condition, less diverse)
- $\emptyset$: Unconditional (drop conditioning with probability during training)

**Effect**: Better alignment with text prompt, sharper images

## Latent Diffusion (Stable Diffusion)
**Problem**: Diffusion in pixel space is slow/expensive

**Solution**: Operate in latent space
1. **VAE Encoder**: $z = E(x)$ (compress image to latent)
2. **Diffusion**: Apply diffusion on $z$ (smaller dimension)
3. **VAE Decoder**: $x = D(z)$ (reconstruct image)

**Advantages**:
- 4-8x faster training/inference
- Lower memory
- Same quality as pixel diffusion

**Architecture** (Stable Diffusion):
- VAE: Compress 512x512 → 64x64 latent
- U-Net: Denoise latent (with text cross-attention)
- Text encoder: CLIP ViT-L/14

## Applications

### Image Generation
- **Text-to-Image**: DALL-E 2, Stable Diffusion, Midjourney
- **Unconditional**: High-quality samples (ImageNet, FFHQ)

### Image Editing
- **Inpainting**: Fill masked regions
- **Super-resolution**: Upscale images
- **Image-to-image**: Edit with text guidance (ControlNet)

### Other Domains
- **Audio**: WaveGrad (speech synthesis)
- **Video**: Video diffusion models
- **3D**: Point clouds, shapes
- **Proteins**: Structure generation

## Comparison with Other Generative Models

| Model | Quality | Diversity | Speed | Likelihood |
|-------|---------|-----------|-------|------------|
| **GAN** | High | Lower | Fast (1 step) | No |
| **VAE** | Lower | High | Fast | Yes (approx) |
| **Flow** | Medium | Medium | Medium | Yes (exact) |
| **Diffusion** | **Highest** | High | Slow (1000→50 steps) | Yes (approx) |

**Why diffusion won**:
- Best sample quality (FID scores)
- More stable training than GANs
- Better mode coverage than GANs
- Scalable conditioning (text, class, etc.)

## Interview Relevance

**Common Questions**:
1. **How do diffusion models work?** Gradually add noise (forward), learn to denoise (reverse)
2. **What does the model predict?** Noise $\epsilon$ added at each step (not image directly)
3. **Why predict noise?** More stable than predicting $x_0$; simplified objective
4. **Forward process formula?** $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
5. **Training objective?** MSE between true noise and predicted noise: $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$
6. **Sampling speed?** DDPM: 1000 steps (slow); DDIM: 50 steps (fast, deterministic)
7. **Latent diffusion advantage?** Operate in compressed latent space (4-8x faster)
8. **Classifier-free guidance?** Amplify conditional signal for better text alignment
9. **Diffusion vs GAN?** Diffusion: better quality, more stable; GAN: faster sampling

**Key Equations**:
- Noisy data: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
- Training loss: $\mathcal{L} = \|\epsilon - \epsilon_\theta(x_t, t)\|^2$
- Classifier-free guidance: $\tilde{\epsilon} = \epsilon_{uncond} + w(\epsilon_{cond} - \epsilon_{uncond})$

**Key Insight**: Diffusion models achieve state-of-the-art generation by learning to reverse a simple noising process, trading inference speed for sample quality and training stability.
