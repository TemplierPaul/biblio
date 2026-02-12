# Deep Learning (GNNs, Diffusion, VLA) - Interview Q&A

**How to use**: Questions are formatted as collapsible headings. Try to answer before expanding.

---

## Part 1: Graph Neural Networks

### What is message passing in GNNs?

**Core paradigm**: Update node representations by **aggregating information from neighbors**.

**Framework**:
$$h_v^{(k+1)} = \text{UPDATE}(h_v^{(k)}, \text{AGGREGATE}(\{h_u^{(k)} : u \in \mathcal{N}(v)\}))$$

**Three steps per layer**:
1. **Message**: Each neighbor sends its representation
2. **Aggregate**: Combine neighbor messages (sum, mean, max, attention)
3. **Update**: Combine aggregated message with own features

**Intuition**: "Tell me about your friends, I'll tell you who you are"

**Example** (social network):
- Layer 1: Learn from direct friends
- Layer 2: Learn from friends-of-friends
- Layer 3: Learn from 3-hop neighborhood

### Explain GCN, GraphSAGE, and GAT - key differences?

**GCN** (Graph Convolutional Network):
- **Aggregation**: Normalized mean (by degree)
- **Formula**: $h_v = \sigma(W \sum_{u \in \mathcal{N}(v)} \frac{h_u}{\sqrt{|N(u)||N(v)||}})$
- **Pros**: Simple, effective
- **Cons**: Fixed weighting (all neighbors equal)

**GraphSAGE**:
- **Aggregation**: Sample + aggregate (mean, max, LSTM)
- **Formula**: $h_v = \sigma(W[h_v \| \text{AGG}(\{h_u\})])$
- **Pros**: Inductive (works on unseen nodes), scalable (sampling)
- **Cons**: Sampling introduces variance

**GAT** (Graph Attention):
- **Aggregation**: Learned attention weights
- **Formula**: $h_v = \sigma(\sum_{u} \alpha_{vu} W h_u)$ where $\alpha_{vu}$ are attention weights
- **Pros**: Different neighbors have different importance
- **Cons**: More parameters, slower

**Key difference**: How neighbors are weighted - GCN (fixed), GraphSAGE (sampling), GAT (learned).

### What's the over-smoothing problem?

**Problem**: After many GNN layers, all node representations **converge to same value**.

**Why it happens**:
- Each layer mixes node with neighbors
- After k layers, node sees k-hop neighborhood
- In connected graph, all nodes eventually see entire graph
- Representations become indistinguishable

**Example**:
- Layer 1: Nodes are different
- Layer 10: All nodes ≈ same (graph-level average)

**Impact**: Can't distinguish nodes, lose local structure

**Solutions**:
1. **Fewer layers**: Use 2-4 layers (not 50+ like CNNs)
2. **Residual connections**: $h^{(k+1)} = h^{(k+1)} + h^{(k)}$
3. **Jumping knowledge**: Combine representations from all layers
4. **Regularization**: DropEdge, layer dropout

### Why do GNNs typically use only 2-4 layers?

**Over-smoothing**: Deep GNNs → all nodes converge to same representation

**Receptive field**:
- 2 layers: 2-hop neighborhood
- 4 layers: 4-hop neighborhood
- In many graphs: 4 hops covers most nodes (small-world property)

**Diminishing returns**: Beyond 4 layers, adding more layers often hurts performance

**Contrast with CNNs**:
- CNNs: 50-200 layers common (ResNet, etc.)
- GNNs: 2-4 layers typical

**Exceptions**: Special architectures (skip connections, normalization) can go deeper, but 2-4 is default.

### What's the difference between node-level, edge-level, and graph-level tasks?

**Node-level**: Predict property of each node
- Example: Classify user's interests, protein function
- Output: Use final node embeddings $h_v^{(K)}$
- Loss: Per-node (cross-entropy, MSE)

**Edge-level**: Predict property of edge or edge existence
- Example: Link prediction, relation classification
- Output: Combine node embeddings $f(h_u, h_v)$ (dot product, concat+MLP)
- Loss: Per-edge

**Graph-level**: Predict property of entire graph
- Example: Molecule toxicity, graph classification
- Output: Aggregate all nodes (sum, mean, attention pooling)
- Loss: Per-graph

### How to do graph-level prediction (readout functions)?

**Readout**: Aggregate all node embeddings into single graph embedding

**Methods**:

1. **Sum**: $h_G = \sum_{v \in V} h_v^{(K)}$
   - Permutation-invariant
   - Simple, works well

2. **Mean**: $h_G = \frac{1}{|V|} \sum_{v} h_v^{(K)}$
   - Normalizes by graph size
   - Better for varying sizes

3. **Max**: $h_G = \max_{v \in V} h_v^{(K)}$ (element-wise)
   - Focuses on most important features

4. **Attention**: $h_G = \sum_v \alpha_v h_v$ where $\alpha_v = \text{softmax}(a^T h_v)$
   - Learned weighting
   - Most expressive

5. **Hierarchical pooling**: Coarsen graph iteratively (DiffPool)

**Choice**: Mean/sum work well, attention for best performance.

### What's the difference between transductive and inductive learning?

**Transductive**:
- Train and test on **same graph**
- Graph structure fixed
- Can see test nodes during training (but not labels)
- Example: GCN on single citation network
- **Use**: Node classification on fixed graph

**Inductive**:
- Generalize to **unseen nodes/graphs**
- Graph structure can change
- Can't see test nodes during training
- Example: GraphSAGE (learns aggregation function)
- **Use**: New nodes added, multiple graphs

**Analogy**:
- Transductive: Closed-world (one graph)
- Inductive: Open-world (new data)

**Most real applications need inductive** (e.g., recommendation, new users).

### When to use GNN vs transformer vs RNN?

**GNN**:
- ✅ Graph-structured data (social, molecular, knowledge graphs)
- ✅ Relational information critical
- ✅ Irregular structure
- Example: Drug discovery, recommendation, traffic

**Transformer**:
- ✅ Sequences (text, time series)
- ✅ Long-range dependencies
- ✅ Parallelizable training
- Example: NLP, vision (as sequences of patches)

**RNN/LSTM**:
- ✅ Sequences with inherent order
- ✅ Streaming/online (process one step at a time)
- ✅ Memory-constrained (long sequences)
- Example: Real-time speech, sensor data

**Overlap**: Transformers can be viewed as fully-connected graph (all-to-all attention).

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
