# Machine Learning - Algorithm Reference

Quick reference for neural architectures, training methods, and modern ML techniques.

---

## Neural Network Architectures

### LSTM (Long Short-Term Memory)
**Paper**: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
**What**: RNN with gating mechanisms for long-term dependencies
**How**: Input, forget, output gates + cell state preserve information
**Key innovation**: Solves vanishing gradient problem in RNNs
**When to use**: Sequential data, long-term dependencies (before Transformers)

### Attention Mechanism
**Paper**: "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2015)
**What**: Dynamic weighting of input features based on query-key-value
**How**: Scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T/√d)V
**Key innovation**: Allows selective focus, parallelizable (vs. RNN sequential)
**Types**: Self-attention, cross-attention, multi-head

### Transformer
**Paper**: "Attention Is All You Need" (Vaswani et al., 2017)
**What**: Encoder-decoder architecture using only attention (no recurrence)
**How**: Multi-head self-attention + position encoding + feed-forward layers
**Key innovation**: Fully parallelizable, scales to massive datasets
**When to use**: NLP, sequences, now also vision (ViT), anything really

### Masking
**What**: Technique to control attention patterns in Transformers
**Types**:
- Causal masking: Prevents attending to future tokens (autoregressive)
- Padding masking: Ignores padded positions
- Custom masks: Task-specific attention patterns
**Why**: Enables autoregressive generation, handles variable-length sequences

---

## Large Language Models (LLMs)

### LLMs (GPT, BERT, T5, etc.)
**Key Papers**: "Attention Is All You Need" (2017), "BERT" (2018), "GPT-2/3" (2019/2020)
**What**: Massive Transformer models pre-trained on internet-scale text
**Paradigms**:
- Autoregressive (GPT): Predict next token, left-to-right
- Masked language modeling (BERT): Predict masked tokens, bidirectional
- Seq2seq (T5): Encoder-decoder, flexible task framing
**Key concepts**: Pre-training, scaling laws, emergent abilities, in-context learning

### Fine-tuning
**What**: Adapting pre-trained model to specific task/domain
**Methods**:
- Full fine-tuning: Update all parameters
- Feature extraction: Freeze backbone, train head only
- Task-specific: Supervised fine-tuning on labeled data
**When to use**: Have task-specific data, want better performance than zero-shot

### LoRA (Low-Rank Adaptation)
**Paper**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
**What**: Parameter-efficient fine-tuning using low-rank weight updates
**How**: Add trainable low-rank matrices A, B to frozen weights: W + BA
**Key innovation**: Fine-tune with <1% parameters, merge at inference (no latency)
**When to use**: Limited compute, many task adaptations, preserve base model

### RLHF (Reinforcement Learning from Human Feedback)
**Paper**: "Training language models to follow instructions with human feedback" (Ouyang et al., 2022)
**What**: Align LLM behavior with human preferences using RL
**How**: Reward model from comparisons → PPO policy training → aligned model
**Key innovation**: Bridges capability and alignment, improves helpfulness/safety
**Alternatives**: DPO (Direct Preference Optimization), GRPO

---

## Generative Models

### Diffusion Models
**Paper**: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
**What**: Generate images by iteratively denoising from pure noise
**How**: Forward: gradually add noise. Reverse: learn to denoise
**Key innovation**: Stable training, high-quality samples (beats GANs on many tasks)
**Variants**: DDPM, DDIM, Latent Diffusion (Stable Diffusion), Consistency Models
**When to use**: Image generation, need high quality + diversity

### Gaussian Processes (GPs)
**What**: Non-parametric Bayesian method for regression/classification
**How**: Define prior over functions via kernel, posterior after observing data
**Key advantage**: Principled uncertainty quantification
**When to use**: Small data, need uncertainty, interpretable kernel structure
**See**: Questions/ML/04_Classical_ML.md for details

---

## Specialized Architectures

### GNN (Graph Neural Networks)
**Paper**: "Semi-Supervised Classification with Graph Convolutional Networks" (Kipf & Welling, 2017)
**What**: Neural networks operating on graph-structured data
**How**: Message passing between nodes, aggregate neighborhood features
**Variants**: GCN, GraphSAGE, GAT (graph attention)
**When to use**: Molecular property prediction, social networks, 3D meshes

### VLA (Vision-Language-Action Models)
**Paper**: "RT-1: Robotics Transformer" (Brohan et al., 2022), "RT-2" (2023)
**What**: Unified models for vision, language, and robot control
**How**: Pre-trained vision-language backbone + action head for robot control
**Key innovation**: Leverages internet-scale vision-language data for robotics
**When to use**: Robotic manipulation, need generalization to new objects/tasks

---

## Summary by Domain

| Domain | Algorithm/Architecture | Why |
|--------|----------------------|-----|
| Sequences (old) | LSTM | Handles long-term dependencies |
| Sequences (modern) | Transformer | Parallelizable, scales better |
| Text generation | LLMs (GPT, etc.) | Pre-trained, few-shot capable |
| Task adaptation | LoRA | Parameter-efficient fine-tuning |
| Alignment | RLHF | Human preference optimization |
| Image generation | Diffusion Models | Stable, high-quality |
| Uncertainty quantification | Gaussian Processes | Bayesian, principled |
| Graph data | GNN | Operates on graph structure |
| Robotics + vision | VLA | Unified multimodal control |

---

## Key Paradigm Shifts

### Recurrence → Attention
- LSTM (sequential) → Transformer (parallel)
- RNNs constrained by sequential processing → Attention enables parallelization

### Task-Specific → Pre-train + Fine-tune
- Train from scratch per task → Pre-train once, adapt to many tasks
- Small datasets → Leverage internet-scale pre-training

### Full Fine-tuning → Parameter-Efficient
- Update all weights → LoRA (update <1% parameters)
- One model per task → Many adapters on shared base

### Direct Supervision → Human Feedback
- Supervised learning on labeled data → RLHF on preference comparisons
- Capability-focused → Alignment-focused

---

## Relationships

- **RNN/LSTM** → **Transformer**: Replace recurrence with attention
- **Transformer** → **LLMs**: Scale up massively with pre-training
- **Pre-trained LLM** → **LoRA**: Parameter-efficient adaptation
- **Supervised LLM** → **RLHF**: Add human preference alignment
- **GAN** → **Diffusion**: Stable training, better mode coverage

---

**For comprehensive ML fundamentals, see Questions/ML/07_ML_Fundamentals.md**
**For probability/statistics foundations, see Questions/ML/08_Probability_Statistics.md**
**For mathematical foundations (optimization, linear algebra, VAE, GAN), see Questions/ML/11_Math_Foundations.md**

---

**See individual files for detailed architectures, training procedures, and implementation tips.**
