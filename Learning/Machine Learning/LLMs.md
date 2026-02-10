# Large Language Models (LLMs)

## Definition
Large Language Models are transformer-based models trained on massive text corpora (typically decoder-only architecture) that exhibit emergent abilities like in-context learning, reasoning, and instruction following at sufficient scale.

## Key Characteristics

### Scale
- **Parameters**: Billions to trillions (GPT-3: 175B, LLaMA 2: 70B, GPT-4: estimated 1.76T)
- **Training Data**: Trillions of tokens from web, books, code
- **Compute**: Thousands of GPU/TPU months

### Emergent Abilities
Capabilities that appear suddenly at scale:
- **In-context Learning**: Learn from examples in prompt without gradient updates
- **Chain-of-Thought**: Step-by-step reasoning when prompted
- **Instruction Following**: Generalize to unseen task descriptions
- Typically emerge at ~10B+ parameters

## Training Pipeline

### 1. Pre-training (Base Model)
**Objective**: Next-token prediction (autoregressive language modeling)
$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t})$$

**Data**:
- Web crawl (Common Crawl, filtered)
- Books (Books3, BookCorpus)
- Code (GitHub, Stack Overflow)
- Academic papers (arXiv, PubMed)
- Typical: 1-5T tokens

**Architecture** (Modern LLMs like LLaMA):
- Decoder-only Transformer
- Pre-normalization (RMSNorm)
- SwiGLU activation
- Rotary Position Embeddings (RoPE)
- Grouped-Query Attention (GQA)

**Optimization**:
- AdamW optimizer
- Large batch sizes (4M+ tokens per batch)
- Learning rate warmup + cosine decay
- Gradient clipping
- Mixed precision (BF16/FP16)

### 2. Supervised Fine-Tuning (SFT)
**Purpose**: Align to desired format (instruction following)

**Data**: High-quality instruction-response pairs
- Human-written demonstrations (~10K-100K examples)
- Format: `{"instruction": "...", "response": "..."}`

**Training**:
- Continue language modeling on instruction data
- Smaller learning rate than pre-training
- Shorter training (few epochs)

### 3. Alignment (RLHF/DPO)
**Purpose**: Align to human preferences (helpful, harmless, honest)

See [[RLHF]] for details.

## Key Innovations by Model Family

### GPT Series (OpenAI)
- **GPT-2** (2019): Demonstrated zero-shot capabilities
- **GPT-3** (2020): In-context learning, few-shot prompting
- **GPT-3.5/ChatGPT** (2022): RLHF, conversational agents
- **GPT-4** (2023): Multimodal, improved reasoning

### LLaMA Series (Meta)
- **LLaMA 1** (2023): Open weights, efficient training
- **LLaMA 2** (2023): Longer context (4K), better performance, commercial license
- **LLaMA 3** (2024): Improved tokenizer, 8K/128K context

### Other Notable Models
- **Claude** (Anthropic): Constitutional AI, long context (100K+)
- **PaLM** (Google): 540B params, strong reasoning
- **Mistral/Mixtral**: Mixture-of-Experts, efficient inference

## Important Concepts

### Tokenization
- **BPE (Byte-Pair Encoding)**: Subword tokenization
- **Vocabulary size**: 30K-100K tokens
- **Impact**: Efficiency, multilingual support, handling rare words

### Context Window
- **Definition**: Maximum sequence length model can process
- **Evolution**: 2K (GPT-3) → 4K (GPT-3.5) → 8K → 32K → 100K+ (Claude)
- **Challenges**: $O(n^2)$ attention complexity, position extrapolation

### KV Cache
- **Purpose**: Cache key/value tensors during autoregressive generation
- **Memory**: $2 \times n_{layers} \times d_{model} \times$ sequence length
- **Optimization**: PagedAttention (vLLM), GQA reduces cache size

### Temperature Sampling
- **Temperature $T$**: Controls randomness in generation
  $$P(x_i) = \frac{\exp(logit_i / T)}{\sum_j \exp(logit_j / T)}$$
- $T < 1$: More deterministic
- $T > 1$: More random
- $T \to 0$: Greedy decoding

## Scaling Laws (Chinchilla)
**Key Finding**: Most models are over-parameterized and under-trained

**Optimal scaling**: Parameters and training tokens should scale equally
- For compute budget $C$: $N_{params} \propto C^{0.5}$, $D_{tokens} \propto C^{0.5}$
- **Implication**: Better to train smaller model on more data than larger model on less data

## Inference Optimization

### Techniques
1. **Quantization**: 16-bit → 8-bit/4-bit (GPTQ, AWQ, GGUF)
2. **Speculative Decoding**: Use small model to draft, large model to verify
3. **Flash Attention**: Fused attention kernel (2-4x speedup)
4. **Batching**: Process multiple requests together
5. **Model Parallelism**: Split model across GPUs (tensor/pipeline parallelism)

## Common Interview Topics

### Training
1. **What's the pre-training objective?** Next-token prediction (causal language modeling)
2. **Why decoder-only?** Simpler, scales better, strong in-context learning
3. **Scaling laws?** Chinchilla: balance parameters and data
4. **Distributed training?** Data parallelism, model parallelism (FSDP, Megatron-LM)

### Architecture
1. **Why RoPE over absolute positional encoding?** Better extrapolation to longer contexts
2. **What's GQA?** Grouped-Query Attention: share K/V across heads, reduces KV cache
3. **Modern improvements?** RMSNorm, SwiGLU, GQA, Flash Attention

### Inference
1. **Memory bottleneck?** KV cache for long sequences
2. **Latency optimization?** Quantization, speculative decoding, optimized kernels
3. **How to serve efficiently?** Batching, paged attention (vLLM), continuous batching

### Capabilities
1. **What enables in-context learning?** Emergent property at scale; model learns to adapt from prompt
2. **Few-shot vs zero-shot?** Few-shot: examples in prompt; zero-shot: instruction only
3. **Chain-of-thought?** Explicit reasoning steps improve performance on complex tasks

## Key Insight
LLMs are not just "bigger transformers" - scale enables qualitatively different capabilities (emergence), and the full pipeline (pre-training + SFT + RLHF) is crucial for useful assistants.
