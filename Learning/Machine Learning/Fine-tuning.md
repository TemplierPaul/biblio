# Fine-tuning

## Definition
Fine-tuning is the process of adapting a pre-trained model to a specific task or domain by continuing training on task-specific data. For LLMs, this typically means instruction tuning or domain adaptation.

## Types of Fine-tuning

### 1. Full Fine-tuning
**Method**: Update all model parameters on new data

**Pros**:
- Maximum flexibility and performance
- Can fully adapt to new domain

**Cons**:
- Memory intensive (need optimizer states for all params)
- Risk of catastrophic forgetting
- Expensive ($1000s for 7B model)
- Storage (need to save full copy per task)

**Typical**: 175B model needs ~1.2TB GPU memory for full fine-tuning (model + gradients + optimizer states)

### 2. Parameter-Efficient Fine-tuning (PEFT)
**Goal**: Adapt model with minimal trainable parameters

See [[LoRA]] for most popular method.

### 3. Instruction Tuning (Supervised Fine-tuning / SFT)
**Purpose**: Teach model to follow instructions

**Data Format**:
```json
{
  "instruction": "Translate to French: Hello, how are you?",
  "input": "",
  "output": "Bonjour, comment allez-vous?"
}
```

**Process**:
1. Start from pre-trained base model
2. Format examples as instruction-response pairs
3. Train with language modeling objective on responses only
4. Use lower learning rate (1e-5 to 1e-4)

**Dataset Examples**:
- **Alpaca**: 52K synthetic instructions (GPT-3.5 generated)
- **FLAN**: 1.8M diverse tasks
- **Dolly**: 15K human-written instructions
- **ShareGPT**: Real ChatGPT conversations

**Result**: Model that responds to instructions instead of just completing text

### 4. Domain Adaptation
**Purpose**: Specialize model for specific domain (medical, legal, code)

**Approaches**:
- Continue pre-training on domain data (DAPT: Domain-Adaptive Pre-Training)
- Then instruction tuning on domain tasks (TAPT: Task-Adaptive Pre-Training)

**Example**: Med-PaLM (medical), CodeLLaMA (programming), BloombergGPT (finance)

## Fine-tuning Hyperparameters

### Critical Settings
- **Learning Rate**: 1e-5 to 5e-5 (much lower than pre-training)
- **Batch Size**: Smaller than pre-training (8-128 examples)
- **Epochs**: 1-3 (more risks overfitting)
- **Warmup**: 100-500 steps
- **Weight Decay**: 0.01-0.1
- **Gradient Clipping**: 1.0

### Learning Rate Considerations
- Too high: Catastrophic forgetting, instability
- Too low: Slow adaptation, underfitting
- Rule of thumb: 1/10 to 1/100 of pre-training LR

## Memory Optimization Techniques

### 1. LoRA (Low-Rank Adaptation)
- Train small rank-decomposition matrices
- Freeze original weights
- See [[LoRA]] for details

### 2. QLoRA (Quantized LoRA)
- Quantize base model to 4-bit
- Train LoRA adapters in higher precision
- Enables fine-tuning 65B model on single 48GB GPU

### 3. Gradient Checkpointing
- Recompute activations during backward pass
- Trades compute for memory
- Enables 2-3x larger batch sizes

### 4. Mixed Precision Training
- Use FP16/BF16 for forward/backward
- Keep FP32 master weights
- 2x memory reduction

### 5. DeepSpeed ZeRO
- **ZeRO-1**: Partition optimizer states
- **ZeRO-2**: Partition gradients
- **ZeRO-3**: Partition parameters
- Enables training larger models on limited GPUs

## Evaluation

### Perplexity
- Measures how well model predicts held-out data
- $\text{PPL} = \exp(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i|x_{<i}))$
- Lower is better
- Useful for domain adaptation

### Task-Specific Metrics
- **Accuracy**: Classification tasks
- **BLEU/ROUGE**: Translation, summarization
- **Exact Match**: QA tasks
- **Human Evaluation**: Instruction following, safety

### Generalization
- Test on held-out tasks (not in training)
- Check if model retains general capabilities
- Evaluate catastrophic forgetting

## Common Pitfalls

### 1. Catastrophic Forgetting
- **Problem**: Model loses general capabilities when fine-tuned on narrow task
- **Solutions**:
  - Lower learning rate
  - Shorter training (early stopping)
  - Mix general data with task-specific data
  - Use PEFT methods (LoRA)

### 2. Overfitting
- **Signs**: Training loss decreases but validation loss increases
- **Solutions**:
  - More data
  - Regularization (weight decay, dropout)
  - Fewer epochs
  - Data augmentation

### 3. Data Quality
- **Issue**: "Garbage in, garbage out" - critical for instruction tuning
- **Best Practice**:
  - Human-written > synthetic
  - Diverse tasks and formats
  - Filter for quality and safety

## Multi-task Fine-tuning

### Approach
Train on mixture of tasks simultaneously

**Advantages**:
- Better generalization
- Positive transfer between tasks
- Single model handles multiple capabilities

**Example**: T5 (Text-to-Text), FLAN (instruction tuning on 1800+ tasks)

**Sampling Strategy**:
- Proportional to dataset size
- Temperature sampling (balance large and small datasets)
- Examples-proportional mixing

## Interview Relevance

**Common Questions**:
1. **Full fine-tuning vs PEFT?** Full: best performance but expensive; PEFT: efficient but may sacrifice some quality
2. **Why lower LR for fine-tuning?** Pre-trained weights are already good; large updates cause catastrophic forgetting
3. **Instruction tuning vs pre-training?** Pre-training: next-token on raw text; Instruction: align to task format
4. **How to prevent catastrophic forgetting?** Lower LR, PEFT, mix general data, early stopping
5. **Memory bottleneck in fine-tuning?** Optimizer states (2x params for Adam); solutions: LoRA, QLoRA, ZeRO
6. **Overfitting signs?** Val loss increases while train loss decreases
7. **How does LoRA help?** Freeze base model, only train small adapters (see [[LoRA]])

**Key Insight**: Fine-tuning is delicate balance - adapt to new task while preserving general capabilities. PEFT methods are increasingly dominant for practical applications.
