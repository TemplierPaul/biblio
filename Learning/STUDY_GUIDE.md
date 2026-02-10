# ML Research Scientist Interview Study Guide

**Purpose**: Progressive learning path for ML research scientist interviews, focusing on deep learning, LLMs, and game theory/multi-agent RL.

**How to use**: Complete each part in order, ensuring you can confidently answer all questions before moving to the next section.

📝 **[[INTERVIEW_ANSWERS|Interview Answers]]** - Complete answers to all study questions (collapsible format)

---

## Part 1: Sequence Models & Recurrence

**Topics**: [[Machine Learning/LSTM|LSTM]]

**Why start here**: Foundation for understanding sequential data and the problems transformers solve.

### Must Answer Confidently:
1. What problem does LSTM solve compared to vanilla RNN?
2. Explain the three gates in LSTM and their purposes (forget, input, output)
3. Write the cell state update equation and explain why it prevents vanishing gradients
4. What's the difference between cell state and hidden state?
5. LSTM vs GRU: when to use each?
6. Why do transformers dominate over LSTMs in NLP today?
7. When would you still use LSTM over transformer?

### Algorithms to Explain:
- **LSTM forward pass**: Step through all gates with example
- **Gradient flow**: Why additive cell state prevents vanishing gradients

### Practice:
- Draw LSTM cell diagram from memory
- Trace forward pass with concrete numbers

**Time estimate**: 1-2 days

---

## Part 2: Attention Mechanism

**Topics**: [[Machine Learning/Attention|Attention]]

**Why now**: Core building block for transformers and modern deep learning.

### Must Answer Confidently:
1. What is the scaled dot-product attention formula?
2. Why do we scale by √d_k?
3. What are Q, K, V matrices and what do they represent?
4. Difference between self-attention and cross-attention?
5. What is multi-head attention and why use it?
6. What's the computational complexity of attention? What's the bottleneck?
7. How does attention compare to RNN for long-range dependencies?

### Algorithms to Explain:
- **Scaled dot-product attention**: Step-by-step computation
- **Multi-head attention**: How heads are computed and concatenated

### Practice:
- Compute attention weights for a small example (4 tokens)
- Implement attention from scratch (pseudocode)

**Time estimate**: 2-3 days

---

## Part 3: Transformer Architecture

**Topics**: [[Machine Learning/Transformer|Transformer]], [[Machine Learning/Masking|Masking]]

**Why now**: Builds on attention, foundation for all modern LLMs.

### Must Answer Confidently:
1. Draw transformer encoder and decoder architecture
2. What are the key components in each transformer layer?
3. Why do we need positional encoding?
4. What's the difference between pre-LN and post-LN?
5. Explain causal masking and why it's needed for generation
6. What's the difference between encoder-only, decoder-only, and encoder-decoder?
7. Why are decoder-only models dominant for LLMs?
8. What is masked language modeling (MLM)?
9. Explain the 80/10/10 split in BERT's MLM

### Algorithms to Explain:
- **Transformer forward pass**: Trace through encoder and decoder
- **Causal masking**: How to implement and why
- **Position encoding**: Sinusoidal formula

### Practice:
- Implement causal mask for sequence length 5
- Trace attention computation through one transformer layer
- Explain why GPT uses causal masking but BERT doesn't

**Time estimate**: 3-4 days

---

## Part 4: Large Language Models

**Topics**: [[Machine Learning/LLMs|LLMs]]

**Why now**: Modern application of transformers at scale.

### Must Answer Confidently:
1. What's the pre-training objective for decoder-only LLMs?
2. Explain the three-stage training pipeline (pre-training → SFT → RLHF)
3. What are emergent abilities and at what scale do they appear?
4. What is in-context learning?
5. Explain the Chinchilla scaling laws
6. What's the KV cache and why is it needed?
7. What's the difference between temperature sampling and greedy decoding?
8. Why use RoPE over absolute positional encoding?
9. What's Grouped-Query Attention (GQA) and why use it?
10. Explain the difference between GPT-3, ChatGPT, and GPT-4

### Algorithms to Explain:
- **Autoregressive generation**: How tokens are generated one by one
- **Temperature sampling**: Formula and effect

### Practice:
- Calculate memory requirements for 7B model (parameters, KV cache)
- Explain inference bottlenecks and optimizations

**Time estimate**: 4-5 days

---

## Part 5: Fine-tuning & PEFT

**Topics**: [[Machine Learning/Fine-tuning|Fine-tuning]], [[Machine Learning/LoRA|LoRA]]

**Why now**: Practical adaptation of pre-trained models.

### Must Answer Confidently:
1. What's the difference between full fine-tuning and PEFT?
2. Why use lower learning rate for fine-tuning?
3. What is catastrophic forgetting and how to prevent it?
4. What's instruction tuning (SFT)?
5. Explain LoRA: what are the A and B matrices?
6. Why does LoRA reduce parameters by 10,000x?
7. What rank should you use for LoRA?
8. Why initialize B to zero in LoRA?
9. What's QLoRA and how does it enable 65B on single GPU?
10. When to use LoRA vs full fine-tuning?

### Algorithms to Explain:
- **LoRA forward pass**: How BA is added to frozen weights
- **LoRA merging**: Combining adapters

### Practice:
- Calculate trainable parameters for LoRA (rank 8) on 7B model
- Implement LoRA layer in pseudocode

**Time estimate**: 2-3 days

---

## Part 6: Alignment & RLHF

**Topics**: [[Machine Learning/RLHF|RLHF]]

**Why now**: Critical for making LLMs helpful and safe.

### Must Answer Confidently:
1. Why do we need RLHF after pre-training?
2. Explain the three stages of RLHF pipeline
3. What's the Bradley-Terry model for reward modeling?
4. Why use PPO for the RL stage?
5. What's the KL penalty term and why is it critical?
6. What is reward hacking? Give examples
7. What's the difference between RLHF and DPO?
8. When does overoptimization occur?
9. Why does RLHF require 4x more compute than SFT?
10. Compare RLHF, DPO, RLAIF, Constitutional AI

### Algorithms to Explain:
- **Reward model training**: Bradley-Terry objective
- **PPO objective**: Reward + KL penalty

### Practice:
- Explain reward hacking scenario
- Design reward model training data collection

**Time estimate**: 3-4 days

---

## Part 7: Graph Neural Networks

**Topics**: [[Machine Learning/GNN|GNN]]

**Why now**: Important for structured/relational data.

### Must Answer Confidently:
1. What is message passing in GNNs?
2. Explain GCN, GraphSAGE, and GAT - key differences?
3. What's the over-smoothing problem?
4. Why do GNNs typically use only 2-4 layers?
5. What's the difference between node-level, edge-level, and graph-level tasks?
6. How to do graph-level prediction (readout functions)?
7. What's the difference between transductive and inductive learning?
8. When to use GNN vs transformer vs RNN?

### Algorithms to Explain:
- **Message passing framework**: Aggregate-Update
- **GCN layer**: With normalization
- **GAT attention**: How attention weights are computed

### Practice:
- Trace message passing through 2 GNN layers on small graph
- Design GNN for molecule property prediction

**Time estimate**: 2-3 days

---

## Part 8: Diffusion Models

**Topics**: [[Machine Learning/Diffusion|Diffusion]]

**Why now**: State-of-the-art generative models.

### Must Answer Confidently:
1. Explain forward diffusion process (noising)
2. What does the model predict during training (noise, x_0, or score)?
3. Write the training objective for DDPM
4. What's the reparameterization trick for x_t?
5. DDPM vs DDIM sampling - differences?
6. What's latent diffusion and why use it?
7. Explain classifier-free guidance
8. Diffusion vs GAN vs VAE - when to use each?
9. Why do diffusion models achieve better sample quality than GANs?

### Algorithms to Explain:
- **Training loop**: Sample x_0, t, ε; predict ε
- **Sampling**: Iterative denoising from x_T to x_0
- **Classifier-free guidance**: Formula

### Practice:
- Trace forward process for t=1,2,3
- Explain Stable Diffusion architecture

**Time estimate**: 3-4 days

---

## Part 9: Gaussian Processes

**Topics**: [[Machine Learning/Gaussian-Processes|Gaussian Processes]]

**Why now**: Bayesian approach, uncertainty quantification.

### Must Answer Confidently:
1. What's a Gaussian Process?
2. Explain mean and covariance (kernel) functions
3. Write the predictive mean and variance formulas
4. Why does GP provide uncertainty estimates?
5. What's the RBF/squared exponential kernel?
6. How to tune hyperparameters (marginal likelihood)?
7. What's the computational complexity of GP?
8. How to scale GPs (sparse GP, inducing points)?
9. When to use GP vs neural network?
10. What's the connection between infinite-width NNs and GPs?

### Algorithms to Explain:
- **GP prediction**: Conditioning Gaussian distribution
- **Marginal likelihood optimization**: NLL as loss

### Practice:
- Compute GP prediction for toy dataset (3 points)
- Explain why variance increases far from data

**Time estimate**: 2-3 days

---

## Part 10: Vision-Language-Action Models

**Topics**: [[Machine Learning/VLA|VLA (RT-2)]]

**Why now**: Application of transformers to robotics.

### Must Answer Confidently:
1. What's the key innovation of VLA models?
2. RT-1 vs RT-2 architecture differences?
3. How are actions represented (discretization)?
4. What's FiLM conditioning?
5. How does RT-2 leverage pre-trained VLMs?
6. What emergent capabilities does RT-2 show?
7. Why co-fine-tune on robot data?
8. What are the limitations of VLAs?

### Algorithms to Explain:
- **RT-2 forward pass**: Vision + language → action tokens
- **Action tokenization**: Discretizing continuous actions

### Practice:
- Design VLA architecture for new task

**Time estimate**: 1-2 days

---

# PART 11: Game Theory Foundations

**Topics**: [[Game Theory/Normal Form|Normal Form]], [[Game Theory/Extensive Form|Extensive Form]], [[Game Theory/Nash Equilibrium|Nash Equilibrium]]

**Why now**: Foundation for multi-agent RL and self-play methods.

### Must Answer Confidently:
1. What's a normal form game? Components?
2. What's a payoff matrix?
3. Pure vs mixed strategy Nash equilibrium?
4. Does every finite game have a Nash equilibrium?
5. What's an extensive form game?
6. Difference between perfect and imperfect information?
7. What are information sets?
8. How to solve perfect information games (backward induction)?
9. What's a subgame perfect equilibrium?
10. Nash equilibrium vs social optimum - are they the same?

### Algorithms to Explain:
- **Finding pure NE**: Best response method (underline method)
- **Finding mixed NE**: Indifference condition
- **Backward induction**: Solving game trees

### Practice:
- Find Nash equilibria for 2x2 games (pure and mixed)
- Solve simple extensive form game with backward induction

**Time estimate**: 2-3 days

---

## Part 12: Social Dilemmas

**Topics**: [[Game Theory/Prisoners-Dilemma|Prisoner's Dilemma]]

**Why now**: Fundamental tension in multi-agent systems.

### Must Answer Confidently:
1. What's the Nash equilibrium of prisoner's dilemma?
2. Is Nash equilibrium Pareto optimal in PD?
3. What's a dominant strategy?
4. What makes it a "dilemma"?
5. How does iterated PD change the game?
6. What's tit-for-tat strategy?
7. Real-world examples of prisoner's dilemmas?
8. How to achieve cooperation in PD?
9. What's the n-player version (public goods game)?

### Algorithms to Explain:
- **Dominant strategy analysis**
- **Tit-for-tat**: Strategy description

### Practice:
- Verify (D,D) is Nash equilibrium
- Explain why cooperation is hard to achieve

**Time estimate**: 1 day

---

## Part 13: Classical Game Algorithms

**Topics**: [[Game Theory/Minimax|Minimax]], [[Game Theory/MCTS|MCTS]]

**Why now**: Core algorithms for perfect information games.

### Must Answer Confidently:
1. What's the minimax principle?
2. How does minimax algorithm work (recursive)?
3. What's alpha-beta pruning and how much does it save?
4. What's the complexity of minimax?
5. When is minimax optimal?
6. Explain the four phases of MCTS (Selection, Expansion, Simulation, Backup)
7. What's UCT formula?
8. What's PUCT and how does it use neural networks?
9. Minimax vs MCTS - when to use each?
10. Why is MCTS better for Go than minimax?

### Algorithms to Explain:
- **Minimax with alpha-beta**: Step through with example tree
- **MCTS iteration**: Complete cycle of 4 phases
- **UCT selection**: Formula and intuition

### Practice:
- Run minimax on tic-tac-toe tree (depth 2)
- Trace MCTS for 5 iterations on simple game
- Implement alpha-beta pruning (pseudocode)

**Time estimate**: 3-4 days

---

## Part 14: AlphaZero

**Topics**: [[Game Theory/AlphaZero|AlphaZero]]

**Why now**: Landmark self-play algorithm combining RL + MCTS.

### Must Answer Confidently:
1. What are the two heads of AlphaZero's network?
2. How does PUCT use the policy network?
3. Why doesn't AlphaZero use rollouts (unlike classic MCTS)?
4. What's the training loop (self-play → train → repeat)?
5. What's the loss function (three components)?
6. How does value network replace rollouts?
7. What's the difference between AlphaGo and AlphaZero?
8. Why is AlphaZero considered "tabula rasa"?
9. What are limitations (perfect info, two-player, etc.)?
10. How does temperature affect move selection?

### Algorithms to Explain:
- **MCTS with neural network**: PUCT selection, value evaluation
- **Self-play data generation**: How (s, π, z) tuples are created
- **Training**: Three-component loss function

### Practice:
- Trace MCTS iteration with PUCT selection
- Explain how policy is distilled from MCTS
- Design AlphaZero variant for new game

**Time estimate**: 3-4 days

---

## Part 15: PSRO & Population-Based Methods

**Topics**: [[Game Theory/Self-play/PSRO|PSRO]], [[Game Theory/JPSRO|JPSRO]]

**Why now**: Scalable game-theoretic solving with deep RL.

### Must Answer Confidently:
1. What's the PSRO algorithm (4 steps)?
2. What's an empirical game?
3. What's a best response oracle?
4. How does PSRO converge to Nash equilibrium?
5. Difference between PSRO and double oracle?
6. What's JPSRO and how does it differ from PSRO?
7. What's Correlated Equilibrium (CE)?
8. What's Coarse Correlated Equilibrium (CCE)?
9. Hierarchy: Nash ⊆ CE ⊆ CCE - explain
10. Why use CCE for general-sum games?
11. How many BRs needed per iteration (PSRO vs JPSRO)?

### Algorithms to Explain:
- **PSRO loop**: Initialize → Meta-game → BR → Expand
- **JPSRO**: Joint distribution BR training
- **Meta-strategy solver**: Nash vs CCE

### Practice:
- Trace PSRO for 3 iterations on simple game
- Explain traffic light game (why CE > Nash)
- Design PSRO variant for specific game

**Time estimate**: 4-5 days

---

## Part 16: Neural Population Learning

**Topics**: [[Game Theory/Self-play/NeuPL|NeuPL]]

**Why now**: Efficient alternative to standard PSRO.

### Must Answer Confidently:
1. What's the key innovation of NeuPL vs PSRO?
2. How does conditional network work: π(a|s,i)?
3. What's the memory advantage (O(1) vs O(N))?
4. What's transfer learning in NeuPL?
5. NeuPL vs NeuPL-JPSRO differences?
6. What equilibrium does NeuPL converge to?
7. When to use NeuPL vs standard PSRO?

### Algorithms to Explain:
- **Conditional policy network**: How policy index conditions network
- **NeuPL training**: BR + distillation

### Practice:
- Design conditional network architecture
- Calculate memory savings vs PSRO

**Time estimate**: 2-3 days

---

## Part 17: Fictitious Self-Play

**Topics**: [[Game Theory/Self-play/FSP|FSP (Fictitious Self-Play)]]

**Why now**: Alternative equilibrium-finding approach.

### Must Answer Confidently:
1. What's fictitious play (classical)?
2. How does FSP use RL and SL?
3. What's NFSP (Neural FSP)?
4. What are the two networks and two buffers?
5. Why does FSP converge to Nash in some games?
6. FSP vs vanilla self-play - why more stable?
7. When to use FSP vs PSRO?

### Algorithms to Explain:
- **NFSP training loop**: RL for BR, SL for average
- **Policy mixing**: ε-greedy between RL and SL policies

### Practice:
- Explain two-network architecture
- Compare FSP convergence to vanilla self-play

**Time estimate**: 2 days

---

## Part 18: Counterfactual Regret Minimization

**Topics**: [[Game Theory/Partial Observation/CFR|CFR]]

**Why now**: Leading algorithm for imperfect information games.

### Must Answer Confidently:
1. What games is CFR designed for (imperfect info, extensive form)?
2. What's an information set?
3. What's counterfactual value?
4. Explain regret matching strategy update
5. Why use average strategy (not current strategy)?
6. What's CFR+ and how does it improve vanilla CFR?
7. What's Monte Carlo CFR (sampling)?
8. What's Deep CFR?
9. Complexity of CFR per iteration?
10. Why does CFR converge to Nash in two-player zero-sum?

### Algorithms to Explain:
- **Regret calculation**: Counterfactual regret formula
- **Regret matching**: Converting regrets to strategy
- **CFR iteration**: Traverse tree, update regrets, average

### Practice:
- Calculate regrets for simple poker game (Kuhn poker)
- Trace CFR iteration
- Explain Libratus/Pluribus use of CFR

**Time estimate**: 4-5 days

---

## Study Schedule Recommendations

### Full-Time Study (8 hours/day):
- **Weeks 1-2**: Parts 1-4 (Sequences, Attention, Transformers, LLMs)
- **Week 3**: Parts 5-6 (Fine-tuning, RLHF)
- **Week 4**: Parts 7-10 (GNN, Diffusion, GP, VLA)
- **Week 5**: Parts 11-13 (Game Theory Foundations, Minimax, MCTS)
- **Weeks 6-7**: Parts 14-18 (AlphaZero, PSRO, NeuPL, FSP, CFR)
- **Week 8**: Review and practice problems

### Part-Time Study (2-3 hours/day):
- **Weeks 1-4**: Parts 1-4
- **Weeks 5-7**: Parts 5-6
- **Weeks 8-12**: Parts 7-10
- **Weeks 13-16**: Parts 11-13
- **Weeks 17-22**: Parts 14-18
- **Weeks 23-24**: Review

---

## Daily Study Routine

### Morning (2-3 hours):
1. Read relevant notes (30-45 min)
2. Answer study questions without notes (30-45 min)
3. Implement algorithm in pseudocode (45-60 min)

### Afternoon (2-3 hours):
1. Watch related lectures/papers (if available) (60 min)
2. Practice explaining concepts out loud (30 min)
3. Work through examples by hand (60 min)

### Evening (2-3 hours):
1. Review previous topics (30 min)
2. Flashcards for key formulas (30 min)
3. Mock interview practice (60 min)
4. Write summary in own words (30 min)

---

## Mastery Checklist

Before moving to next part, ensure you can:

- [ ] Explain all concepts to a layperson
- [ ] Derive key formulas from first principles
- [ ] Implement algorithms in pseudocode from memory
- [ ] Compare and contrast with related methods
- [ ] Identify when to use each approach
- [ ] Answer "why" questions (not just "what")
- [ ] Explain limitations and failure modes
- [ ] Connect to real-world applications

---

## Interview Simulation Topics

After completing all parts, practice explaining:

### 20-Minute Deep Dives:
1. "Walk me through transformer architecture end-to-end"
2. "Explain how RLHF works and why it's needed"
3. "How would you build a system like ChatGPT?"
4. "Explain AlphaZero from scratch"
5. "How does CFR solve poker?"

### Comparison Questions:
1. "LSTM vs Transformer - tradeoffs?"
2. "LoRA vs full fine-tuning - when to use each?"
3. "RLHF vs DPO - which is better?"
4. "Minimax vs MCTS - for what games?"
5. "PSRO vs CFR - for what game types?"

### Design Questions:
1. "Design LLM fine-tuning pipeline for legal domain"
2. "How to make diffusion model generate 3D objects?"
3. "Design self-play system for multi-player game"
4. "Build conversational agent with citations"
5. "Scale GNN to billion-node graph"

### Practical Questions:
1. "Why is my transformer training unstable?"
2. "How to reduce LLM inference latency?"
3. "Why does RLHF reward hacking happen?"
4. "Why do GNNs over-smooth?"
5. "How to make MCTS faster?"

---

## Resources to Supplement

### Papers (Essential):
- **Attention is All You Need** (Vaswani et al., 2017)
- **BERT** (Devlin et al., 2018)
- **GPT-3** (Brown et al., 2020)
- **LoRA** (Hu et al., 2021)
- **InstructGPT** (Ouyang et al., 2022) - RLHF
- **Denoising Diffusion Probabilistic Models** (Ho et al., 2020)
- **AlphaZero** (Silver et al., 2017)
- **PSRO** (Lanctot et al., 2017)
- **CFR** (Zinkevich et al., 2007)

### Online Courses:
- Stanford CS224N (NLP)
- Berkeley CS285 (Deep RL)
- DeepMind x UCL Deep Learning Lectures

### Practice Platforms:
- Implement algorithms from scratch
- Kaggle competitions (NLP, vision)
- CodeForces/LeetCode (algorithms)

---

## Final Preparation (Week Before Interview)

### Day 1-2: Core ML
- Review: Attention, Transformer, LLMs
- Practice: Draw architectures, explain formulas

### Day 3-4: Training & Optimization
- Review: Fine-tuning, LoRA, RLHF
- Practice: Design training pipelines

### Day 5-6: Game Theory & RL
- Review: Nash, Minimax, MCTS, AlphaZero, PSRO, CFR
- Practice: Trace algorithms, explain convergence

### Day 7: Mock Interviews
- Full mock interview (60 min)
- Review weak areas
- Practice concise explanations

---

**Remember**: Understanding > Memorization. Focus on *why* things work, not just *what* they are.

Good luck! 🚀
