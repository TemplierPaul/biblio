# Study Tracker - Interview Preparation Checklist

Track your progress through ML/AI and CS interview topics. Check off items as you master them.

**Legend:**
- ⭐ = Essential/High Priority
- 🔥 = Frequently Asked in Interviews
- 📚 = Theory-Heavy
- 💻 = Code Implementation Required

---

## 📊 Progress Overview

**ML/AI Topics**: [ ] 0/11 files complete
**CS Topics**: [ ] 0/4 files complete
**Overall**: [ ] 0/15 files complete

---

## 🤖 Machine Learning & AI

### 01_ANN_Architecture.md

**Part 1: LSTM** 🔥
- [ ] Explain vanishing gradient problem in RNNs
- [ ] Describe LSTM architecture and gates (forget, input, output)
- [ ] Explain how LSTM solves vanishing gradients
- [ ] Compare LSTM vs GRU vs vanilla RNN
- [ ] Implement LSTM forward pass

**Part 2: Attention Mechanism** ⭐🔥
- [ ] Explain attention mechanism intuition
- [ ] Describe scaled dot-product attention formula
- [ ] Explain Query, Key, Value matrices
- [ ] Understand multi-head attention
- [ ] Know why attention is better than RNNs for long sequences

**Part 3: Transformer Architecture** ⭐🔥
- [ ] Explain full Transformer architecture (encoder-decoder)
- [ ] Understand positional encoding and why it's needed
- [ ] Describe self-attention vs cross-attention
- [ ] Explain layer normalization placement
- [ ] Understand masked attention in decoder
- [ ] Implement basic Transformer block

---

### 02_LLM_Training.md

**Part 1: Large Language Models** ⭐🔥
- [ ] Explain autoregressive language modeling
- [ ] Understand pre-training objectives (CLM, MLM)
- [ ] Know scaling laws for LLMs
- [ ] Explain emergent abilities in large models
- [ ] Understand tokenization (BPE, WordPiece)
- [ ] Explain KV cache optimization

**Part 2: Fine-tuning & PEFT** ⭐
- [ ] Understand supervised fine-tuning (SFT)
- [ ] Explain instruction tuning
- [ ] Describe LoRA (Low-Rank Adaptation)
- [ ] Understand QLoRA (quantized LoRA)
- [ ] Compare full fine-tuning vs PEFT methods
- [ ] Know when to use each fine-tuning approach

**Part 3: Alignment & RLHF** 🔥
- [ ] Explain RLHF pipeline (SFT → Reward Model → PPO)
- [ ] Understand reward modeling
- [ ] Describe PPO for language models
- [ ] Explain DPO (Direct Preference Optimization)
- [ ] Compare RLHF vs DPO
- [ ] Understand Constitutional AI

---

### 03_Deep_Learning.md

**Part 1: Graph Neural Networks** 📚
- [ ] Explain message passing framework
- [ ] Describe GCN (Graph Convolutional Networks)
- [ ] Understand GraphSAGE (sampling and aggregation)
- [ ] Explain GAT (Graph Attention Networks)
- [ ] Know applications of GNNs
- [ ] Implement basic message passing

**Part 2: Diffusion Models** 📚
- [ ] Understand forward diffusion process
- [ ] Explain reverse diffusion process
- [ ] Describe DDPM training and sampling
- [ ] Understand DDIM (faster sampling)
- [ ] Explain latent diffusion models
- [ ] Know classifier-free guidance

**Part 3: Vision-Language-Action Models**
- [ ] Understand VLA architecture
- [ ] Explain RT-1 and RT-2 models
- [ ] Know applications in robotics
- [ ] Understand multimodal learning

---

### 04_Classical_ML.md

**Part 1: Gaussian Processes** 📚
- [ ] Explain GP as distribution over functions
- [ ] Understand kernel functions (RBF, Matérn)
- [ ] Describe GP regression
- [ ] Explain uncertainty quantification
- [ ] Know when to use GPs vs neural networks
- [ ] Understand computational complexity of GPs

---

### 05_Game_Theory.md

**Part 1-2: Foundations** 📚
- [ ] Define Nash equilibrium
- [ ] Understand normal form vs extensive form games
- [ ] Explain mixed strategies
- [ ] Describe social dilemmas (prisoner's dilemma, tragedy of commons)

**Part 3: Classical Algorithms** ⭐
- [ ] Explain Minimax algorithm
- [ ] Understand alpha-beta pruning
- [ ] Describe Monte Carlo Tree Search (MCTS)
- [ ] Explain UCT (Upper Confidence Trees)
- [ ] Implement Minimax with alpha-beta pruning

**Part 4: AlphaZero** 🔥
- [ ] Understand AlphaZero architecture
- [ ] Explain self-play training
- [ ] Describe PUCT (Predictor + UCT) formula
- [ ] Understand policy and value networks
- [ ] Know how AlphaZero differs from AlphaGo

**Part 5-6: Population Methods**
- [ ] Explain PSRO (Policy Space Response Oracles)
- [ ] Understand JPSRO and NeuPL
- [ ] Describe equilibrium approximation
- [ ] Know applications in multi-agent systems

**Part 7-8: Learning Methods**
- [ ] Explain Fictitious Self-Play
- [ ] Understand CFR (Counterfactual Regret Minimization)
- [ ] Describe CFR+ improvements
- [ ] Know poker AI applications

---

### 06_Reinforcement_Learning.md ⭐🔥

**Part 1-2: Foundations** ⭐📚
- [ ] Define MDP (Markov Decision Process)
- [ ] Explain state, action, reward, policy
- [ ] Understand Bellman equations (optimality and expectation)
- [ ] Define value function and Q-function
- [ ] Explain discount factor γ
- [ ] Derive Bellman optimality equation

**Part 3: Dynamic Programming** 📚
- [ ] Explain policy evaluation
- [ ] Describe policy iteration algorithm
- [ ] Understand value iteration algorithm
- [ ] Know computational complexity of DP methods

**Part 4: Monte Carlo Methods**
- [ ] Explain first-visit vs every-visit MC
- [ ] Understand Monte Carlo policy evaluation
- [ ] Describe Monte Carlo control
- [ ] Explain exploration-exploitation tradeoff
- [ ] Understand ε-greedy policies

**Part 5: Temporal Difference Learning** ⭐
- [ ] Explain TD(0) algorithm
- [ ] Understand SARSA (on-policy)
- [ ] Describe Q-learning (off-policy)
- [ ] Compare MC vs TD methods
- [ ] Explain bootstrapping
- [ ] Implement Q-learning

**Part 6: Function Approximation** ⭐
- [ ] Understand why function approximation is needed
- [ ] Explain linear function approximation
- [ ] Describe neural network approximation
- [ ] Understand experience replay
- [ ] Know target networks in DQN

**Part 7: Policy Gradient Methods** ⭐📚
- [ ] Explain REINFORCE algorithm
- [ ] Derive policy gradient theorem
- [ ] Understand advantage functions
- [ ] Describe baselines (state-value, action-value)
- [ ] Implement basic REINFORCE

**Part 8: Actor-Critic Methods** ⭐🔥
- [ ] Explain actor-critic architecture
- [ ] Understand A2C (Advantage Actor-Critic)
- [ ] Describe A3C (Asynchronous A3C)
- [ ] Explain PPO (Proximal Policy Optimization) 🔥
- [ ] Understand TRPO (Trust Region PO)
- [ ] Describe GAE (Generalized Advantage Estimation)
- [ ] Implement PPO

**Part 9: Deep Q-Networks** 🔥
- [ ] Explain DQN algorithm
- [ ] Understand Double DQN
- [ ] Describe Dueling DQN architecture
- [ ] Explain Prioritized Experience Replay
- [ ] Understand Rainbow DQN
- [ ] Implement DQN

**Part 10: Continuous Control** 🔥
- [ ] Explain DDPG (Deep Deterministic PG)
- [ ] Understand TD3 (Twin Delayed DDPG)
- [ ] Describe SAC (Soft Actor-Critic) 🔥
- [ ] Compare on-policy vs off-policy for continuous actions
- [ ] Know when to use each algorithm

**Part 11: Advanced Topics**
- [ ] Understand model-based RL
- [ ] Explain multi-agent RL (MARL)
- [ ] Describe imitation learning
- [ ] Understand inverse RL
- [ ] Know meta-RL concepts

**Part 12: Practical Considerations** ⭐
- [ ] Know how to debug RL algorithms
- [ ] Understand hyperparameter tuning
- [ ] Explain evaluation best practices
- [ ] Describe common pitfalls in RL

---

### 07_ML_Fundamentals.md ⭐🔥

**Part 1: Classification & Regression** ⭐
- [ ] Explain generative vs discriminative models
- [ ] Understand SVM (Support Vector Machines) 🔥
- [ ] Describe k-NN (k-Nearest Neighbors)
- [ ] Know when to use each model type
- [ ] Implement SVM and k-NN

**Part 2: Evaluation Metrics** ⭐🔥
- [ ] Explain precision, recall, F1-score 🔥
- [ ] Understand ROC curve and AUC 🔥
- [ ] Describe confusion matrix
- [ ] Explain BLEU and ROUGE for NLP
- [ ] Understand perplexity for language models
- [ ] Know which metric to use when

**Part 3: Loss Functions** ⭐
- [ ] Explain cross-entropy loss 🔥
- [ ] Understand MSE vs MAE
- [ ] Describe focal loss for imbalanced data
- [ ] Explain contrastive and triplet loss
- [ ] Understand KL divergence
- [ ] Know hinge loss for SVMs

**Part 4: Optimizers** ⭐🔥
- [ ] Explain SGD and momentum 🔥
- [ ] Understand Adam optimizer 🔥
- [ ] Describe AdamW (decoupled weight decay)
- [ ] Explain RMSProp
- [ ] Understand learning rate scheduling
- [ ] Know when to use each optimizer

**Part 5: Regularization** ⭐🔥
- [ ] Explain L1 vs L2 regularization 🔥
- [ ] Understand overfitting and underfitting 🔥
- [ ] Describe dropout
- [ ] Explain early stopping
- [ ] Understand data augmentation
- [ ] Know ensemble methods

**Part 6: Model Debugging** 🔥
- [ ] Diagnose high bias vs high variance
- [ ] Handle class imbalance
- [ ] Improve model from 80% to 85%+ accuracy
- [ ] Understand feature engineering
- [ ] Know hyperparameter tuning strategies
- [ ] Explain learning curves

---

### 08_Probability_Statistics.md ⭐📚

**Part 1: Probability Fundamentals** ⭐
- [ ] Define random variables (discrete vs continuous)
- [ ] Explain mean, median, mode
- [ ] Understand correlation vs dependence 🔥
- [ ] Describe PDF and CDF
- [ ] Explain expected value and variance

**Part 2: Distributions** ⭐📚
- [ ] Understand Bernoulli and Binomial
- [ ] Explain Poisson distribution
- [ ] Describe Normal (Gaussian) distribution 🔥
- [ ] Understand Exponential distribution
- [ ] Explain Beta and Gamma distributions
- [ ] Describe t-distribution and Chi-squared
- [ ] Know when to use each distribution

**Part 3: Bayesian Inference** 📚
- [ ] Explain Bayes' theorem 🔥
- [ ] Understand prior, likelihood, posterior
- [ ] Describe conjugate priors
- [ ] Explain Bayesian updating
- [ ] Compare Bayesian vs Frequentist approaches

**Part 4: Statistical Testing** 🔥
- [ ] Understand hypothesis testing 🔥
- [ ] Explain p-values 🔥
- [ ] Describe t-tests (one-sample, two-sample, paired)
- [ ] Understand Type I and Type II errors
- [ ] Explain confidence intervals
- [ ] Know multiple testing correction

**Part 5: Bias-Variance Tradeoff** ⭐🔥
- [ ] Explain bias-variance decomposition 🔥
- [ ] Understand how to balance bias and variance
- [ ] Describe strategies to reduce each
- [ ] Know empirical demonstrations

---

### 09_ML_Systems.md 💻

**Part 1: Scalable ML Systems** 🔥
- [ ] Explain model quantization
- [ ] Understand knowledge distillation
- [ ] Describe distributed serving architecture
- [ ] Explain load balancing
- [ ] Understand caching strategies (Redis, CDN)
- [ ] Describe Kubernetes for ML deployment
- [ ] Explain request batching for throughput
- [ ] Know how to scale to billions of queries

**Part 2: Recommendation Systems** 🔥
- [ ] Understand collaborative filtering
- [ ] Explain matrix factorization
- [ ] Describe two-tower models
- [ ] Understand ANN search (FAISS)
- [ ] Explain candidate generation vs ranking
- [ ] Describe cold-start problem solutions
- [ ] Understand A/B testing for RecSys

---

### 10_Ethics_AI.md ⭐

**Part 1: Ethical Considerations** ⭐
- [ ] Understand fairness definitions (demographic parity, equalized odds) 🔥
- [ ] Explain transparency and explainability
- [ ] Describe privacy concerns (differential privacy, federated learning)
- [ ] Understand accountability and responsibility
- [ ] Explain safety and robustness
- [ ] Know monitoring and drift detection

**Part 2: Bias Mitigation** 🔥
- [ ] Identify sources of bias
- [ ] Explain pre-processing methods (reweighing, resampling)
- [ ] Understand in-processing (fairness constraints)
- [ ] Describe post-processing (threshold optimization)
- [ ] Know how to measure fairness metrics
- [ ] Implement bias mitigation techniques

**Part 3: Practical Considerations**
- [ ] Understand fairness-accuracy tradeoffs 🔥
- [ ] Explain when to prioritize fairness
- [ ] Describe responsible deployment checklist
- [ ] Know how to monitor deployed models

**Part 4: Interview Q&A** 🔥
- [ ] Handle "fairness issue in production" scenario
- [ ] Explain models to non-technical users
- [ ] Discuss fairness vs accuracy tradeoffs
- [ ] Know when to prioritize fairness over accuracy

---

### 11_Math_Foundations.md 📚

**Part 1: Optimization Theory** ⭐📚
- [ ] Explain gradient descent (batch, stochastic, mini-batch) 🔥
- [ ] Understand second-order optimization methods
- [ ] Derive Newton's method 📚
- [ ] Explain learning rate challenges
- [ ] Understand convergence properties

**Part 2: Linear Algebra** ⭐📚
- [ ] Explain Jacobian matrix 🔥
- [ ] Understand Hessian matrix
- [ ] Describe gradient computation
- [ ] Know chain rule for backpropagation
- [ ] Understand computational efficiency

**Part 3: Tensors & Decompositions** 📚
- [ ] Explain tensor operations
- [ ] Understand eigenvalues and eigenvectors 🔥
- [ ] Describe SVD (Singular Value Decomposition) 🔥
- [ ] Explain eigendecomposition
- [ ] Understand QR, Cholesky, LU decompositions
- [ ] Know applications of each decomposition

**Part 4: Generative Models** 🔥
- [ ] Explain VAE architecture and ELBO 🔥
- [ ] Understand reparameterization trick
- [ ] Describe GAN training dynamics 🔥
- [ ] Explain mode collapse and solutions
- [ ] Compare VAE vs GAN
- [ ] Know variants (DCGAN, WGAN, StyleGAN)

**Part 5: Advanced Topics**
- [ ] Explain Mixture of Experts (MoE)
- [ ] Understand gating networks
- [ ] Describe sparse MoE
- [ ] Know load balancing in MoE

---

## 💻 Computer Science Fundamentals

### 01_Data_Structures.md ⭐🔥

**Part 1: Arrays & Strings** ⭐
- [ ] Understand array access patterns O(1) vs O(n)
- [ ] Explain sliding window technique 🔥
- [ ] Describe two-pointer technique 🔥
- [ ] Implement string manipulation algorithms

**Part 2: Linked Lists**
- [ ] Implement singly and doubly linked lists
- [ ] Understand cycle detection (Floyd's algorithm) 🔥
- [ ] Explain linked list reversal
- [ ] Describe dummy node technique

**Part 3: Stacks & Queues** ⭐
- [ ] Implement stack and queue 🔥
- [ ] Explain monotonic stack
- [ ] Understand min-stack implementation
- [ ] Describe applications (DFS, BFS)

**Part 4: Trees** ⭐🔥
- [ ] Understand Binary Search Tree properties 🔥
- [ ] Implement tree traversals (inorder, preorder, postorder) 🔥
- [ ] Explain level-order traversal (BFS)
- [ ] Describe balanced trees (AVL, Red-Black)
- [ ] Understand Lowest Common Ancestor

**Part 5: Heaps & Priority Queues** 🔥
- [ ] Explain heap properties (min-heap, max-heap) 🔥
- [ ] Implement heapify operations
- [ ] Understand heap sort
- [ ] Describe priority queue applications

**Part 6: Hash Tables** ⭐🔥
- [ ] Explain hash functions 🔥
- [ ] Understand collision resolution (chaining, open addressing)
- [ ] Describe load factor and rehashing
- [ ] Know O(1) operations and when they degrade

**Part 7: Graphs** ⭐🔥
- [ ] Understand graph representations (adjacency list/matrix) 🔥
- [ ] Implement DFS and BFS 🔥
- [ ] Explain cycle detection
- [ ] Understand topological sort
- [ ] Implement Dijkstra's algorithm 🔥
- [ ] Describe union-find (disjoint sets)

**Part 8: Tries**
- [ ] Explain trie (prefix tree) structure
- [ ] Implement autocomplete with tries
- [ ] Compare tries vs hash tables

---

### 02_Algorithms.md ⭐🔥

**Part 1: Sorting** ⭐🔥
- [ ] Implement merge sort 🔥
- [ ] Implement quick sort 🔥
- [ ] Understand heap sort
- [ ] Compare time/space complexity of sorting algorithms
- [ ] Know when to use each algorithm

**Part 2: Searching** ⭐🔥
- [ ] Implement binary search 🔥
- [ ] Understand binary search variants
- [ ] Explain two-pointer technique 🔥
- [ ] Know when binary search applies

**Part 3: Dynamic Programming** ⭐🔥
- [ ] Explain memoization vs tabulation 🔥
- [ ] Solve 0/1 Knapsack problem 🔥
- [ ] Implement Longest Common Subsequence
- [ ] Understand DP pattern recognition
- [ ] Describe state transition equations

**Part 4: Greedy Algorithms** 🔥
- [ ] Explain greedy choice property
- [ ] Solve activity selection problem
- [ ] Understand when greedy works vs when it doesn't
- [ ] Describe Huffman coding

**Part 5: Backtracking** 🔥
- [ ] Explain backtracking template 🔥
- [ ] Solve N-Queens problem
- [ ] Generate subsets and permutations
- [ ] Understand pruning strategies

**Part 6: Complexity Analysis** ⭐📚
- [ ] Explain Big O, Omega, Theta notation 🔥
- [ ] Understand Master Theorem
- [ ] Analyze recurrence relations
- [ ] Describe amortized analysis

---

### 03_Systems_Programming.md 💻

**Part 1: Memory Management** ⭐🔥
- [ ] Explain stack vs heap 🔥
- [ ] Understand memory leaks
- [ ] Describe pointers and references
- [ ] Explain RAII principle
- [ ] Understand smart pointers (unique_ptr, shared_ptr)

**Part 2: Operating Systems** ⭐🔥
- [ ] Explain processes vs threads 🔥
- [ ] Understand race conditions
- [ ] Describe mutex and semaphore
- [ ] Explain deadlock (conditions and prevention)
- [ ] Understand virtual memory

**Part 3: Compilation & Linking**
- [ ] Explain compilation stages (preprocessing, compilation, assembly, linking)
- [ ] Understand static vs dynamic linking
- [ ] Describe virtual functions and vtables

**Part 4: Computer Architecture**
- [ ] Explain floating point representation
- [ ] Understand cache hierarchy
- [ ] Describe CPU pipelining
- [ ] Explain branch prediction

**Part 5: Concurrency** 🔥
- [ ] Explain concurrency vs parallelism 🔥
- [ ] Understand thread pools
- [ ] Describe atomic operations
- [ ] Explain GIL (Global Interpreter Lock)

---

### 04_Programming_Languages.md

**Part 1: Python vs C++** 🔥
- [ ] Compare memory management (automatic vs manual)
- [ ] Understand performance differences
- [ ] Explain when to use each language
- [ ] Describe compilation vs interpretation

**Part 2: OOP** ⭐🔥
- [ ] Explain encapsulation, inheritance, polymorphism 🔥
- [ ] Understand composition vs inheritance
- [ ] Describe SOLID principles
- [ ] Know design patterns

**Part 3: Type Systems**
- [ ] Explain static vs dynamic typing
- [ ] Understand strong vs weak typing
- [ ] Describe type hints in Python
- [ ] Explain duck typing

**Part 4: Advanced Concepts**
- [ ] Understand closures 🔥
- [ ] Explain decorators in Python
- [ ] Describe generators and iterators
- [ ] Understand metaprogramming

---

## 📝 Study Tips

### Priority Order for Interview Prep

**Week 1-2: Essentials** ⭐
- [ ] Questions/ML/07_ML_Fundamentals (all parts)
- [ ] Questions/ML/08_Probability_Statistics (Parts 1-4)
- [ ] Questions/CS/01_Data_Structures (Parts 1, 3, 4, 7)
- [ ] Questions/CS/02_Algorithms (Parts 1-3)

**Week 3-4: Deep Learning & Core Algorithms**
- [ ] Questions/ML/01_ANN_Architecture (all parts)
- [ ] Questions/ML/02_LLM_Training (Parts 1-2)
- [ ] Questions/ML/06_Reinforcement_Learning (Parts 1-2, 5, 8)
- [ ] Questions/CS/02_Algorithms (Parts 4-6)

**Week 5-6: Specialization**
Choose based on role:
- **RL Role**: Questions/ML/06 (full), 05
- **Systems Role**: Questions/ML/09, Questions/CS/03
- **Research Role**: Questions/ML/11, 04
- **Ethics/Fairness**: Questions/ML/10

**Week 7-8: Practice & Polish**
- [ ] Mock interviews
- [ ] Code implementations from scratch
- [ ] Review mistakes and weak areas
- [ ] Practice explaining concepts out loud

### Study Strategies

**For Theory (📚)**:
1. Understand the intuition first
2. Derive key equations yourself
3. Explain to someone else (or rubber duck)
4. Connect to practical applications

**For Coding (💻)**:
1. Implement from scratch (no copy-paste)
2. Test with edge cases
3. Analyze time/space complexity
4. Optimize and refactor

**For Frequently Asked (🔥)**:
1. Practice explaining in 2 minutes
2. Prepare examples
3. Know follow-up questions
4. Review weekly

---

## 🎯 Role-Specific Checklists

### ML Engineer
- [ ] Complete 07_ML_Fundamentals
- [ ] Complete 08_Probability_Statistics
- [ ] Complete 01_ANN_Architecture
- [ ] Complete 02_LLM_Training (Parts 1-2)
- [ ] Complete 09_ML_Systems
- [ ] Complete 10_Ethics_AI
- [ ] CS: Data Structures (Parts 1, 4, 7), Algorithms (Parts 1-3)

### Research Scientist
- [ ] Complete 08_Probability_Statistics
- [ ] Complete 11_Math_Foundations
- [ ] Complete 06_Reinforcement_Learning (full)
- [ ] Complete 05_Game_Theory
- [ ] Complete 02_LLM_Training
- [ ] Complete 03_Deep_Learning
- [ ] Complete 04_Classical_ML

### SWE/Generalist
- [ ] Complete CS/01_Data_Structures
- [ ] Complete CS/02_Algorithms
- [ ] Complete CS/03_Systems_Programming
- [ ] Complete ML/07_ML_Fundamentals (Parts 1-2, 4)
- [ ] Complete ML/08_Probability_Statistics (Parts 1, 4)

### MLOps/Infrastructure
- [ ] Complete 09_ML_Systems
- [ ] Complete 07_ML_Fundamentals (Parts 3-4)
- [ ] Complete 10_Ethics_AI (Part 4)
- [ ] Complete CS/03_Systems_Programming
- [ ] Complete CS/01_Data_Structures (Parts 5-7)

---

## ✅ Completion Tracking

Mark when you've fully mastered each file:

### ML/AI Files
- [ ] 01_ANN_Architecture.md
- [ ] 02_LLM_Training.md
- [ ] 03_Deep_Learning.md
- [ ] 04_Classical_ML.md
- [ ] 05_Game_Theory.md
- [ ] 06_Reinforcement_Learning.md
- [ ] 07_ML_Fundamentals.md
- [ ] 08_Probability_Statistics.md
- [ ] 09_ML_Systems.md
- [ ] 10_Ethics_AI.md
- [ ] 11_Math_Foundations.md

### CS Files
- [ ] 01_Data_Structures.md
- [ ] 02_Algorithms.md
- [ ] 03_Systems_Programming.md
- [ ] 04_Programming_Languages.md

---

**Last Updated**: [Add date when you update progress]

**Current Focus**: [What you're studying now]

**Next Up**: [What you'll study next]

**Notes**:
[Add personal notes, weak areas, questions, etc.]
