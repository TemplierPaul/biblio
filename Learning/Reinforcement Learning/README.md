# Reinforcement Learning - Algorithm Reference

Quick reference for RL algorithms, world models, and specialized methods.

---

## World Models

### Dreamer (DreamerV1/V2/V3)
**Paper**: "Dream to Control" (Hafner et al., 2020), "Mastering Atari with Discrete World Models" (Hafner et al., 2021), "Mastering Diverse Domains through World Models" (Hafner et al., 2023)
**What**: Model-based RL learning world model + policy purely in latent space
**How**: RSSM (Recurrent State-Space Model) for dynamics, actor-critic in imagination
**Key innovation**: Learns entirely from imagined rollouts, no direct env interaction for policy
**When to use**: Sample-efficient learning, complex visual observations, long-horizon tasks
**Versions**:
- DreamerV1: Continuous actions (MuJoCo)
- DreamerV2: Discrete actions (Atari), categorical latents
- DreamerV3: Unified (any action space), symlog predictions, robust across domains

### MuZero
**Paper**: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (Schrittwieser et al., 2020)
**What**: Learns latent dynamics model for planning (MCTS in learned representation)
**How**: Model predicts reward, value, policy in latent space; MCTS plans using model
**Key innovation**: No explicit reconstruction, learns what's needed for planning only
**Difference from Dreamer**: Uses MCTS (tree search) vs. policy gradient in imagination
**When to use**: Discrete actions, planning-intensive domains, need strong performance

---

## Policy Gradient Methods

### PPO (Proximal Policy Optimization)
**Paper**: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
**What**: On-policy actor-critic with clipped surrogate objective
**How**: Clip probability ratio to [1-ε, 1+ε], prevents excessive policy updates
**Key innovation**: Simpler than TRPO, more stable than vanilla policy gradient
**When to use**: General-purpose RL, continuous/discrete control, proven stability
**Components**: Policy network, value network (critic), GAE for advantages

### GRPO (Group Relative Policy Optimization)
**Paper**: "DeepSeek-Math: Pushing the Limits of Mathematical Reasoning" (DeepSeek, 2024)
**What**: Memory-efficient PPO variant for LLM alignment (no critic network)
**How**: Use group-relative advantages instead of learned value function
**Key innovation**: 40-50% memory reduction, dynamic gradient coefficients
**When to use**: LLM fine-tuning, memory constraints, have reward models
**Difference from PPO**: Eliminates critic, uses normalized group rewards as advantages
**Results**: +5% on GSM8K, +5% on MATH (DeepSeek-Math 7B)

---

## Specialized RL Components

### RNN Policies
**What**: Recurrent neural networks as policy/value function (LSTM, GRU)
**Why**: Handle partial observability, temporal dependencies, memory requirements
**When to use**: POMDPs, tasks requiring memory (navigation, multi-step reasoning)
**Key considerations**: Hidden state management, truncated BPTT, burn-in periods

### RL Methods for PSRO
**What**: Specialized RL algorithms optimized for PSRO best-response training
**Covered**: Adaptations of PPO, SAC, TRPO for game-theoretic training
**Key considerations**: Exploitability metrics, opponent modeling, multi-agent credit

---

## Core RL Algorithms (See Questions/ML/06_Reinforcement_Learning.md)

For comprehensive RL coverage including:
- **Value-based**: Q-learning, DQN, Double DQN, Dueling DQN, Rainbow
- **Policy gradient**: REINFORCE, A2C, A3C, PPO, TRPO
- **Actor-critic**: SAC, TD3, DDPG
- **Model-based**: Dyna, MVE
- **Multi-agent**: MARL, QMIX, COMA

See **Questions/ML/06_Reinforcement_Learning.md** (12 parts, comprehensive coverage)

---

## Summary by Use Case

| Use Case | Algorithm | Why |
|----------|-----------|-----|
| General-purpose RL | PPO | Stable, proven, widely used |
| LLM alignment | GRPO | Memory-efficient, designed for language models |
| Sample efficiency, visual | Dreamer | Learns in imagination, world model |
| Planning + learned model | MuZero | MCTS with latent dynamics |
| Partial observability | RNN Policies | Memory via recurrence |
| Multi-agent games | RL for PSRO | Game-theoretic adaptations |
| Foundational RL algorithms | See Questions/ML/06 | Comprehensive coverage (Q-learning, DQN, SAC, etc.) |

---

## Relationships

- **Model-free RL** → **Dreamer**: Add world model, imagine rollouts
- **AlphaZero** → **MuZero**: Perfect info → Imperfect info, learned model
- **Feed-forward policies** → **RNN Policies**: Add memory/recurrence
- **Single-agent RL** → **RL for PSRO**: Multi-agent, game theory integration

---

## Key Distinctions

**Model-Based vs. Model-Free**:
- Model-free (Q-learning, PPO): Learn policy/value directly from experience
- Model-based (Dreamer, MuZero): Learn environment dynamics, use for planning/imagination

**Planning-Based vs. Gradient-Based**:
- Planning (MuZero, AlphaZero): MCTS tree search with model
- Gradient (Dreamer, PPO): Policy gradient updates

**Latent vs. Pixel Reconstruction**:
- Latent (MuZero, DreamerV2/V3): Model predicts in abstract space
- Reconstruction (Early world models): Explicitly reconstruct pixels (wasteful)

---

**See individual files for detailed implementations, hyperparameters, and training procedures.**
