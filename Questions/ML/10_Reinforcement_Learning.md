# Reinforcement Learning - Interview Q&A

**How to use**: Questions are formatted as collapsible headings. Try to answer before expanding.

---


## Table of Contents

- [[#Part 1: MDP Foundations]]
  - [[#What is a Markov Decision Process (MDP)?]]
  - [[#What's the Markov property?]]
  - [[#Difference between deterministic and stochastic MDPs?]]
  - [[#What's a policy?]]
  - [[#What's the return (cumulative reward)?]]
  - [[#Why do we discount future rewards?]]
  - [[#Episodic vs continuing tasks?]]
- [[#Part 2: Value Functions & Bellman Equations]]
  - [[#What's the state-value function V^π(s)?]]
  - [[#What's the action-value function Q^π(s,a)?]]
  - [[#Write the Bellman expectation equation for V^π]]
  - [[#Write the Bellman expectation equation for Q^π]]
  - [[#What's the optimal value function?]]
  - [[#Write the Bellman optimality equation for V*]]
  - [[#Write the Bellman optimality equation for Q*]]
  - [[#How are V* and Q* related?]]
- [[#Part 3: Dynamic Programming]]
  - [[#What's policy evaluation?]]
  - [[#What's policy improvement?]]
  - [[#What's policy iteration?]]
  - [[#What's value iteration?]]
  - [[#Policy iteration vs value iteration - which is better?]]
  - [[#Why does dynamic programming require a model?]]
  - [[#What's the curse of dimensionality in DP?]]
- [[#Part 4: Monte Carlo Methods]]
  - [[#What's Monte Carlo RL?]]
  - [[#Explain first-visit vs every-visit MC]]
  - [[#Write the MC update rule for V(s)]]
  - [[#How does MC estimate Q(s,a)?]]
  - [[#What's the exploration-exploitation dilemma?]]
  - [[#Explain ε-greedy exploration]]
  - [[#On-policy vs off-policy learning?]]
- [[#Part 5: Temporal Difference Learning]]
  - [[#What's temporal difference (TD) learning?]]
  - [[#Write the TD(0) update rule]]
  - [[#What's the TD error?]]
  - [[#TD vs Monte Carlo - bias and variance?]]
  - [[#What's SARSA?]]
  - [[#What's Q-learning?]]
  - [[#SARSA vs Q-learning - key difference?]]
  - [[#What's Expected SARSA?]]
  - [[#What's n-step TD?]]
- [[#Part 6: Function Approximation]]
  - [[#Why do we need function approximation?]]
  - [[#What's a linear function approximator?]]
  - [[#Write the gradient descent update for V(s)]]
  - [[#What's the deadly triad in RL?]]
  - [[#What's experience replay?]]
  - [[#What's a target network?]]
- [[#Part 7: Policy Gradient Methods]]
  - [[#What's a policy gradient method?]]
  - [[#Write the policy gradient theorem]]
  - [[#What's REINFORCE?]]
  - [[#What's a baseline in policy gradients?]]
  - [[#What's the advantage function?]]
- [[#Part 8: Actor-Critic Methods]]
  - [[#What's an actor-critic method?]]
  - [[#Write the basic actor-critic update]]
  - [[#What's A2C (Advantage Actor-Critic)?]]
  - [[#What's A3C?]]
  - [[#What's GAE (Generalized Advantage Estimation)?]]
  - [[#What's PPO (Proximal Policy Optimization)?]]
  - [[#What's TRPO?]]
- [[#Part 9: Deep Q-Networks (DQN)]]
  - [[#What's DQN?]]
  - [[#What problem does Double DQN solve?]]
  - [[#What problem does Dueling DQN solve?]]
  - [[#What's prioritized experience replay?]]
  - [[#What's Rainbow DQN?]]
- [[#Part 10: Continuous Control]]
  - [[#Why can't DQN handle continuous actions?]]
  - [[#What's DDPG?]]
  - [[#What's TD3 (Twin Delayed DDPG)?]]
  - [[#What's SAC (Soft Actor-Critic)?]]
  - [[#DDPG vs TD3 vs SAC - which to use?]]
- [[#Part 11: Advanced Topics]]
  - [[#What's model-based RL?]]
  - [[#What's the Dyna architecture?]]
  - [[#What's imitation learning?]]
  - [[#What's inverse reinforcement learning?]]
  - [[#What's multi-agent RL (MARL)?]]
  - [[#What's reward shaping?]]
  - [[#What's hierarchical RL?]]
- [[#Part 12: Practical Considerations]]
  - [[#How to debug RL algorithms?]]
  - [[#What hyperparameters matter most?]]
  - [[#How to choose RL algorithm for a task?]]
  - [[#Common RL pitfalls?]]
  - [[#How to evaluate RL agents?]]
- [[#Foundations]]
  - [[#What is Continual Reinforcement Learning (CRL)?]]
  - [[#How is CRL formally different from related settings?]]
  - [[#What scenario settings exist in CRL?]]
- [[#The Core Challenge]]
  - [[#What's the main challenge in CRL?]]
  - [[#What is catastrophic forgetting in RL?]]
  - [[#What is plasticity loss and how is it different from forgetting?]]
- [[#The Reset and Reinitialization Literature]]
  - [[#Why should I care about periodic resets?]]
- [[#Taxonomy of CRL Methods]]
  - [[#What are the 4 types of CRL methods?]]
  - [[#How does EWC work?]]
  - [[#How does Progressive Neural Networks (PNN) work?]]
  - [[#How does policy decomposition work?]]
  - [[#How does CLEAR work?]]
- [[#Successor Features and Transfer]]
  - [[#What are Successor Features and why do they matter for CRL?]]
  - [[#What is Generalized Policy Improvement (GPI)?]]
  - [[#What are Universal Value Function Approximators (UVFA)?]]
- [[#Meta-RL as Continual Learning]]
  - [[#How does meta-RL relate to CRL?]]
  - [[#What is AdA and why does it matter?]]
  - [[#What is Algorithm Distillation?]]
  - [[#Meta-RL vs CRL: when to use which?]]
- [[#Representation Learning and Foundation Models]]
  - [[#How do foundation models change CRL?]]
- [[#Model-Based Continual RL]]
  - [[#How does model-based CRL work?]]
- [[#Reward-Focused Methods]]
  - [[#How does reward shaping work in CRL?]]
  - [[#What's intrinsic motivation in CRL?]]
- [[#Multi-Agent Continual RL]]
  - [[#How does CRL interact with multi-agent settings?]]
  - [[#How are population-based methods implicit CRL?]]
- [[#Offline-to-Online Continual RL]]
  - [[#What is the offline-to-online CRL paradigm?]]
- [[#Curriculum and Task Ordering]]
  - [[#Does task order matter in CRL?]]
- [[#Safety and Constraints]]
  - [[#What about safety in CRL?]]
- [[#Evaluation]]
  - [[#How to evaluate CRL agents?]]
  - [[#What benchmarks exist?]]
- [[#Theory]]
  - [[#What theoretical results exist for CRL?]]
- [[#Comparisons]]
  - [[#CRL vs Multi-Task Learning?]]
  - [[#CRL vs Transfer Learning?]]
  - [[#CRL vs Supervised Continual Learning?]]
- [[#Open Problems and Future Directions]]
  - [[#What are the biggest open challenges?]]
- [[#Quick-Fire Interview Questions]]
- [[#Key References]]
  - [[#Must-Read (Priority Order)]]
  - [[#Classic Methods]]
  - [[#Meta-RL Connection]]
  - [[#Evaluation]]
  - [[#Multi-Agent]]

---

## Part 1: MDP Foundations

### What is a Markov Decision Process (MDP)?

**MDP**: Mathematical framework for sequential decision-making.

**Components** (5-tuple):
1. **States** $\mathcal{S}$: Set of all possible states
2. **Actions** $\mathcal{A}$: Set of all possible actions
3. **Transition** $P(s'|s,a)$: Probability of reaching $s'$ from $s$ via action $a$
4. **Reward** $R(s,a,s')$: Immediate reward for transition
5. **Discount** $\gamma \in [0,1]$: Future reward discount factor

**Goal**: Find policy $\pi(a|s)$ that maximizes expected cumulative reward.

**Example**: Robot navigation (states=positions, actions=moves, rewards=goal bonus)

### What's the Markov property?

**Markov property**: Future is independent of past given present.

**Formal**:
$$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0, a_0) = P(s_{t+1} | s_t, a_t)$$

**Intuition**: Current state contains all information needed to predict future.

**Example**:
- **Markov**: Chess (board state is sufficient)
- **Non-Markov**: Poker without memory of previous bets

**Why important**: Enables dynamic programming and efficient algorithms.

### Difference between deterministic and stochastic MDPs?

**Deterministic MDP**:
- Transition: $s' = f(s, a)$ (fixed)
- Example: Tic-tac-toe, Rubik's cube

**Stochastic MDP**:
- Transition: $s' \sim P(\cdot|s,a)$ (probabilistic)
- Example: Robot locomotion (slippery ground), card games

**Most real-world problems**: Stochastic (uncertainty in environment)

**Algorithm implications**:
- Deterministic: Can use simpler planning
- Stochastic: Need expected value calculations

### What's a policy?

**Policy** $\pi$: Mapping from states to actions (agent's behavior).

**Deterministic**: $a = \pi(s)$
- Single action per state

**Stochastic**: $a \sim \pi(\cdot|s)$
- Probability distribution over actions
- More general

**Goal of RL**: Find optimal policy $\pi^*$ maximizing expected return.

**Example**:
- Deterministic: "Always go north in state A"
- Stochastic: "Go north 70%, east 30% in state A"

### What's the return (cumulative reward)?

**Return** $G_t$: Total discounted reward from time $t$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$$

**Components**:
- $R_t$: Immediate reward at time $t$
- $\gamma$: Discount factor (0 to 1)

**Why discount**:
- $\gamma < 1$: Prefer immediate rewards, ensures convergence
- $\gamma = 0$: Only immediate reward matters (myopic)
- $\gamma \to 1$: All future rewards equally important (far-sighted)

**Typical values**: $\gamma \in [0.9, 0.99]$

### Why do we discount future rewards?

**Practical reasons**:
1. **Uncertainty**: Future less predictable
2. **Mathematical**: Ensures finite return (infinite horizon)
3. **Modeling**: Preference for sooner rewards (time value)
4. **Computational**: Bounded values easier to optimize

**Mathematical**:
- Without discount ($\gamma=1$): $\sum_{t=0}^\infty R_t$ may diverge
- With discount ($\gamma<1$): $\sum_{t=0}^\infty \gamma^t R_t < \frac{R_{max}}{1-\gamma}$ (bounded)

**Special case**: Episodic tasks (finite horizon) can use $\gamma=1$.

### Episodic vs continuing tasks?

**Episodic**:
- **Finite horizon**: Task has clear end (terminal state)
- Return: $G_t = R_{t+1} + R_{t+2} + \ldots + R_T$
- Examples: Board games, reaching goal
- Can use $\gamma = 1$

**Continuing**:
- **Infinite horizon**: Task never ends
- Return: $G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$
- Examples: Server maintenance, robot operation
- Need $\gamma < 1$

**Unified formulation**: Treat terminal state as absorbing (0 reward, transitions to itself).

---

## Part 2: Value Functions & Bellman Equations

### What's the state-value function V^π(s)?

**State-value function** $V^\pi(s)$: Expected return starting from state $s$, following policy $\pi$.

$$V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^\infty \gamma^k R_{t+k+1} \bigg| s_t = s\right]$$

**Interpretation**: "How good is state $s$ under policy $\pi$?"

**Example** (Grid world):
- $V^\pi(\text{goal}) = 0$ (terminal, no future reward)
- $V^\pi(\text{near goal}) > V^\pi(\text{far from goal})$

### What's the action-value function Q^π(s,a)?

**Action-value function** (Q-function) $Q^\pi(s,a)$: Expected return starting from state $s$, taking action $a$, then following $\pi$.

$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]$$

**Interpretation**: "How good is action $a$ in state $s$ under policy $\pi$?"

**Relation to V**:
$$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a)$$

**Why useful**: Can extract policy: $\pi(s) = \arg\max_a Q^\pi(s,a)$

### Write the Bellman expectation equation for V^π

**Bellman expectation equation**:

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

**Intuition**: Value of state = expected immediate reward + discounted value of next state

**Breakdown**:
1. $\sum_a \pi(a|s)$: Average over actions according to policy
2. $\sum_{s'} P(s'|s,a)$: Average over next states
3. $R(s,a,s')$: Immediate reward
4. $\gamma V^\pi(s')$: Discounted future value

**Recursive**: Expresses $V$ in terms of itself (dynamic programming).

### Write the Bellman expectation equation for Q^π

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

**Or equivalently**:
$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

**Intuition**: Q-value = expected immediate reward + discounted value of next state (averaged over policy).

### What's the optimal value function?

**Optimal state-value function** $V^*(s)$:
$$V^*(s) = \max_\pi V^\pi(s)$$

Maximum value achievable in state $s$ under any policy.

**Optimal action-value function** $Q^*(s,a)$:
$$Q^*(s,a) = \max_\pi Q^\pi(s,a)$$

**Optimal policy** $\pi^*$:
$$\pi^*(s) = \arg\max_a Q^*(s,a)$$

**Key property**: $V^{\pi^*}(s) = V^*(s)$ for all $s$

### Write the Bellman optimality equation for V*

**Bellman optimality equation**:

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$$

**Intuition**: Optimal value = max over actions of (immediate reward + discounted future optimal value)

**Difference from expectation**: $\max_a$ instead of $\sum_a \pi(a|s)$ (no policy averaging)

**Nonlinear**: Due to $\max$ operator (harder to solve than expectation equation).

### Write the Bellman optimality equation for Q*

$$Q^*(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$$

**Key difference**: $\max_{a'}$ in next state (choose best action)

**Policy extraction**: Once we have $Q^*$, optimal policy is:
$$\pi^*(s) = \arg\max_a Q^*(s,a)$$

**Why Q is useful**: Don't need model $P(s'|s,a)$ to extract policy.

### How are V* and Q* related?

$$V^*(s) = \max_a Q^*(s,a)$$

$$Q^*(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$$

**Combined**:
$$Q^*(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$$

**In practice**: Often learn $Q^*$ directly (model-free) rather than $V^*$.

---

## Part 3: Dynamic Programming

### What's policy evaluation?

**Policy evaluation**: Compute $V^\pi$ for a given policy $\pi$.

**Algorithm** (iterative):
```
Initialize V(s) = 0 for all s
Repeat until convergence:
    For each state s:
        V(s) ← Σ_a π(a|s) Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
```

**Uses**: Bellman expectation equation as update rule.

**Convergence**: Guaranteed as $V_k \to V^\pi$ (contraction mapping).

**Complexity**: $O(|\mathcal{S}|^2 |\mathcal{A}|)$ per iteration.

### What's policy improvement?

**Policy improvement**: Given $V^\pi$, construct better policy $\pi'$.

**Greedy improvement**:
$$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

**Or using Q-values**:
$$\pi'(s) = \arg\max_a Q^\pi(s,a)$$

**Policy improvement theorem**: $V^{\pi'}(s) \geq V^\pi(s)$ for all $s$ (guaranteed improvement).

### What's policy iteration?

**Policy iteration**: Alternate between evaluation and improvement until convergence.

**Algorithm**:
```
1. Initialize π arbitrarily
2. Repeat:
     a) Policy Evaluation: Compute V^π
     b) Policy Improvement: π' ← greedy(V^π)
     c) If π' = π, stop
     d) π ← π'
```

**Convergence**: Finite iterations to $\pi^*$ (for finite MDPs).

**Complexity**: Each iteration is expensive (full evaluation).

### What's value iteration?

**Value iteration**: Directly iterate Bellman optimality equation.

**Algorithm**:
```
Initialize V(s) = 0 for all s
Repeat until convergence:
    For each state s:
        V(s) ← max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
```

**Combines**: Evaluation and improvement in one step.

**Convergence**: $V_k \to V^*$ as $k \to \infty$

**Stopping**: When $\max_s |V_{k+1}(s) - V_k(s)| < \epsilon$

**Extract policy**: $\pi(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]$

### Policy iteration vs value iteration - which is better?

**Policy iteration**:
- ✅ Fewer iterations (each iteration more expensive)
- ✅ Exact evaluation each iteration
- ❌ Expensive: Full evaluation per iteration

**Value iteration**:
- ✅ Simpler: Single update per state
- ✅ Faster per iteration
- ❌ More iterations needed

**In practice**:
- **Small MDPs**: Policy iteration (fewer iterations)
- **Large MDPs**: Value iteration or truncated policy iteration
- **Modern**: Neither (use sampling-based methods)

### Why does dynamic programming require a model?

**DP requires**:
- Transition probabilities $P(s'|s,a)$
- Reward function $R(s,a,s')$

**Why**: Bellman equations involve **expected values**:
$$V(s) = \max_a \underbrace{\sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]}_{\text{need model}}$$

**Limitation**: Model often unknown in real-world problems.

**Solution**: Model-free methods (MC, TD, Q-learning).

### What's the curse of dimensionality in DP?

**Problem**: Computational cost grows exponentially with state space dimensions.

**Example**:
- 10 state variables, each with 10 values
- Total states: $10^{10}$ (10 billion)
- DP update: Must iterate over all states

**Why DP struggles**:
- Need to store $V(s)$ or $Q(s,a)$ for all states
- Need to update all states each iteration

**Solutions**:
- Approximation (function approximation)
- Sampling (MC, TD)
- Asynchronous DP (update subset of states)

---

## Part 4: Monte Carlo Methods

### What's Monte Carlo RL?

**Monte Carlo (MC)**: Learn from complete episodes (sample returns).

**Key idea**: Estimate value by averaging observed returns.

**Requirements**:
- Episodic tasks (must terminate)
- No model needed (model-free)

**Advantage**: Unbiased estimates (true return samples).

**Disadvantage**: High variance, only learns at episode end.

### Explain first-visit vs every-visit MC

**First-visit MC**:
- For each episode, only use first occurrence of state $s$
- Average returns from first visits

**Every-visit MC**:
- Use all occurrences of state $s$ in episode
- Average returns from every visit

**Both converge** to $V^\pi(s)$ as episodes → ∞

**In practice**: First-visit more common (simpler analysis).

**Example** (episode: A→B→A→B→terminal, reward +1 at end):
- First-visit: Count A once (first visit)
- Every-visit: Count A twice (both visits)

### Write the MC update rule for V(s)

**Incremental MC update**:

$$V(s) \leftarrow V(s) + \alpha[G_t - V(s)]$$

Where:
- $G_t$: Observed return from state $s$ at time $t$
- $\alpha$: Learning rate (step size)

**Interpretation**: Move estimate toward observed return.

**Error term**: $G_t - V(s)$ is Monte Carlo error.

**Alternative** (averaging all returns):
$$V(s) = \frac{1}{N(s)}\sum_{i=1}^{N(s)} G_i$$
where $N(s)$ is visit count.

### How does MC estimate Q(s,a)?

**Monte Carlo for Q**:
1. Follow policy $\pi$, generate episode
2. For each $(s,a)$ pair visited, observe return $G$
3. Update: $Q(s,a) \leftarrow Q(s,a) + \alpha[G - Q(s,a)]$

**Why Q not V**: Can extract policy without model:
$$\pi(s) = \arg\max_a Q(s,a)$$

**Exploration problem**: Need to visit all $(s,a)$ pairs (exploration vs exploitation).

### What's the exploration-exploitation dilemma?

**Dilemma**: Balance between:
- **Exploitation**: Choose best known action (maximize reward)
- **Exploration**: Try new actions (gather information)

**Problem**: Optimal policy may never be found without exploration.

**Example**: Two-armed bandit
- Arm A: Known reward 1
- Arm B: Unknown reward (could be 0 or 10)
- Exploit: Always pull A (get 1)
- Explore: Sometimes pull B (might discover 10)

**Solutions**: ε-greedy, softmax, UCB.

### Explain ε-greedy exploration

**ε-greedy policy**:
- With probability $\epsilon$: Choose random action (explore)
- With probability $1-\epsilon$: Choose best action (exploit)

$$\pi(a|s) = \begin{cases}
1-\epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s,a') \\
\frac{\epsilon}{|\mathcal{A}|} & \text{otherwise}
\end{cases}$$

**Typical**: $\epsilon = 0.1$ (10% exploration)

**Annealing**: Decay $\epsilon$ over time ($\epsilon \to 0$)

**Advantage**: Simple, guaranteed exploration.

**Disadvantage**: Explores uniformly (doesn't prefer promising actions).

### On-policy vs off-policy learning?

**On-policy**: Learn value of policy being followed.
- Policy used for **exploration** = policy being **learned**
- Example: SARSA
- Simpler, more stable

**Off-policy**: Learn value of target policy while following different behavior policy.
- **Target policy** (learned): $\pi$ (often greedy)
- **Behavior policy** (followed): $b$ (exploratory)
- Example: Q-learning
- More flexible, better data efficiency

**Trade-off**:
- On-policy: Easier, but less sample efficient
- Off-policy: Harder (importance sampling), but more general

---

## Part 5: Temporal Difference Learning

### What's temporal difference (TD) learning?

**TD learning**: Combine MC and DP ideas.
- Like MC: Model-free (learn from experience)
- Like DP: Bootstrap (use estimates for updates)

**Key idea**: Update estimates based on other estimates (bootstrapping).

**Advantage over MC**: Learn online (every step, not episode end).

**Advantage over DP**: No model needed.

**Most important RL algorithm family**: TD(0), SARSA, Q-learning.

### Write the TD(0) update rule

**TD(0) update**:

$$V(s_t) \leftarrow V(s_t) + \alpha[R_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

**TD target**: $R_{t+1} + \gamma V(s_{t+1})$

**TD error**: $\delta_t = R_{t+1} + \gamma V(s_{t+1}) - V(s_t)$

**Interpretation**: Update current estimate toward one-step lookahead estimate.

**Key**: Uses $V(s_{t+1})$ (bootstrap) instead of true return $G_t$ (MC).

### What's the TD error?

**TD error** $\delta_t$:

$$\delta_t = R_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

**Interpretation**: Difference between expected value and current estimate.

**Signal**: How much to update value estimate.
- $\delta > 0$: State better than expected (increase value)
- $\delta < 0$: State worse than expected (decrease value)

**Important**: Central to actor-critic methods (advantage estimation).

### TD vs Monte Carlo - bias and variance?

**Monte Carlo**:
- ✅ **Unbiased**: $G_t$ is true return sample
- ❌ **High variance**: Return depends on many random transitions
- Converges slowly, but to correct value

**TD**:
- ❌ **Biased**: $R + \gamma V(s')$ uses estimate $V(s')$
- ✅ **Low variance**: Only depends on one transition
- Faster convergence, lower noise

**Trade-off**:
- MC: Slow but accurate
- TD: Fast but initially biased

**In practice**: TD often better (lower variance outweighs bias).

### What's SARSA?

**SARSA** (State-Action-Reward-State-Action): On-policy TD control.

**Update rule**:
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[R_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

**Name origin**: Uses $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ tuple.

**On-policy**: Learns Q-value of policy being followed (e.g., ε-greedy).

**Algorithm**:
```
Initialize Q(s,a) arbitrarily
Repeat for each episode:
    Initialize s
    Choose a from s using policy derived from Q (e.g., ε-greedy)
    Repeat for each step:
        Take action a, observe r, s'
        Choose a' from s' using policy derived from Q
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        s ← s', a ← a'
```

### What's Q-learning?

**Q-learning**: Off-policy TD control.

**Update rule**:
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[R_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$$

**Key difference from SARSA**: Uses $\max_{a'} Q(s', a')$ instead of $Q(s', a')$.

**Off-policy**: Learns optimal policy while following exploratory policy.

**Algorithm**:
```
Initialize Q(s,a) arbitrarily
Repeat for each episode:
    Initialize s
    Repeat for each step:
        Choose a from s using policy derived from Q (e.g., ε-greedy)
        Take action a, observe r, s'
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        s ← s'
```

**Converges to Q***: With sufficient exploration.

### SARSA vs Q-learning - key difference?

**SARSA** (on-policy):
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$$
- Uses action $a'$ actually taken (from policy)
- Learns value of policy being followed

**Q-learning** (off-policy):
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$
- Uses max over actions (not action taken)
- Learns value of optimal policy

**Example** (cliff walking):
- SARSA: Learns safe path (accounts for exploration)
- Q-learning: Learns risky path near cliff (assumes optimal actions)

**When to use**:
- SARSA: Safety critical (learns exploratory policy)
- Q-learning: Want optimal policy (more sample efficient)

### What's Expected SARSA?

**Expected SARSA**: Hybrid of SARSA and Q-learning.

**Update rule**:
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \sum_{a'} \pi(a'|s') Q(s',a') - Q(s,a)]$$

**Takes expectation** over next actions (instead of sampling).

**Properties**:
- **Deterministic optimal policy**: Equivalent to Q-learning
- **Stochastic policy**: Between SARSA and Q-learning
- **Lower variance** than SARSA (expectation vs sample)

**Complexity**: More computation (sum over actions).

### What's n-step TD?

**n-step TD**: Generalization between TD(0) and MC.

**n-step return**:
$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \ldots + \gamma^{n-1} R_{t+n} + \gamma^n V(s_{t+n})$$

**Update**:
$$V(s_t) \leftarrow V(s_t) + \alpha[G_t^{(n)} - V(s_t)]$$

**Spectrum**:
- n=1: TD(0) (one-step)
- n=∞: Monte Carlo (complete return)

**Trade-off**: Bias-variance
- Small n: Low variance, high bias
- Large n: High variance, low bias

---

## Part 6: Function Approximation

### Why do we need function approximation?

**Problem**: Tabular methods (store V(s) or Q(s,a) for each state) don't scale.

**Example**:
- Atari game: $10^{9}$ states (256×256 pixels × 4 frames)
- Can't store table

**Solution**: **Function approximation**
- Represent $\hat{V}(s; \mathbf{w})$ or $\hat{Q}(s,a; \mathbf{w})$ with parameters $\mathbf{w}$
- Generalize to unseen states

**Types**: Linear, neural networks, decision trees.

### What's a linear function approximator?

**Linear approximation**:
$$\hat{V}(s; \mathbf{w}) = \mathbf{w}^T \phi(s) = \sum_i w_i \phi_i(s)$$

Where:
- $\phi(s)$: Feature vector (basis functions)
- $\mathbf{w}$: Weight vector (parameters)

**Example** (grid world):
- $\phi_1(s)$ = distance to goal
- $\phi_2(s)$ = number of obstacles nearby
- $\hat{V}(s) = w_1 \phi_1(s) + w_2 \phi_2(s)$

**Advantage**: Convex optimization, convergence guarantees.

**Limitation**: Limited expressiveness (can't learn nonlinear functions well).

### Write the gradient descent update for V(s)

**Objective**: Minimize squared error
$$J(\mathbf{w}) = \mathbb{E}_\pi[(V^\pi(s) - \hat{V}(s; \mathbf{w}))^2]$$

**Gradient descent**:
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha[V^\pi(s) - \hat{V}(s; \mathbf{w})] \nabla_\mathbf{w} \hat{V}(s; \mathbf{w})$$

**With TD target** (since we don't know $V^\pi$):
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha[R + \gamma \hat{V}(s'; \mathbf{w}) - \hat{V}(s; \mathbf{w})] \nabla_\mathbf{w} \hat{V}(s; \mathbf{w})$$

**For neural network**: $\nabla_\mathbf{w} \hat{V}$ computed via backpropagation.

### What's the deadly triad in RL?

**Deadly triad**: Combination that can cause divergence:

1. **Function approximation** (not tabular)
2. **Bootstrapping** (TD learning)
3. **Off-policy learning** (Q-learning)

**Problem**: Approximate + bootstrap + off-policy can diverge (instability).

**Example**: Q-learning with neural networks can diverge.

**Solutions**:
- Experience replay (DQN)
- Target networks (DQN)
- Remove one element (e.g., on-policy like A2C)

### What's experience replay?

**Experience replay**: Store transitions in buffer, sample randomly for training.

**Algorithm**:
1. Store transition $(s, a, r, s')$ in replay buffer $\mathcal{D}$
2. Sample random mini-batch from $\mathcal{D}$
3. Train on mini-batch

**Advantages**:
1. **Breaks correlation**: Consecutive samples are correlated; random sampling breaks this
2. **Sample efficiency**: Reuse experiences multiple times
3. **Stability**: Smooth out distribution changes

**Used in**: DQN, DDPG, SAC, TD3.

### What's a target network?

**Target network**: Separate, slowly-updated network for computing TD targets.

**Setup**:
- **Online network** $Q(s,a; \theta)$: Updated every step
- **Target network** $Q(s,a; \theta^-)$: Updated every $C$ steps

**Q-learning with target network**:
$$\text{Loss} = [r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s,a; \theta)]^2$$

**Update**: $\theta^- \leftarrow \theta$ every $C$ steps (or soft update: $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$).

**Why**: Prevents moving target problem (target changes during training).

**Used in**: DQN and variants.

---

## Part 7: Policy Gradient Methods

### What's a policy gradient method?

**Policy gradient**: Directly parametrize and optimize policy $\pi_\theta(a|s)$.

**Contrast value-based**:
- Value-based: Learn Q(s,a), derive policy (implicit)
- Policy gradient: Learn π(a|s) directly (explicit)

**Advantages**:
- Handle continuous actions naturally
- Can learn stochastic policies
- Better convergence properties (in some cases)

**Disadvantage**: High variance, sample inefficient.

### Write the policy gradient theorem

**Policy gradient theorem**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s,a)\right]$$

or with advantage:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) A^{\pi_\theta}(s,a)\right]$$

**Intuition**: Increase probability of actions with positive advantage (better than average).

**Components**:
- $\nabla_\theta \log \pi_\theta(a|s)$: Direction to increase $\pi(a|s)$
- $Q(s,a)$ or $A(s,a)$: How good is action (credit assignment)

**Key**: Can estimate gradient from samples (Monte Carlo).

### What's REINFORCE?

**REINFORCE**: Monte Carlo policy gradient algorithm.

**Update**:
$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) G_t$$

Where $G_t$ is the return from time $t$.

**Algorithm**:
```
Initialize θ
Repeat:
    Generate episode s_0, a_0, r_1, ..., s_T
    For each step t:
        G ← return from step t
        θ ← θ + α∇_θ log π_θ(a_t|s_t) G
```

**Properties**:
- Unbiased gradient estimate
- High variance (full return)
- On-policy

**Improvement**: Use baseline to reduce variance.

### What's a baseline in policy gradients?

**Baseline**: Subtract from return to reduce variance without adding bias.

**Modified gradient**:
$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) (G_t - b(s))\right]$$

**Common baseline**: State value $b(s) = V(s)$

**Why it works**:
- Bias: $\mathbb{E}[\nabla \log \pi(a|s) b(s)] = b(s) \mathbb{E}[\nabla \log \pi] = b(s) \cdot 0 = 0$
- Variance: Reduced when $b \approx G$

**Best baseline**: $b(s) = V(s)$ leads to advantage:
$$A(s,a) = Q(s,a) - V(s)$$

**Interpretation**: How much better is action than average?

### What's the advantage function?

**Advantage function**:
$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

**Interpretation**: How much better is action $a$ compared to average action in state $s$.

**Properties**:
- $A(s,a) > 0$: Action better than average
- $A(s,a) < 0$: Action worse than average
- $A(s,a) = 0$: Action is average

**Why important**: Reduces variance in policy gradients.

**Estimation**:
- **TD error**: $\delta = r + \gamma V(s') - V(s)$ (biased estimator)
- **GAE**: Generalized Advantage Estimation (better)

---

## Part 8: Actor-Critic Methods

### What's an actor-critic method?

**Actor-critic**: Combines policy gradient (actor) with value function (critic).

**Two components**:
1. **Actor** $\pi_\theta(a|s)$: Policy (chooses actions)
2. **Critic** $V_w(s)$ or $Q_w(s,a)$: Value function (evaluates actions)

**Update**:
- **Critic**: Update value estimate (e.g., TD learning)
- **Actor**: Update policy using critic's evaluation

**Advantages**:
- Lower variance than REINFORCE (bootstrapping)
- More sample efficient

**Examples**: A2C, A3C, PPO, SAC.

### Write the basic actor-critic update

**Critic update** (TD):
$$w \leftarrow w + \alpha_w \delta \nabla_w V_w(s)$$

where TD error: $\delta = r + \gamma V_w(s') - V_w(s)$

**Actor update** (policy gradient):
$$\theta \leftarrow \theta + \alpha_\theta \delta \nabla_\theta \log \pi_\theta(a|s)$$

**Key**: Use TD error $\delta$ as advantage estimate.

**Approximation**: $\delta \approx A(s,a)$ (biased but low variance).

### What's A2C (Advantage Actor-Critic)?

**A2C** (Synchronous Advantage Actor-Critic):

**Architecture**:
- Shared network: $s \to$ features
- Actor head: features → $\pi(a|s)$
- Critic head: features → $V(s)$

**Loss**:
$$\mathcal{L} = \mathcal{L}_{actor} + c_1 \mathcal{L}_{critic} - c_2 H(\pi)$$

Where:
- $\mathcal{L}_{actor} = -\mathbb{E}[A \log \pi(a|s)]$ (policy gradient)
- $\mathcal{L}_{critic} = \mathbb{E}[(G - V(s))^2]$ (value loss)
- $H(\pi)$ = entropy bonus (encourages exploration)

**Synchronous**: Multiple workers collect data, update synchronously.

### What's A3C?

**A3C** (Asynchronous Advantage Actor-Critic):

**Key innovation**: **Asynchronous** parallel training.

**Architecture**:
- Multiple workers (agents) in parallel environments
- Each worker: own copy of network
- Shared global network

**Training**:
1. Workers collect experience asynchronously
2. Compute gradients locally
3. Update global network (asynchronously)
4. Synchronize local network with global

**Advantages**:
- **Parallelism**: Speed up training
- **Diversity**: Different workers explore different policies (decorrelation)
- No experience replay needed

**Limitation**: A2C often better (synchronous simpler, GPUs prefer batches).

### What's GAE (Generalized Advantage Estimation)?

**GAE**: Method to estimate advantage with bias-variance trade-off.

**Formula**:
$$A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ (TD error).

**Special cases**:
- $\lambda = 0$: $A = \delta$ (high bias, low variance)
- $\lambda = 1$: $A = G - V(s)$ (low bias, high variance)

**Typical**: $\lambda \in [0.9, 0.99]$ (good balance).

**Why**: Smooths advantage estimates, reduces variance.

**Used in**: PPO, TRPO, modern actor-critic.

### What's PPO (Proximal Policy Optimization)?

**PPO**: State-of-the-art policy gradient method.

**Key idea**: Constrain policy updates to avoid destructively large updates.

**Clipped objective**:
$$L^{CLIP}(\theta) = \mathbb{E}\left[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)\right]$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$ (probability ratio)
- $\epsilon = 0.2$ typically

**Interpretation**: Clip ratio to $[1-\epsilon, 1+\epsilon]$ (limit how much policy can change).

**Advantages**:
- Simple implementation
- Robust (works across many tasks)
- Sample efficient

**Used everywhere**: Default RL algorithm for many applications.

### What's TRPO?

**TRPO** (Trust Region Policy Optimization): Predecessor to PPO.

**Key idea**: Constrain policy update using KL divergence.

**Constraint**:
$$\max_\theta \mathbb{E}[A_{\theta_{old}}]$$
$$\text{s.t. } \mathbb{E}[KL(\pi_{\theta_{old}} \| \pi_\theta)] \leq \delta$$

**Implementation**: Second-order optimization (conjugate gradient).

**Advantages**:
- Strong theoretical guarantees (monotonic improvement)
- More stable than vanilla PG

**Disadvantages**:
- Complex implementation (Fisher information matrix)
- Computationally expensive

**PPO simplifies TRPO**: Clips instead of KL constraint (easier, similar performance).

---

## Part 9: Deep Q-Networks (DQN)

### What's DQN?

**DQN** (Deep Q-Network): Q-learning with deep neural networks.

**Key innovations** (2015, Atari):
1. **Experience replay**: Break correlation
2. **Target network**: Stabilize training
3. **CNN**: Process raw pixels

**Architecture**: $s \to$ CNN → FC → $Q(s,a)$ for each action

**Loss**:
$$L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2]$$

**Breakthrough**: First to achieve human-level performance on Atari from pixels.

### What problem does Double DQN solve?

**Problem**: DQN overestimates Q-values.

**Why**: $\max_{a'} Q(s',a')$ uses same network for selection and evaluation
- Maximization bias: max of noisy estimates is biased upward

**Double DQN fix**: Decouple selection and evaluation.

$$Y^{DoubleDQN} = r + \gamma Q(s', \arg\max_{a'} Q(s',a';\theta); \theta^-)$$

- **Select** action with online network $\theta$
- **Evaluate** action with target network $\theta^-$

**Result**: More accurate Q-values, better performance.

### What problem does Dueling DQN solve?

**Dueling DQN**: Separate value and advantage streams.

**Architecture**:
$$Q(s,a) = V(s) + \left(A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')\right)$$

**Two streams**:
1. **Value stream**: $V(s)$ (how good is state)
2. **Advantage stream**: $A(s,a)$ (how much better is action vs average)

**Why**: Many states don't need to distinguish action values (value enough).

**Example**: Empty room in Atari - all actions equally bad/good.

**Advantage**: Learn value even when actions don't matter (better generalization).

### What's prioritized experience replay?

**Standard replay**: Sample uniformly from buffer.

**Prioritized replay**: Sample important transitions more frequently.

**Priority**: $p_i = |\delta_i| + \epsilon$ where $\delta$ is TD error.

**Sampling probability**:
$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

**Importance sampling weights** (correct bias):
$$w_i = \left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^\beta$$

**Advantages**:
- Focus on surprising transitions (high TD error)
- Faster learning

**Used in**: Rainbow DQN, Ape-X.

### What's Rainbow DQN?

**Rainbow**: Combines 6 DQN improvements.

**Components**:
1. **Double DQN**: Reduce overestimation
2. **Dueling**: Separate V and A
3. **Prioritized replay**: Sample important transitions
4. **Multi-step**: n-step returns
5. **Distributional**: Model full return distribution (not just mean)
6. **Noisy nets**: Learned exploration

**Result**: SOTA on Atari at the time (2017).

**Insight**: Orthogonal improvements combine well.

---

## Part 10: Continuous Control

### Why can't DQN handle continuous actions?

**Problem**: DQN requires $\max_a Q(s,a)$.

**Discrete actions**: Iterate over all actions (feasible if small set).

**Continuous actions**: Infinite actions, can't enumerate.
- Example: Robot joint angles ∈ ℝⁿ

**Solutions**:
- Discretize (loses precision, curse of dimensionality)
- Policy gradient (directly output action)
- Actor-critic for continuous (DDPG, TD3, SAC)

### What's DDPG?

**DDPG** (Deep Deterministic Policy Gradient): Actor-critic for continuous control.

**Components**:
1. **Actor** $\mu_\theta(s)$: Deterministic policy
2. **Critic** $Q_w(s,a)$: Q-function

**Update**:
- **Critic**: Minimize $(r + \gamma Q(s',\mu(s');\theta^-) - Q(s,a))^2$
- **Actor**: Maximize $Q(s,\mu(s))$ via chain rule:
  $$\nabla_\theta J = \mathbb{E}[\nabla_a Q(s,a)|_{a=\mu(s)} \nabla_\theta \mu(s)]$$

**Key features**:
- Experience replay
- Target networks (both actor and critic)
- Ornstein-Uhlenbeck noise for exploration

**Limitation**: Sensitive to hyperparameters, can be unstable.

### What's TD3 (Twin Delayed DDPG)?

**TD3**: Improved DDPG addressing overestimation and instability.

**Three tricks**:

1. **Clipped Double Q-learning**: Use two critics, take minimum
   $$y = r + \gamma \min_{i=1,2} Q_{\theta_i'}(s', \pi_\theta(s'))$$

2. **Delayed policy updates**: Update actor less frequently than critic
   - Update critic every step
   - Update actor every $d$ steps (e.g., $d=2$)

3. **Target policy smoothing**: Add noise to target actions
   $$y = r + \gamma Q(s', \pi(s') + \epsilon), \epsilon \sim N(0, \sigma)$$

**Result**: Much more stable than DDPG, better performance.

### What's SAC (Soft Actor-Critic)?

**SAC**: Maximum entropy RL framework.

**Key idea**: Maximize reward + entropy
$$J(\pi) = \mathbb{E}\left[\sum_t r_t + \alpha H(\pi(\cdot|s_t))\right]$$

**Components**:
1. **Actor**: Stochastic policy $\pi_\theta(a|s)$ (Gaussian)
2. **Critic**: Two Q-functions $Q_w(s,a)$ (Double Q)
3. **Temperature** $\alpha$: Controls exploration (auto-tuned)

**Advantages**:
- **Stable**: Entropy encourages exploration, prevents collapse
- **Sample efficient**: Off-policy + replay buffer
- **Robust**: Works across many tasks without tuning

**Most popular** continuous control algorithm currently.

### DDPG vs TD3 vs SAC - which to use?

**DDPG**:
- ❌ Unstable, sensitive
- Use: Research baseline only

**TD3**:
- ✅ Stable, good performance
- ✅ Deterministic policy (some applications prefer)
- Use: When SAC doesn't work

**SAC**:
- ✅ Most stable
- ✅ Best sample efficiency
- ✅ Robust across tasks
- Use: **Default choice** for continuous control

**Trend**: SAC is current best practice.

---

## Part 11: Advanced Topics

### What's model-based RL?

**Model-based RL**: Learn model of environment, use for planning.

**Model**: Learn $\hat{P}(s'|s,a)$ and/or $\hat{R}(s,a,s')$

**Use model for**:
1. **Planning**: Simulate rollouts, improve policy
2. **Data augmentation**: Generate synthetic experience
3. **Prediction**: Predict future states

**Advantages**:
- **Sample efficiency**: Learn from fewer real interactions
- **Interpretability**: Understand environment dynamics

**Disadvantages**:
- **Model errors**: Wrong model leads to poor policy
- **Complexity**: Need to learn model + policy

**Examples**: Dyna-Q, MuZero, World Models.

### What's the Dyna architecture?

**Dyna**: Combine model-free and model-based RL.

**Two learning processes**:
1. **Direct RL**: Learn from real experience (Q-learning)
2. **Planning**: Learn from simulated experience (using model)

**Algorithm**:
```
Repeat:
    (a) Take action in environment, observe s, a, r, s'
    (b) Direct RL: Q(s,a) ← Q(s,a) + α[r + γmax_a'Q(s',a') - Q(s,a)]
    (c) Model learning: Model(s,a) ← s', r
    (d) Planning (n times):
         - Sample s, a from experience
         - Simulate s', r from Model(s,a)
         - Q(s,a) ← Q(s,a) + α[r + γmax_a'Q(s',a') - Q(s,a)]
```

**Benefit**: Sample efficiency (reuse experience via simulation).

### What's imitation learning?

**Imitation learning**: Learn policy from expert demonstrations.

**Two main approaches**:

1. **Behavioral cloning** (BC):
   - Supervised learning: $(s,a)$ pairs from expert
   - Train policy: $\pi_\theta(a|s)$ to match expert actions
   - Simple but distribution mismatch problem

2. **Inverse RL** (IRL):
   - Infer reward function from expert behavior
   - Use recovered reward to train policy
   - More robust but computationally expensive

**Use cases**:
- Bootstrap RL training (warm start)
- Learn from human demonstrations
- Leverage offline data

### What's inverse reinforcement learning?

**Inverse RL**: Infer reward function from expert policy.

**Problem**: Given expert trajectories $\tau^*$, find reward $R$ such that expert is optimal.

**Why**: Easier to demonstrate than specify reward.

**Example**: Self-driving
- Hard to define reward function
- Easy to demonstrate good driving

**Methods**:
- **MaxEnt IRL**: Maximum entropy principle
- **GAIL**: Adversarial imitation (like GAN)

**Applications**: Robotics, autonomous driving.

### What's multi-agent RL (MARL)?

**MARL**: Multiple agents learning simultaneously in shared environment.

**Challenges**:
1. **Non-stationarity**: Other agents' policies change (environment is non-stationary)
2. **Credit assignment**: Which agent caused reward?
3. **Scaling**: Exponential joint action space
4. **Coordination**: Agents must learn to cooperate/compete

**Settings**:
- **Cooperative**: Shared reward (team)
- **Competitive**: Zero-sum (adversarial)
- **Mixed**: General-sum

**Algorithms**: Independent Q-learning, QMIX, MADDPG, AlphaZero (adversarial).

### What's reward shaping?

**Reward shaping**: Modify reward function to speed up learning.

**Example** (robot reaching goal):
- **Sparse**: $r = +1$ at goal, $0$ elsewhere (hard to learn)
- **Dense**: $r = -\text{distance to goal}$ (easier to learn)

**Potential-based shaping** (preserves optimal policy):
$$F(s,s') = \gamma \Phi(s') - \Phi(s)$$

where $\Phi$ is potential function.

**Trade-offs**:
- ✅ Faster learning
- ❌ Can introduce unintended behaviors (reward hacking)

**Best practice**: Design carefully, avoid over-shaping.

### What's hierarchical RL?

**Hierarchical RL**: Decompose task into hierarchy of subtasks.

**Motivation**: Complex tasks easier to solve with temporal abstraction.

**Options framework**:
- **Option**: Tuple $(I, \pi, \beta)$
  - $I$: Initiation set (where option can start)
  - $\pi$: Policy (what to do)
  - $\beta$: Termination condition (when to stop)

**Example** (navigate building):
- High-level: "Go to room A"
- Low-level: "Open door", "Walk forward"

**Benefits**:
- Faster learning (reuse skills)
- Better exploration
- Interpretability

**Methods**: Options, Feudal RL, HAM.

---

## Part 12: Practical Considerations

### How to debug RL algorithms?

**Common issues**:

1. **Not learning**:
   - Check reward scale (too small/large)
   - Verify exploration (enough randomness?)
   - Check learning rate (too low/high)

2. **Unstable**:
   - Reduce learning rate
   - Add gradient clipping
   - Use target networks (if value-based)

3. **Poor performance**:
   - Visualize trajectories (what's agent doing?)
   - Check advantage distribution (too positive/negative?)
   - Tune hyperparameters

**Debugging checklist**:
- ✓ Environment works correctly?
- ✓ Reward signal makes sense?
- ✓ Agent can reach positive reward?
- ✓ Observation normalized?
- ✓ Policy explores enough?

### What hyperparameters matter most?

**Critical** (tune these first):
1. **Learning rate**: Most important (try 1e-3, 1e-4)
2. **Network architecture**: Size and depth
3. **Discount factor** $\gamma$: Typical 0.99
4. **Exploration** ($\epsilon$, entropy): Enough exploration?

**Secondary**:
5. **Batch size**: 32-256 typical
6. **Replay buffer size**: 1e6 for DQN
7. **Target update frequency**: 1000-10000 steps
8. **Entropy coefficient** (for PPO/SAC)

**Start with**: Published hyperparameters for similar task.

### How to choose RL algorithm for a task?

**Decision tree**:

**1. Action space?**
- Discrete → DQN, PPO
- Continuous → PPO, SAC, TD3

**2. On-policy or off-policy?**
- Need sample efficiency → Off-policy (DQN, SAC)
- Have simulator, parallelize → On-policy (PPO, A2C)

**3. Episodic or continuing?**
- Episodic → Any algorithm
- Continuing → TD methods

**4. Model available?**
- Yes → Model-based (MuZero, Dyna)
- No → Model-free

**Default recommendations**:
- **Discrete**: PPO or DQN
- **Continuous**: SAC or PPO
- **Multi-agent**: PPO or specialized (QMIX)

### Common RL pitfalls?

1. **Sparse rewards**: Agent never finds reward
   - Fix: Reward shaping, curriculum, exploration

2. **Unstable training**: Loss explodes, policy collapse
   - Fix: Lower LR, grad clipping, smaller networks

3. **Local optima**: Agent finds suboptimal solution
   - Fix: More exploration, reset exploration schedule

4. **Overfitting**: Works in train, fails in test
   - Fix: Domain randomization, regularization

5. **Reward hacking**: Agent exploits reward function
   - Fix: Better reward design, add constraints

6. **Catastrophic forgetting**: Forgets old skills
   - Fix: Replay buffer, regularization

**Best practice**: Start simple (small network, simple environment), scale gradually.

### How to evaluate RL agents?

**Metrics**:
1. **Mean episode return**: Average total reward
2. **Learning curve**: Return vs environment steps
3. **Sample efficiency**: Steps to reach threshold
4. **Wall-clock time**: Real time to train

**Evaluation protocol**:
- Train on multiple seeds (5-10)
- Report mean ± std
- Use separate evaluation episodes (no exploration)
- Visualize trajectories

**Benchmarks**:
- Atari: 57 games
- MuJoCo: Continuous control
- DM Control: Physics simulation

**Important**: RL has high variance - always use multiple seeds!

# Continual Reinforcement Learning

---

## Foundations

### What is Continual Reinforcement Learning (CRL)?

**CRL**: Learning from a stream of tasks or non-stationary environments over an open-ended lifetime, acquiring new competencies while retaining and refining old ones.

**Setting**:
- Agent faces tasks sequentially: $\mathcal{T} = \{T_1, T_2, \ldots, T_N\}$
- Each task $T_k = (\mathcal{S}_k, \mathcal{A}_k, P_k, R_k, \gamma_k)$ is an MDP
- Limited or no access to previous task data
- Must maintain performance on all learned tasks
- Should leverage past knowledge for faster learning on new tasks

**Example**: Robot learns warehouse navigation → hospital navigation → outdoor delivery, must remember all three and ideally each new environment makes it better at the others.

**Why it matters**: Real-world agents face changing environments. A warehouse robot gets redeployed. A game agent encounters new opponents. An LLM must align to shifting preferences. Standard RL assumes a fixed MDP and breaks in all these cases.

### How is CRL formally different from related settings?

This is a critical distinction that interviewers test. Abel et al. (2024) provide the cleanest formalization:

| Setting | Key Difference from CRL |
|---------|------------------------|
| **Multi-task RL** | Simultaneous access to all tasks; no sequential constraint. MTL is the upper bound for CRL. |
| **Meta-RL** | Optimizes for fast adaptation from a known task distribution; assumes i.i.d. tasks. |
| **Transfer RL** | One-shot: source → target. Don't care about source performance after transfer. |
| **Curriculum learning** | Task ordering is controlled; goal is single-task performance, not multi-task retention. |
| **Non-stationary RL** | Environment changes within a single task (e.g., drifting dynamics), not across discrete tasks. |

**The sharp version**: CRL is distinguished by requiring (1) sequential task presentation, (2) retention of performance on all tasks, and (3) bounded resources. If you drop any one of these, you get a different problem.

### What scenario settings exist in CRL?

Borrowed from supervised CL and adapted:

| Scenario | What Changes | Task Identity at Test |
|----------|-------------|----------------------|
| **Task-incremental** | Dynamics, rewards, goals | Known (agent told which task) |
| **Domain-incremental** | Observation distribution (e.g., visual appearance) | Unknown |
| **Reward-incremental** | Reward function only | Unknown |

**Most benchmarks are task-incremental with known boundaries.** The harder, more realistic setting is **boundary-free CRL** where the agent must detect distributional shifts itself. This is an open problem.

**Follow-up you should expect**: "What if there are no task boundaries at all?" Answer: This is the hardest setting — continuous non-stationary RL. Methods need online change detection (e.g., monitoring prediction error spikes, value function divergence, or policy performance drops) to implicitly segment the stream.

---

## The Core Challenge

### What's the main challenge in CRL?

**Triangular challenge** (unique to CRL — supervised CL only has the first two):

**1. Stability**: Retain old knowledge. Prevent catastrophic forgetting. Maintain performance on tasks 1–5 while learning task 6.

**2. Plasticity**: Adapt to new knowledge. Forward transfer (old helps new), backward transfer (new improves old), and raw ability to fit new tasks.

**3. Scalability**: Bounded resources. Memory constraints, compute constraints. RL needs orders of magnitude more samples than supervised learning — each task may require millions of interactions.

**The trilemma in practice**:

| Prioritize | Sacrifice | Mechanism |
|-----------|-----------|-----------|
| Stability | Plasticity | Freeze weights, heavy regularization → can't learn new tasks |
| Plasticity | Stability | Free updates → catastrophic forgetting |
| Scalability | Both | Limit memory/compute → insufficient capacity |
| Stability + Plasticity | Scalability | Store everything, grow architecture → unbounded resources |

**Every CRL method is a different point in this triangle.** A good interview answer identifies where a specific method sits.

### What is catastrophic forgetting in RL?

**Definition**: Significant performance decline on previously learned tasks when learning new ones.

$$CF_i = \max(p_{i,i} - p_{N,i}, 0)$$
- $p_{i,i}$: Performance on task $i$ right after training on it
- $p_{N,i}$: Performance on task $i$ after training on all $N$ tasks

**Why it happens in neural networks**: Gradient descent on new task loss overwrites weight configurations that encoded old task knowledge. There's no built-in mechanism to identify or protect "important" weights.

**Why it's worse in RL than supervised learning**:
- RL data is non-i.i.d. (trajectories are correlated)
- Reward signals are stochastic, delayed, and sparse
- Policy changes affect data distribution (on-policy methods)
- Value functions are bootstrapped — small weight changes cascade

**Mitigation families**: Regularization (constrain weight changes), replay (mix old and new data), architectural (isolate parameters per task), distillation (compress old knowledge).

### What is plasticity loss and how is it different from forgetting?

**This is arguably the most important recent finding and a strong interview differentiator.**

**Plasticity loss**: The network progressively loses the ability to learn *anything new*, even with unlimited data from a fresh task. This is NOT about forgetting old tasks — it's about the network becoming unable to fit new ones.

**Key distinction**:
- **Catastrophic forgetting**: Old task performance drops when learning new tasks
- **Plasticity loss**: New task learning becomes slower/impossible regardless of forgetting

**Causes** (Lyle et al., 2023, 2024; Dohare et al., 2024):

1. **Dormant/dead neurons**: ReLU units saturate at zero and never recover. After many tasks, a large fraction of the network is effectively dead capacity.

2. **Feature rank collapse**: Learned representations become increasingly low-rank. Effective dimensionality shrinks, reducing capacity to represent new functions.

3. **Weight magnitude growth**: Weights drift to large magnitudes, pushing activations into saturated regimes where gradients vanish.

4. **Loss of gradient signal**: Combination of the above means gradients become small or uninformative.

**Solutions**:

| Method | Mechanism | Reference |
|--------|-----------|-----------|
| **Continual Backpropagation** | Reinitialize dormant neurons based on utility | Dohare et al. (2024, Nature) |
| **Neuron Recycling** | Detect and reset dead units | Abbas et al. (2023) |
| **Layer Normalization** | Prevents representation collapse | Lyle et al. (2023) |
| **Periodic Resets** | Reset network, keep replay buffer | Nikishin et al. (2022) |
| **Shrink & Perturb** | $\theta \leftarrow \alpha\theta + (1-\alpha)\theta_{\text{init}}$ | Ash & Adams (2020) |
| **CReLU** | Preserves gradient flow through both signs | Lyle et al. (2024) |
| **Weight Decay** | Simple but effective at preventing magnitude growth | Standard |

**Why this matters for CRL**: If you solve forgetting perfectly but your network can't learn task 50, you've solved the wrong problem. Any serious CRL system must address both. And they can conflict: methods that protect old weights (EWC, frozen columns) may accelerate plasticity loss by reducing effective free capacity.

**Interview tip**: If asked "what's the biggest challenge in CRL?", mentioning plasticity loss as distinct from forgetting signals depth.

---

## The Reset and Reinitialization Literature

### Why should I care about periodic resets?

**Because they set embarrassingly strong baselines that complex methods often can't beat.**

**Primacy bias** (Nikishin et al., 2022): Networks trained on early data become biased toward early features and can't effectively learn from later data. Simply resetting the network periodically (keeping the replay buffer) restores learning ability.

**Key methods**:

| Method | Mechanism | Reference |
|--------|-----------|-----------|
| **Periodic full reset** | Reset all weights every $K$ steps, keep buffer | Nikishin et al. (2022) |
| **Shrink & Perturb** | Interpolate toward initialization: $\theta \leftarrow \alpha\theta + (1-\alpha)\theta_0$ | Ash & Adams (2020) |
| **Plasticity injection** | Selectively reinitialize parts that lost plasticity | Lyle et al. (2024) |
| **UPGD** | Perturb low-utility parameters only | Elsayed & Mahmood (2024) |

**Why this matters**: Any new CRL method should be benchmarked against "periodic reset + replay buffer." It's simple, cheap, and often wins. If your complex method can't beat this baseline, it's not contributing.

**Interview angle**: If asked about baselines for CRL experiments, mentioning this shows you understand what actually matters empirically vs. what's methodologically exciting.

---

## Taxonomy of CRL Methods

### What are the 4 types of CRL methods?

**Taxonomy based on what knowledge is stored/transferred** (Pan et al., 2025):

**1. Policy-Focused**: Store/transfer policies $\pi(a|s)$ or value functions $Q(s,a)$
- Sub-types: Policy reuse, decomposition, merging (regularization/distillation)
- Examples: EWC, PNN, CSP, P&C

**2. Experience-Focused**: Store/transfer past transitions $(s, a, r, s')$
- Sub-types: Direct replay, generative replay
- Examples: CLEAR, RePR, 3RL

**3. Dynamic-Focused**: Store/transfer environment dynamics $P(s'|s,a)$
- Sub-types: Direct modeling, latent modeling
- Examples: HyperCRL, LILAC, Continual-Dreamer

**4. Reward-Focused**: Store/transfer reward shaping or intrinsic motivation
- Sub-types: Potential-based shaping, curiosity bonuses
- Examples: SR-LLRL, IML, MT-Core

**But this taxonomy misses several major families** (good to mention in an interview):
- Successor features / GPI (transfer via value decomposition)
- Meta-RL approaches (learn to learn)
- Representation-based approaches (frozen pre-trained backbones)
- Reset/reinitialization methods (maintain plasticity)

### How does EWC work?

**EWC (Elastic Weight Consolidation)**: Regularization protecting important weights.

**Loss function**:
$$\mathcal{L}(\theta) = \mathcal{L}_{\text{new}}(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta^*_{\text{old},i})^2$$

- $F_i$: Fisher Information — measures how much parameter $i$ matters for old tasks
- $\theta^*_{\text{old}}$: Optimal parameters after old tasks
- $\lambda$: Regularization strength (stability-plasticity knob)

**Fisher Information**:
$$F_i = \mathbb{E}\left[\left(\frac{\partial \log \pi(a|s;\theta)}{\partial \theta_i}\right)^2\right]$$

**Interpretation**: High $F_i$ → parameter $i$ is critical for old tasks → large penalty for changing it. Low $F_i$ → free to adapt.

**Pros**: No memory for data, parameter-efficient, constant model size.

**Cons**: 
- Accumulates quadratic penalties → increasingly constrains the network → plasticity loss over many tasks
- Fisher is approximate (diagonal, computed at single point)
- No forward transfer mechanism
- Assumes tasks don't require overlapping weight regions

**Variants**: Online EWC (running Fisher average), SI (Synaptic Intelligence — online importance), MAS (Memory Aware Synapses — sensitivity-based importance).

**Interview follow-up**: "What happens after 100 tasks with EWC?" The Fisher penalties accumulate, effectively freezing most weights. This is exactly the plasticity loss problem — EWC trades plasticity for stability and eventually can't learn at all. This motivates methods like Progress & Compress that periodically consolidate knowledge.

### How does Progressive Neural Networks (PNN) work?

**Architecture**: Add new column per task, freeze old columns, add lateral connections.

```
Task 1:     [Col 1]
Task 2:     [Col 1] ← [Col 2]
Task 3:     [Col 1] ← [Col 2] ← [Col 3]
            (frozen)   (frozen)   (trainable)
```

**Lateral connections**:
$$h_i^{(k)} = f\left(W^{(k)}_i h^{(k)}_{i-1} + \sum_{j<k} U^{(k:j)}_i h^{(j)}_{i-1}\right)$$

**Pros**:
- **Zero forgetting**: Old columns never modified (guaranteed)
- **Forward transfer**: Via lateral connections
- **Simple**: No complex regularization

**Cons**:
- **Linear parameter growth**: $N$ tasks = $N$ columns
- **Linear inference cost**: Must evaluate all columns
- **No backward transfer**: Old columns frozen, can't improve
- **No capacity sharing**: Each task gets full column even if it's similar to past tasks

**When to use**: Memory unconstrained, need guaranteed zero forgetting, few tasks.

**Interview contrast**: Compare PNN to PackNet (prune-then-train — share capacity) and Supermask (binary masks over shared weights). PNN wastes capacity; modular methods are more efficient but risk interference.

### How does policy decomposition work?

**Core idea**: $\theta_k = L \cdot s_k$ — separate shared structure from task-specific adaptation.

**Four approaches**:

**1. Factor Decomposition**: $L \in \mathbb{R}^{d \times m}$ is shared latent basis, $s_k \in \mathbb{R}^m$ is task-specific. PG-ELLA, LPG-FTW. Scales well (only $m$ parameters per new task) but requires related tasks.

**2. Multi-head**: Shared trunk + task heads. Trunk learns common features, heads specialize. Simple, widely used, but requires knowing task identity at inference.

**3. Hierarchical**: Options/skills at low level (reusable), meta-policy at high level (task-specific). HLifeRL. Good when tasks share subtask structure.

**4. Modular/MoE**: Multiple expert modules + gating. Routes different inputs to different experts. Good capacity efficiency but gating is hard to learn.

**Key insight for interviews**: Factor decomposition ($\theta_k = L \cdot s_k$) is structurally related to conditioning in NeuPL (mixture-conditioned policy parameterization). Both learn a shared representation that is modulated by a task/mixture descriptor.

### How does CLEAR work?

**CLEAR** (Continual Learning with Experience And Replay): Dual-buffer system inspired by complementary learning systems in the brain.

**Architecture**:
```
Short-term buffer (hippocampus)    Long-term buffer (neocortex)
Current task experiences      ←→   Selected experiences from all tasks
```

**Training**:
- Sample from both buffers
- **Behavior cloning loss** on replayed data: $\mathcal{L}_{BC} = -\log \pi(a|s)$
- **Policy gradient loss** on current data: $\mathcal{L}_{PG}$
- Combined: $\mathcal{L} = \mathcal{L}_{PG} + \lambda \mathcal{L}_{BC}$

**Key innovation**: Behavior cloning (supervised imitation) on replayed data rather than re-running RL on it. This is more stable because BC is a supervised objective.

**V-Trace correction**: Off-policy correction for replayed data that may come from a different policy version.

**Pros**: Effective, stable, works with on-policy algorithms.
**Cons**: Memory overhead (buffer), $\lambda$ hyperparameter, stored experiences go off-policy.

**Interview question**: "Why behavior cloning instead of just replaying experiences through the RL loss?" Because the RL loss (e.g., policy gradient) is high-variance and off-policy correction is imperfect. BC is a stable supervised signal that directly imitates the known-good behavior.

---

## Successor Features and Transfer

### What are Successor Features and why do they matter for CRL?

**This is a major methodological family that many CRL surveys miss. Knowing this well signals depth.**

**Core decomposition** (Barreto et al., 2017, 2018):

Assume reward is linear in features: $r(s,a,s') = \phi(s,a,s')^\top \mathbf{w}$

Then the Q-function decomposes as:
$$Q^\pi(s,a) = \psi^\pi(s,a)^\top \mathbf{w}$$

Where:
- $\psi^\pi(s,a) = \mathbb{E}^\pi\left[\sum_{t=0}^\infty \gamma^t \phi(s_t, a_t, s_{t+1}) \mid s_0=s, a_0=a\right]$ — **successor features** (depend on dynamics + policy, NOT reward)
- $\mathbf{w}$ — **reward weights** (depend only on the reward function)

**For CRL**: If tasks share dynamics but differ in reward (very common!), learn $\psi$ once and only update $\mathbf{w}$ per task. This gives **zero-shot transfer** to new reward functions.

### What is Generalized Policy Improvement (GPI)?

Given policies $\pi_1, \ldots, \pi_k$ from previous tasks with known successor features, the GPI policy for new task with weights $\mathbf{w}_{k+1}$:

$$\pi_{\text{GPI}}(s) = \arg\max_a \max_i \psi^{\pi_i}(s,a)^\top \mathbf{w}_{k+1}$$

**Intuition**: For each state-action, evaluate which previous policy would be best under the new reward, and use that one.

**Properties**:
- Zero-shot (no training needed for new task)
- Performance monotonically improves as policies are added
- Comes with formal guarantees

**Limitation**: Assumes reward linearity in known features $\phi$. This is restrictive but covers many practical cases (navigation to different goals, manipulation with different objectives).

### What are Universal Value Function Approximators (UVFA)?

Schaul et al. (2015): $Q(s, a, g)$ — value function conditioned on goal/task descriptor $g$.

**For CRL**: The task descriptor $g$ parameterizes non-stationarity. A single network handles all tasks if conditioned on what task it's doing. This is closely related to:
- Goal-conditioned RL
- Hindsight Experience Replay (HER)
- Context-conditioned policies in meta-RL

**Connection to other methods**: UVFA is a form of policy decomposition where the "task-specific" part is a conditioning input rather than separate parameters. Similar in spirit to FiLM conditioning or mixture-conditioned policies.

---

## Meta-RL as Continual Learning

### How does meta-RL relate to CRL?

**Instead of protecting old knowledge, learn to learn quickly.** If the agent adapts to any new task in a few episodes, forgetting matters less.

**Key approaches**:

| Method | Mechanism |
|--------|-----------|
| **MAML** | Find initialization that's a few gradient steps from any task optimum |
| **RL²** | RNN hidden state becomes the "learning algorithm" — learns in-context |
| **PEARL** | Probabilistic context inference for task identification |
| **AdA** | Transformer-scale open-ended adaptation (DeepMind, 2023) |
| **Algorithm Distillation** | Distill RL learning histories into a transformer (Laskin et al., 2022) |

### What is AdA and why does it matter?

**Adaptive Agent (AdA)** (DeepMind, 2023): Large transformer trained on a massive distribution of procedurally-generated tasks in XLand.

**Key properties**:
- Adapts to new tasks on human timescales (minutes, not millions of steps)
- No explicit CRL mechanism — adaptation is emergent from scale + architecture
- Suggests sufficient scale + diverse training may sidestep CRL entirely

**Implication**: Maybe the future of lifelong learning isn't clever forgetting-prevention algorithms but simply building models large and diverse enough that adaptation is trivial. This is the "foundation model" hypothesis for CRL.

### What is Algorithm Distillation?

Laskin et al. (2022): Train a transformer on *entire RL learning histories* — sequences of (observation, action, reward) across the learning process of many tasks. The transformer learns to replicate the learning algorithm itself.

**For CRL**: The transformer can "learn in-context" — given a few episodes of a new task, it improves without any weight updates. This is a form of meta-RL that could enable continual adaptation without forgetting (no weights change).

### Meta-RL vs CRL: when to use which?

| Aspect | Meta-RL | CRL |
|--------|---------|-----|
| **Assumption** | Tasks from known distribution | Arbitrary task sequence |
| **Goal** | Fast adaptation | No forgetting + transfer |
| **Failure mode** | Out-of-distribution tasks | Forgetting + plasticity loss |
| **Scalability** | Fixed-size model | Often growing buffers/architectures |
| **When to use** | Known task distribution, fast adaptation needed | Open-ended deployment, all tasks matter |

**The emerging consensus**: Hybrid approaches combining meta-learning with CRL mechanisms are likely the future. Algorithm Distillation is an early example.

---

## Representation Learning and Foundation Models

### How do foundation models change CRL?

**The frozen backbone approach** is increasingly the dominant practical method for robotics CRL:

1. Pre-train a large visual/language encoder on diverse data
2. Freeze it
3. Train only a lightweight policy head per task

**Key models**:
- **R3M** (Nair et al., 2022): Pre-trained on Ego4D human videos
- **VIP** (Ma et al., 2023): Value-Implicit Pre-training as a universal reward/representation
- **Voltron** (Karamcheti et al., 2023): Language-conditioned visual features
- **RT-2** (Brohan et al., 2023): Vision-Language-Action model

**Why this helps CRL**: If the representation is frozen, catastrophic forgetting mostly disappears — you're only updating a small head. Plasticity loss is also mitigated because the head is small and can be reset cheaply.

**When this fails**:
- New tasks require genuinely novel features not in pre-training
- Observation modality changes
- Fine-grained motor control needs task-specific perceptual features

**Interview angle**: If asked "what's the most practical approach to CRL in robotics right now?", the answer is probably "frozen pre-trained backbone + per-task head" rather than any of the classic CRL algorithms.

---

## Model-Based Continual RL

### How does model-based CRL work?

**Core idea**: Learn environment dynamics $P(s'|s,a)$, use for planning and transfer.

**Direct modeling**: Explicit transition models per task.

| Algorithm | Mechanism |
|-----------|-----------|
| **HyperCRL** | Hypernetwork generates task-conditional dynamics parameters |
| **LLIRL** | Chinese Restaurant Process — dynamically instantiate new models for genuinely new dynamics |
| **MOLe** | Meta-learning for online dynamics adaptation |
| **VBLRL** | Variational Bayesian lifelong RL |

**Mixture model approach**:
$$P(s'|s,a) = \sum_{k=1}^K w_k(s,a) P_k(s'|s,a)$$
Gating weights $w_k$ select which dynamics model applies. New task either reuses existing model or spawns a new one.

**Indirect/latent modeling**: World models that predict in latent space.

| Algorithm | Mechanism |
|-----------|-----------|
| **LILAC** | Lifelong Latent Actor-Critic, DP-MDP framework |
| **LiSP** | Lifelong Skill Planning in latent space |
| **Continual-Dreamer** | World model + reservoir sampling for continual learning |

**Pros**: Sample efficient (plan without interactions), dynamics knowledge transfers, enables multi-step lookahead.

**Cons**: Model errors compound over long rollouts, hard to scale to high-dimensional observations, model class must be expressive enough.

**When to use**: Sample efficiency critical, tasks share dynamics structure, planning is needed.

---

## Reward-Focused Methods

### How does reward shaping work in CRL?

**Add auxiliary rewards to guide learning and transfer knowledge:**
$$R^{\text{shaped}}_t = R^{\text{env}}_t + h(s_t, a_t, s_{t+1})$$

**Potential-based shaping** (preserves optimal policy):
$$h(s, s') = \gamma \Phi(s') - \Phi(s)$$

**In CRL**: The potential function $\Phi$ learned on old tasks transfers to new tasks. If $\Phi$ captures "distance to goal" structure, it transfers to new goals immediately.

**Algorithms**: SR-LLRL (visit-count shaping), ELIRL (shared latent reward structure).

### What's intrinsic motivation in CRL?

**Agent generates internal curiosity rewards:**
$$R^{\text{total}}_t = R^{\text{env}}_t + \alpha \cdot R^{\text{intrinsic}}_t$$

**Types**:

| Type | Formula/Mechanism | When Useful |
|------|-------------------|------------|
| **Prediction-based** | $R^I_t = \|\hat{s}_{t+1} - s_{t+1}\|^2$ | Dense exploration needed |
| **Count-based** | $R^I_t = 1/\sqrt{N(s_t)}$ | Tabular/discrete states |
| **Multi-timescale** (IML) | $R^I_t = \beta_1 R^{\text{short}}_t + \beta_2 R^{\text{long}}_t$ | Long-horizon exploration |
| **LLM-driven** (MT-Core) | LLM decomposes tasks into subtasks | Structured task spaces |

**In CRL**: Intrinsic motivation transfers across tasks (same curiosity mechanism), automatically adapts exploration to what's genuinely novel given past experience.

**Caveat**: Prediction-based intrinsic motivation can be fooled by stochastic environments (random noise is permanently "surprising"). Count-based methods scale poorly to continuous states.

---

## Multi-Agent Continual RL

### How does CRL interact with multi-agent settings?

**In multi-agent RL, non-stationarity is endogenous** — the environment changes because other agents adapt. This makes CRL fundamentally harder:

- The "task" isn't switching between separate MDPs — it's continuous co-adaptation
- Forgetting has strategic consequences (forgetting a counter-strategy = losing to it)
- Forward transfer means generalization to novel opponents

### How are population-based methods implicit CRL?

**PSRO** (Policy-Space Response Oracles): Iteratively computes best responses to the current meta-strategy. Each iteration adds a new policy while maintaining competence against all previous ones. This IS continual learning — just in strategy space.

**NeuPL** (Neural Population Learning): A single conditional network represents an expanding population. Mixture-conditioned policies must maintain competence across the entire strategy space as it grows — structurally analogous to task-conditioned CRL.

**The "spinning top" problem**: In adversarial settings, agents cycle through strategies without genuine improvement. Forgetting old counter-strategies causes non-transitive loops. This is CRL's catastrophic forgetting manifesting as strategic cycling.

| CRL Concept | Multi-Agent Analog |
|-------------|-------------------|
| Task sequence | Opponent adaptation sequence |
| Catastrophic forgetting | Losing ability to counter old strategies |
| Forward transfer | Generalization to novel opponents |
| Backward transfer | Improved counter-play against old opponents via new knowledge |
| Scalability | Population size |

**Interview angle**: If asked to connect CRL to your research, NeuPL's mixture-conditioning is doing the same thing as task-conditioned CRL — the mixture weights are the task descriptor.

---

## Offline-to-Online Continual RL

### What is the offline-to-online CRL paradigm?

Increasingly the practical approach:
1. **Pre-train** on large offline datasets (logs, demonstrations, simulations)
2. **Fine-tune** online in the real environment
3. **Continually adapt** as conditions change

**Key challenges**:
- **Distribution shift**: Offline data is off-policy; naive fine-tuning collapses performance
- **Conservative-to-exploratory transition**: Offline methods (CQL, IQL) are deliberately conservative; must "unlearn" conservatism for online exploration
- **Forgetting offline knowledge**: Pre-trained capabilities lost during online adaptation

**Methods**: Cal-QL (calibrated offline-to-online), RLPD (mixing offline+online in replay buffer), foundation model fine-tuning (RT-2, Octo).

---

## Curriculum and Task Ordering

### Does task order matter in CRL?

**Yes, enormously.** This is often underappreciated.

**Wolczyk et al. (2022)** showed that forward transfer, backward transfer, and forgetting all depend on task ordering. The same method looks brilliant or terrible depending on sequence order.

**Ordering strategies**:
- **Easy → hard**: Classic curriculum learning (Bengio et al., 2009)
- **Similarity-based**: Group similar tasks to maximize positive transfer
- **Diversity-based**: Alternate diverse tasks to prevent overfitting
- **Adversarial**: Worst-case ordering (hardest test of a method)

**Implication for evaluation**: Any CRL result on a single task ordering is incomplete. Robust methods need testing on multiple orderings including adversarial. This is a common methodological weakness to point out in papers.

---

## Safety and Constraints

### What about safety in CRL?

**When transferring to a new task, you can't explore freely if safety matters.** A robot can't randomly explore a hospital.

**Constrained CRL**:
- **Safe exploration**: Use uncertainty to avoid dangerous states in new environments
- **Constraint transfer**: "Don't collide with humans" should transfer from warehouse to hospital
- **Conservative initialization**: Start cautious, expand safe set gradually
- **Formal**: $\mathbb{E}[\sum_t c(s_t, a_t)] \leq d$ must hold throughout learning

**Sim-to-real**: A special CRL case (simulator → real world) where safety is critical because real exploration is expensive and dangerous.

---

## Evaluation

### How to evaluate CRL agents?

**Core metrics**:

**Average Performance**: $A_N = \frac{1}{N} \sum_{i=1}^{N} p_{N,i}$

**Average Forgetting**: $FG = \frac{1}{N-1} \sum_{i=1}^{N-1} \max(p_{i,i} - p_{N,i}, 0)$

**Forward Transfer**: $FT_i = p_{0,i}^{\text{CRL}} - p_{0,i}^{\text{random}}$ (jumpstart improvement on new task)

**Backward Transfer**: $BT = \frac{1}{N-1} \sum_{i=1}^{N-1} (p_{N,i} - p_{i,i})$ (positive = new learning improved old tasks)

**Efficiency**: Model size after $N$ tasks, sample efficiency, memory footprint, wall-clock time.

**Evaluation protocol**:
```
for each task i = 1..N:
    train on task i
    evaluate on ALL tasks 1..i
    record full performance matrix P[i,j]
```

**Required baselines**:

| Baseline | Purpose |
|----------|---------|
| **Multi-Task Learning** | Upper bound (access to all tasks) |
| **Single-Task Expert** | Per-task ceiling |
| **Fine-tuning** | Lower bound (catastrophic forgetting) |
| **Periodic Reset + Replay** | Strong simple baseline (often omitted!) |
| **Frozen Backbone + Head** | Foundation model baseline |

**What to report** (completeness checklist):
- Full performance matrix $P_{ij}$
- All three dimensions: forgetting, transfer, resources
- Multiple task orderings (random + adversarial minimum)
- ≥5 random seeds with variance
- Learning curves, not just final numbers

**Red flags in CRL papers**:
- Single seed, single task ordering
- Only final performance (no forgetting metric)
- Missing simple baselines (especially periodic reset)
- Only one benchmark

### What benchmarks exist?

| Benchmark | Domain | Tasks | Key Feature |
|-----------|--------|-------|-------------|
| **Continual World** | Meta-World robotics | 10–20 | Standardized manipulation |
| **CORA** | Atari, Procgen, MiniHack | Variable | Multi-domain, comprehensive |
| **COOM** | ViZDoom 3D | Multiple | Visual embodied perception |
| **Atari Sequences** | Atari games | Configurable | Classic, diverse dynamics |
| **LIBERO** | Robotic manipulation | Multiple | Long-horizon tasks |
| **Lifelong Hanabi** | Card game | Variable | Multi-agent CRL |

**Choosing**: Robotics → Continual World. General algorithm → CORA. Visual → COOM/Atari. Quick iteration → Minigrid.

**Report on ≥2 benchmarks from different domains.**

---

## Theory

### What theoretical results exist for CRL?

**Current state: underdeveloped relative to empirical work.**

**What we have**:
- Convergence guarantees for EWC under restrictive assumptions (quadratic loss, well-separated tasks)
- PAC-Bayes bounds for lifelong learning (Pentina & Lampert, 2014)
- Regret analysis for non-stationary bandits (simplified CRL)
- SF/GPI formal guarantees on zero-shot transfer when reward-linearity holds

**What we lack**:
- Sample complexity bounds for CRL in general
- Information-theoretic limits on stability-plasticity trade-off
- Formal characterization of when forward/backward transfer is possible
- Theory of plasticity loss (why/when it happens)
- Convergence guarantees with function approximation

**Key open theoretical questions**:
1. **Capacity limits**: How many tasks can a fixed network support? How does this scale with width/depth?
2. **Task similarity**: When can we formally guarantee positive transfer? MDP bisimulation metrics are a starting point.
3. **Optimal memory allocation**: Given fixed budget, optimal split between parameters, buffer, and task-specific modules?
4. **Impossibility results**: Are there fundamental limits on simultaneously achieving low forgetting, high transfer, and bounded compute?

---

## Comparisons

### CRL vs Multi-Task Learning?

| Aspect | MTL | CRL |
|--------|-----|-----|
| **Data access** | All tasks always | Sequential, limited past |
| **Forgetting** | No | Primary challenge |
| **Architecture** | Fixed | May grow |
| **Real-world** | Rare (need all tasks upfront) | Common (tasks emerge) |
| **Role in CRL** | Upper bound for evaluation | The actual setting |

### CRL vs Transfer Learning?

| Aspect | Transfer | CRL |
|--------|----------|-----|
| **Tasks to maintain** | Target only | All $N$ tasks |
| **Forgetting** | Acceptable | Must prevent |
| **Direction** | Source → target (one-way) | Bidirectional |
| **Evaluation** | Target performance only | All tasks + transfer + forgetting |

**Transfer is a component of CRL** — forward transfer in CRL ≈ transfer learning, but CRL also requires stability.

### CRL vs Supervised Continual Learning?

| Aspect | Supervised CL | CRL |
|--------|---------------|-----|
| **Data** | i.i.d. within task | Non-i.i.d. trajectories |
| **Feedback** | Deterministic labels | Stochastic, delayed, sparse rewards |
| **Core challenge** | Stability-Plasticity | Stability-Plasticity-Scalability |
| **Exploration** | N/A | Critical |
| **Off-policy** | N/A | Stored data goes off-policy |
| **Compute** | Moderate | Massive |

---

## Open Problems and Future Directions

### What are the biggest open challenges?

**1. Task boundary detection**: Most methods assume known switches. Real-world has gradual drift. Need online change detection from the data stream.

**2. True backward transfer**: Almost everything focuses on preventing forgetting. Actively improving old tasks from new knowledge is far harder and rarely achieved.

**3. Scaling to thousands of tasks**: Current benchmarks test 10–100. Open-ended learning needs unbounded sequences with sublinear resource growth.

**4. Plasticity at scale**: Maintaining learning ability over very long training horizons without periodic resets.

**5. Theory**: Formal trade-off analysis, convergence guarantees, sample complexity bounds.

**6. Safety-constrained CRL**: Can't freely explore during task transitions in the real world.

**7. Multi-agent CRL**: Endogenous non-stationarity from co-adapting agents.

**8. LLM continual alignment**: RLHF with shifting preferences without forgetting base capabilities.

**9. Out-of-distribution detection**: Is this genuinely new (need new capacity) or a variation (just adapt)?

**10. Hybrid methods**: No single paradigm dominates. Optimal combinations of replay + regularization + decomposition + meta-learning are unknown.

---

## Quick-Fire Interview Questions

**Q: "Name three families of CRL methods and when you'd use each."**
A: Regularization (EWC — memory limited, few tasks), replay (CLEAR — memory available, need stability), architectural (PNN — guaranteed zero forgetting, don't care about scale). Also mention SF/GPI for shared-dynamics-different-reward settings and meta-RL for known task distributions.

**Q: "What's the difference between catastrophic forgetting and plasticity loss?"**
A: Forgetting = old tasks get worse. Plasticity loss = new tasks can't be learned. They're distinct problems that can even conflict — protecting old weights (solving forgetting) can accelerate plasticity loss.

**Q: "What's the simplest competitive CRL baseline?"**
A: Periodic network reset + experience replay buffer. Embarrassingly strong, often beats EWC and other complex methods. Any new CRL paper should compare against this.

**Q: "How would you do CRL for a robot in practice today?"**
A: Frozen pre-trained visual backbone (R3M, VIP) + small policy head per task. Forgetting mostly disappears because the backbone is frozen. Head is small enough to reset cheaply. Classic CRL algorithms are backup for when the representation doesn't transfer.

**Q: "How does population-based training relate to CRL?"**
A: PSRO/NeuPL are implicit CRL — maintaining competence against an expanding strategy space without forgetting earlier counter-strategies. The mixture-conditioning in NeuPL is structurally identical to task-conditioning in CRL. The spinning top problem IS catastrophic forgetting in strategy space.

**Q: "What would you work on next in CRL?"**
A: Task-boundary detection in the wild, true backward transfer (improving old tasks from new knowledge), and hybrid meta-RL + CRL approaches. Also: formalizing the connection between population-based multi-agent methods and CRL — NeuPL-style architectures already solve a version of CRL that the single-agent literature hasn't absorbed.

---

## Key References

### Must-Read (Priority Order)
1. Khetarpal et al. (2022) — *Towards Continual RL* — best conceptual entry
2. Pan et al. (2025) — arXiv 2506.21872 — most comprehensive taxonomy
3. Lyle et al. (2023, 2024) — Plasticity loss (the most important recent finding)
4. Dohare et al. (2024, Nature) — Loss of plasticity + continual backpropagation
5. Abel et al. (2024) — Formal definition of CRL
6. Barreto et al. (2017, 2018) — Successor features and GPI
7. Nikishin et al. (2022) — Primacy bias (why resets work)

### Classic Methods
- Kirkpatrick et al. (2017) — EWC
- Rusu et al. (2016) — Progressive Neural Networks
- Rolnick et al. (2019) — CLEAR
- Schwarz et al. (2018) — Progress & Compress

### Meta-RL Connection
- Finn et al. (2017) — MAML
- Team et al. (2023) — AdA
- Laskin et al. (2022) — Algorithm Distillation

### Evaluation
- Wolczyk et al. (2021, 2022) — Continual World + disentangling transfer
- Powers et al. (2022) — CORA benchmark

### Multi-Agent
- Lanctot et al. (2017) — PSRO
- Muller et al. (2020) — NeuPL