# Reinforcement Learning - Interview Q&A

**How to use**: Questions are formatted as collapsible headings. Try to answer before expanding.

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
