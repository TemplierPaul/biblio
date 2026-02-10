# Prisoner's Dilemma

## Definition
The Prisoner's Dilemma is a classic game theory example demonstrating that rational individual behavior can lead to collectively suboptimal outcomes. It illustrates the conflict between individual and collective rationality.

## The Story
Two criminals are arrested and interrogated separately. Each has two options:
- **Cooperate** (C): Stay silent
- **Defect** (D): Betray the other

## Payoff Matrix

|          | Cooperate | Defect |
|----------|-----------|--------|
| **Cooperate** | (-1, -1) | (-3, 0) |
| **Defect**    | (0, -3)  | (-2, -2) |

**Interpretation**:
- Both cooperate: Each gets -1 year (best collective outcome)
- Both defect: Each gets -2 years (Nash equilibrium)
- One defects, other cooperates: Defector goes free (0), cooperator gets -3 years

## Game Analysis

### Nash Equilibrium
**(Defect, Defect)** is the unique Nash equilibrium

**Why?**
- If opponent cooperates, better to defect (0 > -1)
- If opponent defects, better to defect (-2 > -3)
- **Dominant strategy**: Defect regardless of opponent's choice

### Pareto Efficiency
**(Cooperate, Cooperate)** is Pareto optimal (better for both than Nash)

**Problem**: Not stable - both have incentive to deviate

### Social Dilemma
- Individual rationality → (D, D) with payoff (-2, -2)
- Collective rationality → (C, C) with payoff (-1, -1)
- **Gap**: Nash equilibrium is not socially optimal

## Variants

### Iterated Prisoner's Dilemma (IPD)
**Setup**: Play repeatedly (finite or infinite rounds)

**Changes dynamics**:
- **Cooperation emerges**: Future interactions create incentives
- **Folk Theorem**: Many outcomes sustainable as equilibria
- **Tit-for-Tat**: Start cooperate, then copy opponent (successful strategy)

**Strategies**:
- **Always Defect**: Myopic Nash
- **Tit-for-Tat**: Cooperate if opponent cooperated last round
- **Grim Trigger**: Cooperate until opponent defects, then always defect
- **Pavlov**: Win-stay, lose-shift

**Key Insight**: Reputation and repeated interaction enable cooperation

### Continuous Prisoner's Dilemma
- Actions are continuous: cooperation level $c \in [0, 1]$
- Payoffs: $u_i(c_i, c_j) = b \cdot c_j - c \cdot c_i$
  - Benefit from opponent's cooperation: $b \cdot c_j$
  - Cost of own cooperation: $c \cdot c_i$
  - Assumption: $b > c$ (cooperation is efficient)

### n-Player Prisoner's Dilemma (Public Goods Game)
- $n$ players each decide contribution $c_i \in [0, C]$
- Total public good: $G = \sum_i c_i$
- Payoff: $u_i = \alpha \cdot G - c_i$ where $\alpha < 1$ but $n \alpha > 1$
- **Dilemma**: Individual incentive to free-ride, but collective benefit from contribution

## Real-World Examples

### Environmental Issues
- **Climate change**: Reduce emissions (cooperate) vs pollute (defect)
- Individual countries incentivized to defect, but collective harm

### Arms Race
- **Disarm** (cooperate) vs **Arm** (defect)
- Both arming is expensive but stable (Nash)

### Price Competition
- **High prices** (cooperate) vs **Low prices** (defect)
- Price wars harm both firms

### Vaccination
- **Vaccinate** (cooperate) vs **Don't vaccinate** (defect)
- Free-riding on herd immunity

### Open Source Contribution
- **Contribute** (cooperate) vs **Use only** (defect)
- Tragedy of the commons

## Solutions to Dilemma

### 1. Repeated Interaction
- Build reputation
- Conditional cooperation (tit-for-tat)
- Folk theorem: enforce cooperation via punishment

### 2. Communication
- Pre-play communication (non-binding)
- Establish norms and trust
- Limited effectiveness without enforcement

### 3. Contracts / Commitment
- Binding agreements
- Third-party enforcement
- Change payoff structure

### 4. Altruism / Social Preferences
- Players care about others' payoffs
- Internalize externalities
- Changes game structure

### 5. Incomplete Information
- Uncertainty about opponent's type
- Reputation effects
- Signaling cooperation

## Multi-Agent RL Perspective

### Challenge
- Independent learning converges to mutual defection
- Agents optimize myopically
- Need coordination mechanisms

### Approaches
- **Opponent modeling**: Predict opponent strategy
- **Communication**: Signal intentions
- **Emergent cooperation**: Through learning dynamics (LOLA, SOS)
- **Meta-learning**: Learn to cooperate across games

## Interview Relevance

**Common Questions**:
1. **What's the Nash equilibrium?** (Defect, Defect) - unique, dominant strategy
2. **Is Nash optimal?** No - (Cooperate, Cooperate) is better for both (Pareto superior)
3. **Why is it called a dilemma?** Individual rationality leads to collective irrationality
4. **How to achieve cooperation?** Repeated games, reputation, contracts, altruism
5. **Iterated PD difference?** Future interactions incentivize cooperation (e.g., tit-for-tat)
6. **Real-world examples?** Climate change, arms race, public goods, vaccination
7. **Dominant strategy?** Defect - best regardless of opponent's choice
8. **Social optimum?** (Cooperate, Cooperate) but unstable

**Key Concepts**:
- **Dominant strategy**: Defect
- **Nash equilibrium**: (D, D)
- **Pareto optimal**: (C, C)
- **Social dilemma**: Nash ≠ Social optimum
- **Folk theorem**: Repeated interaction enables cooperation

**Key Insight**: Prisoner's Dilemma demonstrates that individual rationality doesn't guarantee collective welfare - a fundamental tension in multi-agent systems, economics, and society.
