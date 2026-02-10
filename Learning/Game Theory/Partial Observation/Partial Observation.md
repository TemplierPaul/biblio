# Partial Observation in Game Theory

In many real-world scenarios, agents do not have access to the full state of the environment or the private information of other agents. This is modeled as **Partial Observation**.

## Core Models

### Partially Observable Stochastic Games (POSG)
A general framework for multi-agent decision-making under uncertainty.
-   **Structure**: Extension of Markov Decision Processes (MDPs) to multiple agents with partial observability.
-   **Difficulty**: Solving POSGs generally is NEXP-Complete (extremely hard).

### Dec-POMDP (Cooperative)
**Decentralized POMDP** is a specialization of POSG where all agents share the **same reward function** (common goal).
-   **Usage**: Multi-robot coordination, distributed sensor networks.

## Extensive Form with Imperfect Information (Competitive)
Standard games like Poker are often modeled as extensive form games with **Information Sets**.

### [[CFR|Counterfactual Regret Minimization (CFR)]]
The leading algorithm for solving large imperfect-information games (e.g., Texas Hold'em).
-   **Iterative Self-Play**: Minimizes regret against the set of all possible opponent strategies.
-   **Variants**: CFR+, MCCFR, and Deep CFR.
-   See detailed notes: [[Partial Observation/CFR|CFR Algorithm]].

## Interview Relevance
-   **Information Sets**: Nodes in the same information set are indistinguishable to the player.
-   **Believe State**: Tracking the probability distribution over possible states.
