# Nash Equilibrium (NE)

A fundamental concept in game theory where no player can benefit by changing their strategy if the strategies of all other players remain unchanged.

## Formal Definition
A strategy profile $s^* = (s_1^*, \dots, s_n^*)$ is a Nash Equilibrium if for every player $i$:
$$u_i(s_i^*, s_{-i}^*) \geq u_i(s_i, s_{-i}^*) \quad \forall s_i \in S_i$$
Where $s_{-i}^*$ denotes the strategies of all players other than $i$.

## Types of Equilibria
-   **Pure Strategy NE**: Players choose a single action with probability 1.
-   **Mixed Strategy NE**: Players choose a probability distribution over their actions.
-   **Existence**: Nash's Theorem (1950) states that every finite game has at least one Nash Equilibrium in mixed strategies.

## Finding NE in 2x2 Games
1.  **Check for Pure NE**: Use the "underline" method on the payoff matrix (underline best responses). If both payoffs in a cell are underlined, it's a Pure NE.
2.  **Find Mixed NE**:
    -   Assume Player 1 plays `Up` with probability $p$ and `Down` with $1-p$.
    -   Player 1's choice should make Player 2 indifferent between their own strategies.
    -   Set Player 2's expected payoffs from `Left` and `Right` equal to each other and solve for $p$.

## Refinements & Variants
-   **Subgame Perfect NE**: For sequential games.
-   **Correlated Equilibrium**: A broader concept where a "choreographer" suggests actions.
-   **Coarse Correlated Equilibrium (CCE)**: Relevant in NeuPL and large-scale MARL.

## Interview Questions
-   "Does every game have a Nash Equilibrium?" (Yes, in mixed strategies for finite games).
-   "Is a Nash Equilibrium always the best outcome for everyone?" (No, see **Prisoner's Dilemma**).
-   "How do you compute NE for large games?" (PSRO, CFR, Double Oracle).
