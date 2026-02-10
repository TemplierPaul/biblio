# MuZero (2019)

"Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"

## Core Innovation
MuZero plans (using MCTS) with a learned model **without ever reconstructing the observation**.
-   Unlike AlphaZero, applied to Atari where state is pixels (high dimensional).
-   Unlike "World Models" (Dreamer), it focuses on "value equivalent" representations rather than pixel reconstruction.

## The Model
MuZero uses three neural networks:

1.  **Representation Function ($h$)**:
    $$s^0 = h(o_1, \dots, o_t)$$
    Encodes current observations into a hidden state $s^0$.

2.  **Dynamics Function ($g$)**:
    $$s^{k+1}, r^k = g(s^k, a^{k+1})$$
    Predicts the next hidden state and immediate reward given an action. This allows "imagining" future trajectories in latent space.

3.  **Prediction Function ($f$)**:
    $$p^k, v^k = f(s^k)$$
    Predicts the policy (probabilities over actions) and value (expected future return) from the hidden state, used by MCTS.

## Training
Trained end-to-end to minimize the error in policy ($p$), value ($v$), and reward ($r$) predictions against the ground truth from self-play games.
-   **No pixel reconstruction loss**: The hidden state $s$ only needs to contain information useful for predicting value and policy.

## Interview Relevance
-   **Model-Based RL**: MuZero is the bridge between model-free (DQN/PPO) and model-based methods.
-   **Planning in Latent Space**: The key idea that unlocked planning in pixel environments.
