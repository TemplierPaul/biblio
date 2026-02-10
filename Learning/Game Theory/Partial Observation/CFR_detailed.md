# CFR — Detailed Implementation Notes

> **Quick overview**: [[CFR]]

## Paper

**Title**: Regret Minimization in Games with Incomplete Information

**Authors**: Zinkevich, Johanson, Bowling, Piccione (U. Alberta)

**Published**: NIPS 2007

## Vanilla CFR Pseudocode

```python
class VanillaCFR:
    def __init__(self, game):
        self.regrets = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))

    def get_strategy(self, info_set):
        actions = self.game.legal_actions(info_set)
        regret_sum = sum(max(self.regrets[info_set][a], 0) for a in actions)
        strategy = {}
        for a in actions:
            if regret_sum > 0:
                strategy[a] = max(self.regrets[info_set][a], 0) / regret_sum
            else:
                strategy[a] = 1.0 / len(actions)
        return strategy

    def cfr(self, history, player, reach_probs):
        if self.game.is_terminal(history):
            return self.game.utility(history, player)

        if self.game.is_chance(history):
            value = 0
            for action, prob in self.game.chance_outcomes(history):
                value += prob * self.cfr(history + action, player, reach_probs)
            return value

        info_set = self.game.info_set(history)
        strategy = self.get_strategy(info_set)
        actions = self.game.legal_actions(info_set)

        action_values = {}
        cf_value = 0
        for action in actions:
            new_reach = reach_probs.copy()
            current_player = self.game.current_player(history)
            new_reach[current_player] *= strategy[action]
            action_values[action] = self.cfr(history + action, player, new_reach)
            cf_value += strategy[action] * action_values[action]

        if self.game.current_player(history) == player:
            for action in actions:
                regret = action_values[action] - cf_value
                opp_reach = prod(reach_probs[p] for p in range(num_players) if p != player)
                self.regrets[info_set][action] += opp_reach * regret
                self.strategy_sum[info_set][action] += reach_probs[player] * strategy[action]

        return cf_value

    def train(self, iterations):
        for _ in range(iterations):
            for player in range(self.game.num_players()):
                self.cfr("", player, [1.0] * self.game.num_players())

    def get_average_strategy(self):
        avg = {}
        for info_set in self.strategy_sum:
            total = sum(self.strategy_sum[info_set].values())
            avg[info_set] = {a: v / total for a, v in self.strategy_sum[info_set].items()}
        return avg
```

## CFR Variants

### CFR+
```
R^{t+1}(I,a) = max(R^t(I,a) + regret_update, 0)
```
Floor regrets at 0, alternating updates, linear averaging (w_t = t). Used in Libratus.

### MCCFR (Monte Carlo CFR)
- **External Sampling**: Sample chance + opponent, explore all own actions
- **Outcome Sampling**: Sample single path (fastest, high variance)

### Linear CFR
Weight w(t) = t → O(1/T²) convergence vs O(1/T) for vanilla.

### Discounted CFR
R^{t+1} = α·R^t + current_regret, where α ∈ (0,1).

## Deep CFR

Neural networks approximate regrets instead of tabular storage:

```
Architecture:
    Input: Info set encoding (cards, betting history)
    → LSTM/Transformer for sequential info (or MLP)
    → Regret head: R_θ(I,a)
    → Strategy head: σ(a) ∝ max(R(a), 0) or separate network

Training:
    for iteration t:
        for sampled trajectories:
            Compute counterfactual regrets
            Store (I, regret_vector)
        Train: loss = MSE(R_θ(I), target_regrets)
        Update average strategy via behavioral cloning
```

**Paper**: Brown et al., ICML 2019 — [1811.00164](https://arxiv.org/abs/1811.00164)

## Hyperparameters

### Vanilla CFR
| Parameter | Value |
|-----------|-------|
| Iterations | 10k – 10M |
| Other | Parameter-free |

### Deep CFR
| Parameter | Value |
|-----------|-------|
| Architecture | LSTM/Transformer + MLP |
| Hidden units | 128–512 |
| LR | 1e-4 to 1e-3 |
| Batch size | 128–2048 |
| Buffer size | 1M–10M |

## Evaluation Metrics

1. **Exploitability**: ε(σ) = max_{σ_opp} u(σ_opp, σ) - game_value. Nash ⇒ ε = 0.
2. **NashConv**: Σᵢ [max_{σᵢ} uᵢ(σᵢ, σ₋ᵢ) - uᵢ(σ)]
3. **Convergence**: Vanilla O(1/T), CFR+ O(1/T) lower constants, Linear O(1/T²)

## Benchmark Results

| Game | Info Sets | CFR Success |
|------|-----------|-------------|
| Kuhn Poker | ~10 | Solved exactly |
| Leduc Poker | ~10⁶ | Near-optimal |
| Limit Texas Hold'em | ~10¹⁴ | Superhuman (Cepheus, 2015) |
| No-Limit Texas Hold'em | ~10¹⁶¹ | Superhuman (Libratus/Pluribus) |

## Comparison

| Method | Info | Scalability | Convergence | Sample Eff. |
|--------|------|-------------|-------------|-------------|
| **CFR** | Imperfect | High (with variants) | Guaranteed | Medium |
| **NFSP** | Imperfect | High | Approximate | Medium |
| **AlphaZero** | Perfect | Very High | Optimal | High |
| **PSRO** | Both | Medium | Approximate | Low-Medium |

## Code Resources

- [OpenSpiel CFR](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/cfr.py)
- [PyTorch CFR](https://github.com/tansey/pycfr)
- [CFR Tutorial](https://justinsermeno.com/posts/cfr/)
- [LabML CFR](https://nn.labml.ai/cfr/)

## References

- [Regret Minimization in Games (Zinkevich et al., NIPS 2007)](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf)
- [Deep CFR (Brown et al., ICML 2019)](https://arxiv.org/abs/1811.00164)

## Related

- [[CFR]] — Quick overview (with formulas)
- [[FSP]] / [[FSP_detailed]] — Alternative equilibrium-finding
- [[PSRO]] — Population-based alternative
