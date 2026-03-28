# Backpropagation — From Scratch

High-level overview of backpropagation: the chain rule, common derivatives, and how gradients flow to every weight in a neural network.

---

## Table of Contents

- [What is Backpropagation?](#what-is-backpropagation)
- [The Chain Rule](#the-chain-rule)
  - [Scalar version](#scalar-version)
  - [Multivariate version](#multivariate-version)
  - [Vector / matrix version](#vector--matrix-version)
- [Standard Derivatives Reference](#standard-derivatives-reference)
- [Common Derivatives in Deep Learning](#common-derivatives-in-deep-learning)
  - [Activation functions](#activation-functions)
  - [Loss functions](#loss-functions)
  - [Linear layer](#linear-layer-the-workhorse)
  - [Batch Normalisation](#batch-normalisation)
  - [Self-Attention](#self-attention)
- [The Computational Graph](#the-computational-graph)
- [Full Backpropagation Through a 2-Layer Network](#full-backpropagation-through-a-2-layer-network)
- [Vanishing & Exploding Gradients](#vanishing--exploding-gradients)
- [Key Facts for Interviews](#key-facts-for-interviews)

---

## What is Backpropagation?

**Backpropagation** is the algorithm that computes the gradient of a scalar loss $L$ with respect to every parameter in a neural network. It is not a separate learning rule — it is just an efficient application of the **chain rule of calculus** traversed in reverse through the network's computational graph.

Training a neural network requires gradient descent:

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial L}{\partial \theta} \quad \text{for every parameter } \theta$$

Backprop computes all $\frac{\partial L}{\partial \theta}$ in a single backward pass at cost $O(P)$, where $P$ is the number of parameters. The naïve finite-difference alternative costs $O(P^2)$ and is completely impractical.

**Two phases:**
1. **Forward pass** — evaluate the network and compute $L$, caching all intermediate values
2. **Backward pass** — propagate $\partial L$ upstream through every operation, using cached values

---

## The Chain Rule

### Scalar version

If $z = f(y)$ and $y = g(x)$:

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

**Example:** $z = \sin(x^2)$

$$\text{let } u = x^2 \implies \frac{du}{dx} = 2x, \quad \frac{dz}{du} = \cos(u)$$

$$\frac{dz}{dx} = \cos(x^2) \cdot 2x$$

### Multivariate version

If $L = f(z_1, z_2, \ldots, z_n)$ and each $z_i = g_i(x)$:

$$\frac{\partial L}{\partial x} = \sum_i \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial x}$$

This **sum over paths** is critical: if $x$ influences $L$ through multiple routes (e.g. appears in two nodes of the graph), gradients from all paths are added.

### Vector / matrix version

For a vector-valued function $\mathbf{z} = f(\mathbf{x})$ with $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{z} \in \mathbb{R}^m$, and scalar $L$:

$$\frac{\partial L}{\partial \mathbf{x}} = \mathbf{J}^\top \cdot \frac{\partial L}{\partial \mathbf{z}}$$

where $\mathbf{J} = \frac{\partial \mathbf{z}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}$ is the Jacobian.

For element-wise operations (most activations), $\mathbf{J}$ is diagonal, so:

$$\frac{\partial L}{\partial \mathbf{x}} = \sigma'(\mathbf{x}) \odot \frac{\partial L}{\partial \mathbf{z}} \qquad (\odot = \text{element-wise multiply})$$

---

## Standard Derivatives Reference

### Elementary functions

| Function $f(x)$ | Derivative $f'(x)$ | Notes |
|---|---|---|
| $c$ (constant) | $0$ | |
| $x^n$ | $n x^{n-1}$ | Power rule |
| $e^x$ | $e^x$ | Unique: derivative = itself |
| $a^x$ | $a^x \ln a$ | General exponential |
| $\ln x$ | $\dfrac{1}{x}$ | Natural log |
| $\log_a x$ | $\dfrac{1}{x \ln a}$ | General log |
| $\dfrac{1}{x}$ | $-\dfrac{1}{x^2}$ | |
| $\sqrt{x}$ | $\dfrac{1}{2\sqrt{x}}$ | |

### Trigonometric

| Function | Derivative | Notes |
|---|---|---|
| $\sin x$ | $\cos x$ | |
| $\cos x$ | $-\sin x$ | Minus sign! |
| $\tan x$ | $\sec^2 x = \dfrac{1}{\cos^2 x}$ | |
| $\arcsin x$ | $\dfrac{1}{\sqrt{1-x^2}}$ | |
| $\arccos x$ | $-\dfrac{1}{\sqrt{1-x^2}}$ | |
| $\arctan x$ | $\dfrac{1}{1+x^2}$ | Useful in attention analysis |
| $\sinh x$ | $\cosh x$ | Hyperbolic |
| $\cosh x$ | $\sinh x$ | |
| $\tanh x$ | $1 - \tanh^2 x$ | Used in LSTM gates |

### Combination rules

| Rule | Formula |
|---|---|
| **Sum** | $(f + g)' = f' + g'$ |
| **Product** | $(fg)' = f'g + fg'$ |
| **Quotient** | $\left(\dfrac{f}{g}\right)' = \dfrac{f'g - fg'}{g^2}$ |
| **Chain** | $f(g(x))' = f'(g(x)) \cdot g'(x)$ |
| **Inverse** | $(f^{-1})'(x) = \dfrac{1}{f'(f^{-1}(x))}$ |

### Worked examples relevant to ML

**$\log \sigma(x)$ — appears in BCE loss:**

$$\frac{d}{dx}\left[\log \sigma(x)\right] = \frac{1}{\sigma(x)} \cdot \sigma(x)(1-\sigma(x)) = 1 - \sigma(x)$$

**$\log(1 + e^x)$ — softplus / log-sum-exp:**

$$\frac{d}{dx}\left[\log(1 + e^x)\right] = \frac{e^x}{1 + e^x} = \sigma(x)$$

Softplus is the antiderivative of sigmoid.

**$x \log x$ — entropy terms:**

$$\frac{d}{dx}[x \log x] = \log x + 1$$

**$\frac{1}{2}\|x\|^2$ — MSE / weight decay:**

$$\frac{d}{dx}\left[\frac{1}{2}x^2\right] = x \qquad \text{(gradient = the value itself)}$$

---

## Common Derivatives in Deep Learning

### Activation functions

**Sigmoid** $\sigma(x) = \dfrac{1}{1+e^{-x}}$

$$\sigma'(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr)$$

Derivation:

$$\sigma(x) = (1+e^{-x})^{-1}$$

$$\frac{d\sigma}{dx} = e^{-x}(1+e^{-x})^{-2} = \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} = \sigma(x)(1-\sigma(x))$$

Range of $\sigma'$: $[0,\, 0.25]$ — saturates → vanishing gradient!

---

**Tanh**

$$\tanh'(x) = 1 - \tanh^2(x)$$

Range: $(0,\,1]$ — still saturates but centred output helps.

---

**ReLU** $\max(0, x)$

$$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$

No saturation for $x > 0$; gradient $= 1$ always flows. Cons: "dead ReLU" — if $h < 0$ the neuron stops learning.

---

**Leaky ReLU** $\max(\alpha x,\, x)$, $\alpha \approx 0.01$

$$\text{LReLU}'(x) = \begin{cases} 1 & x > 0 \\ \alpha & x \leq 0 \end{cases}$$

---

**GELU** (used in GPT, BERT)

$$\text{GELU}(x) = x \cdot \Phi(x), \qquad \text{GELU}'(x) = \Phi(x) + x \cdot \phi(x)$$

where $\Phi$ is the standard normal CDF and $\phi$ its PDF.

---

**Softmax** $S_i = \dfrac{e^{x_i}}{\sum_j e^{x_j}}$

$$\frac{\partial S_i}{\partial x_j} = S_i(\delta_{ij} - S_j) \qquad \text{(Jacobian is NOT diagonal)}$$

In matrix form: $\mathbf{J} = \operatorname{diag}(\mathbf{S}) - \mathbf{S}\mathbf{S}^\top$

Combined with cross-entropy loss $L = -\log S_y$:

$$\frac{\partial L}{\partial x_i} = S_i - \mathbf{1}[i = y]$$

This is the elegant "output minus one-hot target" rule.

---

### Loss functions

| Loss | Formula | $\partial L / \partial \hat{y}$ |
|---|---|---|
| MSE | $\frac{1}{N}\sum(\hat{y}-y)^2$ | $\frac{2}{N}(\hat{y}-y)$ |
| MAE | $\frac{1}{N}\sum|\hat{y}-y|$ | $\frac{1}{N}\operatorname{sign}(\hat{y}-y)$ |
| Cross-Entropy | $-\sum_i y_i \log \hat{y}_i$ | $-y/\hat{y}$ (element-wise) |
| Binary CE | $-\bigl[y\log\hat{y} + (1-y)\log(1-\hat{y})\bigr]$ | $\frac{\hat{y}-y}{\hat{y}(1-\hat{y})}$ |
| Hinge | $\max(0,\, 1-y\hat{y})$ | $-y$ if margin violated, else $0$ |

**Softmax + CE combined:**

$$L = -\log\frac{e^{x_y}}{\sum_j e^{x_j}} \implies \frac{\partial L}{\partial x_i} = S_i - \mathbf{1}[i=y]$$

Predictions that are too high get pushed down; the true class gets pushed up.

---

### Linear layer (the workhorse)

Layer: $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$, with $\mathbf{z} \in \mathbb{R}^m$, $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{W} \in \mathbb{R}^{m \times n}$.

Given upstream gradient $\boldsymbol{\delta} = \frac{\partial L}{\partial \mathbf{z}} \in \mathbb{R}^m$:

$$\frac{\partial L}{\partial \mathbf{W}} = \boldsymbol{\delta} \mathbf{x}^\top \qquad \text{shape } [m \times n] \quad \text{(outer product)}$$

$$\frac{\partial L}{\partial \mathbf{b}} = \boldsymbol{\delta} \qquad \text{shape } [m]$$

$$\frac{\partial L}{\partial \mathbf{x}} = \mathbf{W}^\top \boldsymbol{\delta} \qquad \text{shape } [n] \quad \leftarrow \text{sent to previous layer}$$

**Intuition for $\frac{\partial L}{\partial \mathbf{W}} = \boldsymbol{\delta}\mathbf{x}^\top$:**

$W_{ij}$ connects $x_j$ to $z_i$. The scalar gradient is:

$$\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial W_{ij}} = \delta_i \cdot x_j$$

Written for all $i, j$ simultaneously: $\boldsymbol{\delta}\mathbf{x}^\top$ (outer product).

**Batched** ($\mathbf{x}$ is $[n \times B]$, $B$ = batch size):

$$\frac{\partial L}{\partial \mathbf{W}} = \frac{1}{B}\,\boldsymbol{\delta}\,\mathbf{x}^\top, \qquad \frac{\partial L}{\partial \mathbf{b}} = \frac{1}{B}\sum_b \boldsymbol{\delta}_b, \qquad \frac{\partial L}{\partial \mathbf{x}} = \mathbf{W}^\top \boldsymbol{\delta}$$

---

### Batch Normalisation

Forward:

$$\mu = \frac{1}{B}\sum_i x_i, \qquad \sigma^2 = \frac{1}{B}\sum_i (x_i - \mu)^2$$

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta$$

Backward:

$$\frac{\partial L}{\partial \gamma} = \sum_i \delta_i \hat{x}_i, \qquad \frac{\partial L}{\partial \beta} = \sum_i \delta_i$$

$$\frac{\partial L}{\partial x_i} = \frac{\gamma}{B\sigma}\Bigl[B\delta_i - \sum_j \delta_j - \hat{x}_i \sum_j \delta_j \hat{x}_j\Bigr]$$

The gradient passes through $\mu$ and $\sigma^2$, both of which depend on all $x_i$ — hence the correction terms.

---

### Self-Attention

Forward:

$$\mathbf{A} = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right), \qquad \mathbf{out} = \mathbf{A}\mathbf{V}$$

Backward (given $\frac{\partial L}{\partial \mathbf{out}}$):

$$\frac{\partial L}{\partial \mathbf{V}} = \mathbf{A}^\top \frac{\partial L}{\partial \mathbf{out}}$$

$$\frac{\partial L}{\partial \mathbf{A}} = \frac{\partial L}{\partial \mathbf{out}} \mathbf{V}^\top$$

$$\frac{\partial L}{\partial \mathbf{Q}} = \frac{1}{\sqrt{d}}\, \frac{\partial L}{\partial \text{scores}} \cdot \mathbf{K}, \qquad \frac{\partial L}{\partial \mathbf{K}} = \frac{1}{\sqrt{d}}\left(\frac{\partial L}{\partial \text{scores}}\right)^\top \mathbf{Q}$$

---

## The Computational Graph

A neural network can be represented as a **directed acyclic graph (DAG)**:
- **Nodes** = operations (add, multiply, sigmoid, matmul, …)
- **Edges** = tensors flowing between operations

**Backprop traversal rules:**
1. Seed the output with gradient $\frac{\partial L}{\partial L} = 1$
2. Process nodes in **reverse topological order**
3. At each node: multiply upstream gradient by the local Jacobian
4. If a node receives gradients from multiple downstream nodes: **sum them**
5. Send the result to each input of the node

**Example** — $L = (ab + c)^2$, with $a=2,\, b=3,\, c=1$:

```
     a ──[×]── s₁ ──[+]── s₂ ──[²]── L
     b ──┘          │
     c ─────────────┘
```

Forward: $s_1 = 6,\; s_2 = 7,\; L = 49$

Backward $\left(\frac{\partial L}{\partial L} = 1\right)$:

$$\frac{\partial L}{\partial s_2} = 2s_2 = 14, \quad \frac{\partial L}{\partial s_1} = 14, \quad \frac{\partial L}{\partial c} = 14$$

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial s_1} \cdot b = 14 \cdot 3 = 42, \quad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial s_1} \cdot a = 14 \cdot 2 = 28$$

**Reverse mode vs forward mode:**

| Mode | Traversal | Cost | Best for |
|---|---|---|---|
| Reverse (backprop) | Output → input | $O(P)$ | Scalar loss, many params |
| Forward | Input → output | $O(P \cdot \text{outputs})$ | Few inputs, many outputs |

Neural nets: 1 scalar loss, millions of params → reverse mode always.

---

## Full Backpropagation Through a 2-Layer Network

**Architecture:**

$$\mathbf{x} \in \mathbb{R}^n \xrightarrow{W_1, b_1} \mathbf{h}_1 = \mathbf{W}_1\mathbf{x}+\mathbf{b}_1 \xrightarrow{\sigma} \mathbf{a}_1 \xrightarrow{W_2, b_2} h_2 = \mathbf{W}_2\mathbf{a}_1+b_2 \xrightarrow{\sigma} \hat{y} \xrightarrow{\text{BCE}} L$$

### Forward pass

$$\mathbf{h}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1 \quad \text{(pre-activation layer 1, cache)}$$

$$\mathbf{a}_1 = \sigma(\mathbf{h}_1) \quad \text{(activation layer 1, cache)}$$

$$h_2 = \mathbf{W}_2 \mathbf{a}_1 + b_2 \quad \text{(pre-activation layer 2)}$$

$$\hat{y} = \sigma(h_2) \quad \text{(output, cache)}$$

$$L = -\bigl[y \log \hat{y} + (1-y)\log(1-\hat{y})\bigr]$$

### Backward pass

**Step 1 — seed + output layer (BCE + sigmoid combined):**

$$\delta_2 = \hat{y} - y$$

**Step 2 — gradients for $\mathbf{W}_2$, $b_2$:**

$$\frac{\partial L}{\partial \mathbf{W}_2} = \delta_2 \, \mathbf{a}_1^\top, \qquad \frac{\partial L}{\partial b_2} = \delta_2$$

**Step 3 — pass gradient backward through $\mathbf{W}_2$:**

$$\frac{\partial L}{\partial \mathbf{a}_1} = \mathbf{W}_2^\top \delta_2$$

**Step 4 — backprop through sigmoid of layer 1:**

$$\boldsymbol{\delta}_1 = \frac{\partial L}{\partial \mathbf{a}_1} \odot \mathbf{a}_1 \odot (1 - \mathbf{a}_1)$$

**Step 5 — gradients for $\mathbf{W}_1$, $\mathbf{b}_1$:**

$$\frac{\partial L}{\partial \mathbf{W}_1} = \boldsymbol{\delta}_1 \mathbf{x}^\top, \qquad \frac{\partial L}{\partial \mathbf{b}_1} = \boldsymbol{\delta}_1$$

**Summary table:**

| Gradient | Expression | Needs from cache |
|---|---|---|
| $\delta_2$ | $\hat{y} - y$ | $\hat{y}$ |
| $\partial L/\partial \mathbf{W}_2$ | $\delta_2 \mathbf{a}_1^\top$ | $\mathbf{a}_1,\, \delta_2$ |
| $\partial L/\partial b_2$ | $\delta_2$ | $\delta_2$ |
| $\boldsymbol{\delta}_1$ | $(\mathbf{W}_2^\top\delta_2) \odot \mathbf{a}_1(1-\mathbf{a}_1)$ | $\mathbf{W}_2,\, \mathbf{a}_1,\, \delta_2$ |
| $\partial L/\partial \mathbf{W}_1$ | $\boldsymbol{\delta}_1 \mathbf{x}^\top$ | $\mathbf{x},\, \boldsymbol{\delta}_1$ |
| $\partial L/\partial \mathbf{b}_1$ | $\boldsymbol{\delta}_1$ | $\boldsymbol{\delta}_1$ |

### NumPy implementation

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Init
rng = np.random.default_rng(0)
W1 = rng.normal(0, 0.01, (4, 2))    # 4 hidden, 2 inputs
b1 = np.zeros(4)
W2 = rng.normal(0, 0.01, (1, 4))    # 1 output, 4 hidden
b2 = np.zeros(1)
lr = 0.1

def forward(x, y):
    h1   = W1 @ x + b1
    a1   = sigmoid(h1)
    h2   = W2 @ a1 + b2
    yhat = sigmoid(h2)
    loss = -(y * np.log(yhat + 1e-9) + (1 - y) * np.log(1 - yhat + 1e-9))
    cache = (x, h1, a1, h2, yhat)
    return loss, cache

def backward(cache, y):
    x, h1, a1, h2, yhat = cache

    # Layer 2
    delta2  = yhat - y                         # [1]
    dW2     = np.outer(delta2, a1)             # [1, 4]
    db2     = delta2.copy()                    # [1]

    # Layer 1
    da1     = W2.T @ delta2                    # [4]
    delta1  = da1 * a1 * (1 - a1)             # [4]
    dW1     = np.outer(delta1, x)              # [4, 2]
    db1     = delta1.copy()                    # [4]

    return dW1, db1, dW2, db2

def update(dW1, db1, dW2, db2):
    global W1, b1, W2, b2
    W1 -= lr * dW1;  b1 -= lr * db1
    W2 -= lr * dW2;  b2 -= lr * db2

# Training loop
x = np.array([0.5, 0.8]);  y = np.array([1.0])
for step in range(1000):
    loss, cache = forward(x, y)
    grads = backward(cache, y)
    update(*grads)
    if step % 200 == 0:
        print(f"step {step:4d}  loss {loss[0]:.4f}")
```

---

## Vanishing & Exploding Gradients

For an $L$-layer network the gradient chain is:

$$\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial \mathbf{a}_L} \cdot \mathbf{J}_L \cdot \mathbf{J}_{L-1} \cdots \mathbf{J}_2 \cdot \mathbf{J}_1$$

where $\mathbf{J}_k = \frac{\partial \mathbf{a}_k}{\partial \mathbf{a}_{k-1}}$ is the Jacobian of layer $k$.

- If the largest singular value of each $\mathbf{J}_k < 1$: product $\to 0$ exponentially **(vanishing)**
- If the largest singular value of each $\mathbf{J}_k > 1$: product $\to \infty$ exponentially **(exploding)**

### Sigmoid saturates gradients

$$\sigma'(x) \leq 0.25 \quad \forall x$$

For a 10-layer sigmoid network:

$$\text{gradient} \leq 0.25^{10} \approx 10^{-6}$$

### Solutions

| Problem | Solution | Why it helps |
|---|---|---|
| Vanishing | ReLU | $\text{ReLU}'(x) = 1$ for $x>0$, no saturation |
| Vanishing | Residual connections | Identity term in gradient; see below |
| Vanishing | BatchNorm | Keeps pre-activations in active regime |
| Vanishing | LSTM / GRU gates | Gates selectively pass gradients over time |
| Exploding | Gradient clipping | Cap norm: $\mathbf{g} \leftarrow \mathbf{g} \cdot \min\!\left(1, \frac{c}{\|\mathbf{g}\|}\right)$ |
| Both | Xavier / He init | Preserves variance layer-to-layer |

**Residual connections:**

$$\mathbf{y} = F(\mathbf{x}) + \mathbf{x} \implies \frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \cdot \left(\frac{\partial F}{\partial \mathbf{x}} + \mathbf{I}\right)$$

The $+\mathbf{I}$ term means the gradient always has a direct path back regardless of $\frac{\partial F}{\partial \mathbf{x}}$.

**Xavier initialisation** (sigmoid/tanh) — preserves forward and backward variance:

$$\operatorname{Var}[W] = \frac{2}{n_\text{in} + n_\text{out}}$$

**He initialisation** (ReLU) — compensates for ReLU zeroing half its inputs:

$$\operatorname{Var}[W] = \frac{2}{n_\text{in}}$$

---

## Key Facts for Interviews

1. **Backprop = chain rule on a DAG**, applied in reverse topological order
2. **Two passes**: forward (cache activations) + backward (use cache to compute gradients)
3. **Cost**: $O(P)$ — same order as a forward pass
4. $\frac{\partial L}{\partial \mathbf{W}} = \boldsymbol{\delta} \mathbf{x}^\top$ for every linear layer — outer product of error signal and input
5. **BCE + sigmoid output** $\Rightarrow$ gradient $= \hat{y} - y$ — the "output minus target" rule
6. **Vanishing gradients**: product of small Jacobians; fix with ReLU, skip connections, init
7. **Exploding gradients**: product of large Jacobians; fix with clipping, normalisation

---

## Related Topics

- [Optimization (gradient descent, Adam)](../README.md)
- [Questions/ML/01_Math_Foundations.md](../../Questions/ML/01_Math_Foundations.md) — Q&A on backprop
- [Losses](./Losses.md) — loss function gradients
- [Questions/ML/07_ML_Fundamentals.md](../../Questions/ML/07_ML_Fundamentals.md) — training Q&A
