# Classical ML - Interview Q&A

**How to use**: Questions are formatted as collapsible headings. Try to answer before expanding.

---

## Part 1: Gaussian Processes

### What's a Gaussian Process?

**Definition**: Collection of random variables, any finite subset of which follows a **multivariate Gaussian** distribution.

**Distribution over functions**:
$$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

Where:
- $m(x)$: Mean function (often 0)
- $k(x, x')$: Covariance (kernel) function

**For any points** $X = \{x_1, \ldots, x_n\}$:
$$f(X) \sim \mathcal{N}(\mu, K)$$
Where $K_{ij} = k(x_i, x_j)$

**Intuition**: Instead of learning parameters, define distribution over entire functions.

### Explain mean and covariance (kernel) functions

**Mean function** $m(x)$:
- Expected value of $f(x)$
- Often set to 0 (assume data is centered)
- Can encode prior belief (e.g., linear trend)

**Kernel function** $k(x, x')$:
- **Covariance** between $f(x)$ and $f(x')$
- Encodes **similarity**: $k(x, x')$ high → $f(x) \approx f(x')$
- Determines function properties (smoothness, periodicity, etc.)

**Key insight**: "Similar inputs have similar outputs" - kernel defines similarity.

### Write the predictive mean and variance formulas

Given training data $(X, y)$, predict at test points $X_*$:

**Predictive mean**:
$$\mu_{f_*|y} = K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1} y$$

**Predictive variance**:
$$\Sigma_{f_*|y} = K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1} K(X, X_*)$$

Where:
- $K(X, X)$: Train-train covariance
- $K(X_*, X)$: Test-train covariance
- $K(X_*, X_*)$: Test-test covariance
- $\sigma_n^2$: Noise variance

**Key**: Exact Bayesian inference (condition Gaussian on observations).

### Why does GP provide uncertainty estimates?

**Bayesian framework**: Maintains full **posterior distribution** over functions.

**Predictive variance** tells us:
- **Near data**: Low variance (confident)
- **Far from data**: High variance (uncertain)

**Example**:
- Training points: x = [1, 2, 3], test point x = 1.5
- Variance at 1.5: Low (interpolating)
- Test point x = 10: High variance (extrapolating)

**Contrast with NN**:
- NN gives point prediction (no uncertainty)
- Need special techniques (ensembles, dropout) for uncertainty

**Why it works**: GP prediction is Gaussian → variance comes for free from conditioning.

### What's the RBF/squared exponential kernel?

$$k(x, x') = \sigma_f^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

**Parameters**:
- $\ell$: **Length scale** (how quickly correlation decays with distance)
- $\sigma_f^2$: Signal variance (vertical scale)

**Properties**:
- Smooth: Infinitely differentiable (smooth functions)
- Stationary: Depends only on $\|x - x'\|$
- Most common kernel

**Interpretation**:
- Small $\ell$: Functions vary quickly (wiggly)
- Large $\ell$: Functions vary slowly (smooth)

### How to tune hyperparameters (marginal likelihood)?

**Hyperparameters**: $\theta = \{\ell, \sigma_f^2, \sigma_n^2\}$ (kernel params + noise)

**Objective**: Maximize **marginal likelihood** $p(y | X, \theta)$

**Log marginal likelihood**:
$$\log p(y|X,\theta) = -\frac{1}{2} y^T K_y^{-1} y - \frac{1}{2} \log |K_y| - \frac{n}{2} \log 2\pi$$

Where $K_y = K(X,X) + \sigma_n^2 I$

**Three terms**:
1. Data fit: $y^T K_y^{-1} y$ (how well model explains data)
2. Complexity: $\log |K_y|$ (penalizes complex models)
3. Constant

**Optimization**: Gradient-based (L-BFGS, Adam) on $-\log p(y|X,\theta)$ (NLL)

**Automatic trade-off**: Balances fit and complexity (Bayesian Occam's razor).

### What's the computational complexity of GP?

**Training** (hyperparameter optimization):
- **Matrix inversion**: $O(n^3)$ for $K^{-1}$
- **Determinant**: $O(n^3)$ for $\log |K|$

**Prediction** (mean):
- $O(n^2)$ to compute $K(X_*, X) K^{-1}$
- $O(n)$ per test point

**Storage**: $O(n^2)$ for covariance matrix

**Bottleneck**: $O(n^3)$ scaling → **doesn't scale beyond ~10,000 points**

**Why**: Need to invert dense $n \times n$ matrix.

### How to scale GPs (sparse GP, inducing points)?

**Sparse GP** (inducing points):
1. Select $m \ll n$ inducing points (subset or pseudo-inputs)
2. Approximate full GP using only $m$ points
3. Complexity: $O(nm^2)$ training, $O(m^2)$ prediction

**Methods**:
- **FITC**: Fully Independent Training Conditional
- **SVGP**: Stochastic Variational GP (mini-batches)

**Trade-off**: $m$ = 100-1000 → approximate but scalable

**Other approaches**:
- **Local GPs**: Separate GPs per region
- **Deep GP**: Stack GPs (compositional)
- **Structured kernels**: Exploit Kronecker, Toeplitz structure

**Modern**: For very large n, use neural networks with uncertainty (ensembles, Bayesian NNs).

### When to use GP vs neural network?

**GP**:
- ✅ Small data (<10K points)
- ✅ Need uncertainty quantification
- ✅ Structured problem (can encode via kernel)
- ✅ Interpretability important
- ❌ Doesn't scale to large n or high dimensions
- **Use**: Bayesian optimization, robotics, active learning

**Neural Network**:
- ✅ Large data (millions of points)
- ✅ High-dimensional inputs
- ✅ Raw features (images, text)
- ❌ No natural uncertainty (need special techniques)
- **Use**: Deep learning, computer vision, NLP

**Middle ground**: Deep kernel learning (NN features + GP), neural tangent kernels.

### What's the connection between infinite-width NNs and GPs?

**Neural Tangent Kernel (NTK)** theorem:

**Infinite-width neural network** (during training) behaves like a **GP** with specific kernel.

**Key insight**:
- Finite width: Complex, non-linear dynamics
- **Infinite width**: Linearizes around initialization → GP with NTK kernel

**Implications**:
1. Connects deep learning and GPs
2. Explains why NNs generalize (GP prior)
3. Theoretical tool for understanding training dynamics

**Practical**: Finite-width NNs differ (non-linear feature learning), but infinite-width limit gives insights.
