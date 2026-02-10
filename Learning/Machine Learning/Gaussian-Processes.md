# Gaussian Processes (GP)

## Definition
A Gaussian Process is a collection of random variables, any finite subset of which follows a multivariate Gaussian distribution. It's a non-parametric Bayesian approach to regression and classification that provides uncertainty estimates.

## Core Concept

### Function Distribution
Instead of learning parameters, GP defines a **distribution over functions**:
$$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

Where:
- $m(x)$: Mean function (often 0)
- $k(x, x')$: Covariance (kernel) function

**Intuition**: "Similar inputs have similar outputs"

### Finite-Dimensional Distribution
For any finite set of points $X = \{x_1, \ldots, x_n\}$:
$$f(X) = [f(x_1), \ldots, f(x_n)]^T \sim \mathcal{N}(\mu, K)$$

Where:
- $\mu = [m(x_1), \ldots, m(x_n)]^T$
- $K_{ij} = k(x_i, x_j)$ (covariance matrix)

## Gaussian Process Regression

### Setup
- **Training data**: $(X, y)$ where $y = f(X) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma_n^2 I)$
- **Goal**: Predict $f_*$ at test points $X_*$

### Prior
$$\begin{bmatrix} f \\ f_* \end{bmatrix} \sim \mathcal{N}\left(0, \begin{bmatrix} K(X, X) + \sigma_n^2 I & K(X, X_*) \\ K(X_*, X) & K(X_*, X_*) \end{bmatrix}\right)$$

### Posterior (Prediction)
**Predictive mean**:
$$\mu_{f_*|y} = K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1} y$$

**Predictive variance**:
$$\Sigma_{f_*|y} = K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma_n^2 I]^{-1} K(X, X_*)$$

**Key property**: Prediction is **exact** (no approximation, just conditioning)

### Uncertainty
- **Epistemic**: Model uncertainty (captured by GP variance)
- Variance **decreases** near training data
- Variance **increases** far from data
- Critical for active learning, Bayesian optimization

## Kernel Functions

### 1. Squared Exponential (RBF)
$$k(x, x') = \sigma_f^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

- $\ell$: Length scale (how quickly correlation decays)
- $\sigma_f^2$: Signal variance
- **Properties**: Infinitely differentiable, smooth functions
- **Most common** kernel

### 2. Matérn Kernel
$$k(x, x') = \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\sqrt{2\nu} \frac{r}{\ell}\right)^\nu K_\nu\left(\sqrt{2\nu} \frac{r}{\ell}\right)$$

Where $r = \|x - x'\|$

**Special cases**:
- $\nu = 1/2$: Exponential (rough functions)
- $\nu = 3/2$: Once differentiable
- $\nu = 5/2$: Twice differentiable
- $\nu \to \infty$: RBF

**Advantage**: Control smoothness via $\nu$

### 3. Linear Kernel
$$k(x, x') = \sigma_b^2 + \sigma_v^2 (x - c)(x' - c)$$

- Represents linear functions
- $c$: Center

### 4. Periodic Kernel
$$k(x, x') = \sigma^2 \exp\left(-\frac{2\sin^2(\pi |x-x'|/p)}{\ell^2}\right)$$

- $p$: Period
- For repeating patterns (seasonality)

### Kernel Composition
**Sum**: $k_1 + k_2$ (OR: satisfy $k_1$ OR $k_2$)
**Product**: $k_1 \times k_2$ (AND: satisfy both)

**Example**: Periodic + linear trend
$$k = k_{\text{periodic}} + k_{\text{linear}}$$

## Hyperparameters

### Kernel Hyperparameters
- Length scale $\ell$
- Signal variance $\sigma_f^2$
- Noise variance $\sigma_n^2$

### Optimization: Marginal Likelihood

**Log marginal likelihood**:
$$\log p(y|X, \theta) = -\frac{1}{2} y^T K_y^{-1} y - \frac{1}{2} \log |K_y| - \frac{n}{2} \log 2\pi$$

Where $K_y = K(X, X) + \sigma_n^2 I$

**Three terms**:
1. **Data fit**: $y^T K_y^{-1} y$ (how well data explained)
2. **Complexity penalty**: $\log |K_y|$ (model complexity)
3. **Constant**: $\frac{n}{2} \log 2\pi$

**Optimization**: Gradient-based (L-BFGS, Adam)

### Negative Log-Likelihood (NLL)
$$\text{NLL} = -\log p(y|X, \theta) = \frac{1}{2} y^T K_y^{-1} y + \frac{1}{2} \log |K_y| + \frac{n}{2} \log 2\pi$$

**Used as loss function** for hyperparameter tuning

## Computational Complexity

### Training
- **Matrix inversion**: $O(n^3)$
- **Storage**: $O(n^2)$ for covariance matrix

**Problem**: Doesn't scale beyond ~10,000 points

### Solutions for Scalability

#### 1. Sparse GP (Inducing Points)
- Select $m \ll n$ inducing points
- Approximate full GP with $m$ points
- Complexity: $O(nm^2)$
- Examples: FITC, SVGP

#### 2. Local GP
- Partition input space
- Train separate GPs per region
- Combine predictions

#### 3. Deep GP
- Stack GPs (compositional)
- Can model more complex functions
- Still expensive

#### 4. Neural Network Approximations
- **Deep Kernel Learning**: GP with neural network features
- **Neural Tangent Kernel**: Infinite-width NNs equivalent to GPs

## GP Classification

### Binary Classification
- **Likelihood**: Bernoulli $p(y|f) = \sigma(f)^y (1-\sigma(f))^{1-y}$
- **Prior**: $f \sim \mathcal{GP}(0, k)$

**Problem**: Posterior is **non-Gaussian** (no closed form)

**Solutions**:
1. **Laplace Approximation**: Gaussian approximation at mode
2. **Expectation Propagation**: Iterative refinement
3. **Variational Inference**: Lower bound optimization
4. **MCMC**: Sample from posterior

## Applications

### Bayesian Optimization
- **Goal**: Optimize expensive black-box function $f(x)$
- **GP models $f$**: Provides mean and uncertainty
- **Acquisition function**: Balance exploration/exploitation
  - EI (Expected Improvement)
  - UCB (Upper Confidence Bound)
  - PI (Probability of Improvement)

**Used in**: Hyperparameter tuning, A/B testing, experimental design

### Robotics
- Model dynamics: $s_{t+1} = f(s_t, a_t)$
- Uncertainty-aware control
- Safe exploration

### Spatial Statistics (Kriging)
- Geostatistics: predict values at unobserved locations
- Weather, mining, environmental monitoring

### Time Series
- With periodic/Matérn kernels
- Forecasting with uncertainty

## Advantages

1. **Uncertainty quantification**: Natural Bayesian framework
2. **Non-parametric**: Complexity grows with data
3. **Kernel flexibility**: Encode prior knowledge (periodicity, smoothness)
4. **Exact inference**: No approximation in standard GP regression
5. **Interpretability**: Kernel hyperparameters have clear meaning

## Disadvantages

1. **Scalability**: $O(n^3)$ training, $O(n^2)$ storage
2. **Kernel choice**: Requires domain knowledge
3. **High dimensions**: Curse of dimensionality (need many samples)
4. **Classification**: No closed-form posterior, need approximations

## GP vs Neural Networks

| Aspect | GP | Neural Network |
|--------|----|-|
| Uncertainty | Natural | Requires special techniques |
| Data efficiency | High | Requires large datasets |
| Scalability | Limited (~10K points) | Millions of points |
| Interpretability | High (kernel) | Low (black box) |
| Prior knowledge | Kernel design | Architecture design |

**When to use GP**: Small data, need uncertainty, structured problems
**When to use NN**: Large data, high-dimensional, raw features

## Interview Relevance

**Common Questions**:
1. **What is a GP?** Distribution over functions, any finite subset is Gaussian
2. **Mean and covariance?** $m(x)$ (often 0), $k(x, x')$ (kernel)
3. **GP prediction?** Condition joint Gaussian on observed data (exact)
4. **Uncertainty?** Predictive variance: high far from data, low near data
5. **Common kernels?** RBF (smooth), Matérn (control smoothness), Periodic, Linear
6. **Hyperparameter tuning?** Maximize marginal likelihood (NLL as loss)
7. **Computational cost?** $O(n^3)$ training (matrix inversion)
8. **Scalability solutions?** Sparse GP, inducing points, local GPs
9. **GP vs NN?** GP: uncertainty, data-efficient; NN: scalable, large data
10. **Applications?** Bayesian optimization, robotics, spatial statistics

**Key Formulas**:
- Predictive mean: $\mu_{f_*|y} = K(X_*, X) [K(X,X) + \sigma_n^2 I]^{-1} y$
- NLL: $\frac{1}{2} y^T K_y^{-1} y + \frac{1}{2} \log |K_y| + \text{const}$
- RBF kernel: $k(x, x') = \sigma_f^2 \exp(-\frac{\|x-x'\|^2}{2\ell^2})$

**Key Insight**: GPs provide a principled Bayesian approach to regression with uncertainty quantification, trading scalability for exact probabilistic inference.
