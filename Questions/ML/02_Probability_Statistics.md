# Probability & Statistics - Interview Q&A

Comprehensive coverage of probability theory, statistical distributions, hypothesis testing, and Bayesian inference essential for machine learning.

---


## Table of Contents

- [[#Part 1: Probability Fundamentals]]
  - [[#What is a random variable?]]
  - [[#What is mean, median, and mode of a probability distribution?]]
  - [[#What's the difference between dependence and correlation?]]
- [[#Part 2: Distributions]]
  - [[#Review common probability distributions]]
- [[#Part 3: Bayesian Inference]]
  - [[#Explain the Bayes theorem]]
  - [[#What is a conjugate prior?]]
- [[#Part 4: Statistical Testing]]
  - [[#Explain hypothesis testing, p-value, and t-test]]
  - [[#Explain the Central Limit Theorem]]
- [[#Part 5: Bias-Variance Tradeoff]]
  - [[#Explain the bias-variance tradeoff]]

---

## Part 1: Probability Fundamentals

### What is a random variable?

**Definition:**
A random variable is a function that maps outcomes from a sample space to real numbers. It provides a numerical description of the outcomes of a random phenomenon.

**Types:**

**Discrete Random Variable:**
- Takes countable values
- Examples: dice roll, number of heads in coin flips, number of customers

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Example: Dice roll
outcomes = [1, 2, 3, 4, 5, 6]
probabilities = [1/6] * 6

plt.bar(outcomes, probabilities)
plt.xlabel('Outcome')
plt.ylabel('Probability')
plt.title('Discrete Random Variable: Fair Die')
plt.show()

# Probability mass function (PMF)
def pmf_dice(x):
    return 1/6 if 1 <= x <= 6 else 0

# Cumulative distribution function (CDF)
def cdf_dice(x):
    if x < 1:
        return 0
    elif x >= 6:
        return 1
    else:
        return int(x) / 6
```

**Continuous Random Variable:**
- Takes uncountably infinite values
- Examples: height, temperature, time

```python
# Example: Normal distribution
mu, sigma = 0, 1
x = np.linspace(-4, 4, 100)

# Probability density function (PDF)
pdf = stats.norm.pdf(x, mu, sigma)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, pdf)
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('PDF of Standard Normal Distribution')

# Cumulative distribution function (CDF)
cdf = stats.norm.cdf(x, mu, sigma)

plt.subplot(1, 2, 2)
plt.plot(x, cdf)
plt.xlabel('x')
plt.ylabel('Cumulative Probability')
plt.title('CDF of Standard Normal Distribution')

plt.tight_layout()
plt.show()

# Sample from distribution
samples = np.random.normal(mu, sigma, 1000)
print(f"Mean: {np.mean(samples):.3f}")
print(f"Std: {np.std(samples):.3f}")
```

**Key Properties:**

```python
# Expected Value (Mean)
def expected_value_discrete(values, probabilities):
    return np.sum(values * probabilities)

# Variance
def variance_discrete(values, probabilities):
    mu = expected_value_discrete(values, probabilities)
    return np.sum(probabilities * (values - mu)**2)

# Standard Deviation
def std_discrete(values, probabilities):
    return np.sqrt(variance_discrete(values, probabilities))

# Example: Dice
values = np.array([1, 2, 3, 4, 5, 6])
probs = np.array([1/6] * 6)

print(f"E[X] = {expected_value_discrete(values, probs):.3f}")  # 3.5
print(f"Var[X] = {variance_discrete(values, probs):.3f}")      # 2.917
print(f"Std[X] = {std_discrete(values, probs):.3f}")          # 1.708
```

**Transformations:**

```python
# If Y = g(X), what is the distribution of Y?

# Example: X ~ N(0, 1), Y = X²
X_samples = np.random.normal(0, 1, 10000)
Y_samples = X_samples**2

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(X_samples, bins=50, density=True, alpha=0.7)
plt.xlabel('X')
plt.ylabel('Density')
plt.title('X ~ N(0, 1)')

plt.subplot(1, 2, 2)
plt.hist(Y_samples, bins=50, density=True, alpha=0.7)
plt.xlabel('Y')
plt.ylabel('Density')
plt.title('Y = X² follows Chi-squared(1)')

plt.tight_layout()
plt.show()
```

---

### What is mean, median, and mode of a probability distribution?

**Mean (Expected Value, μ):**

The average value, weighted by probabilities.

```python
# Discrete
mean_discrete = np.sum(values * probabilities)

# Continuous (from samples)
mean_continuous = np.mean(samples)

# Properties of mean
# E[aX + b] = a*E[X] + b
# E[X + Y] = E[X] + E[Y]
# E[X*Y] = E[X]*E[Y] if X, Y independent

# Example
data = np.array([1, 2, 2, 3, 4, 5, 5, 5, 6, 7])
mean = np.mean(data)
print(f"Mean: {mean:.2f}")  # 4.0
```

**Median:**

The middle value that divides the distribution in half.

```python
# Median (50th percentile)
median = np.median(data)
print(f"Median: {median:.2f}")  # 4.5

# For continuous distributions
from scipy.stats import norm

# Median of standard normal is 0
median_norm = norm.median()
print(f"Median of N(0,1): {median_norm}")  # 0.0

# Percentiles
p25 = np.percentile(data, 25)
p75 = np.percentile(data, 75)
print(f"25th percentile: {p25}")
print(f"75th percentile: {p75}")
```

**Mode:**

The most frequent value (discrete) or maximum density point (continuous).

```python
from scipy import stats

# Mode for discrete data
mode_result = stats.mode(data)
mode = mode_result.mode
print(f"Mode: {mode}")  # 5

# Mode for continuous distribution (peak of PDF)
# For normal distribution, mode = mean = median
mode_norm = 0  # For N(0,1)

# Custom distribution with multiple modes
from scipy.stats import gaussian_kde

# Bimodal distribution
data_bimodal = np.concatenate([
    np.random.normal(-2, 0.5, 1000),
    np.random.normal(2, 0.5, 1000)
])

kde = gaussian_kde(data_bimodal)
x_range = np.linspace(-5, 5, 1000)
density = kde(x_range)

plt.plot(x_range, density)
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Bimodal Distribution (2 modes)')
plt.show()

# Find modes (local maxima)
from scipy.signal import argrelextrema
modes = x_range[argrelextrema(density, np.greater)[0]]
print(f"Modes: {modes}")
```

**Comparison:**

```python
# Symmetric distribution: mean = median = mode
symmetric = np.random.normal(0, 1, 10000)

# Right-skewed: mean > median > mode
right_skewed = np.random.exponential(1, 10000)

# Left-skewed: mean < median < mode
left_skewed = -right_skewed

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, data, title in zip(axes,
                            [symmetric, right_skewed, left_skewed],
                            ['Symmetric', 'Right-skewed', 'Left-skewed']):
    ax.hist(data, bins=50, density=True, alpha=0.7)
    ax.axvline(np.mean(data), color='r', linestyle='--', label='Mean')
    ax.axvline(np.median(data), color='g', linestyle='--', label='Median')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()

# Robustness to outliers
data_clean = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
data_outlier = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])

print(f"Clean data - Mean: {np.mean(data_clean):.1f}, Median: {np.median(data_clean):.1f}")
print(f"With outlier - Mean: {np.mean(data_outlier):.1f}, Median: {np.median(data_outlier):.1f}")
# Mean is affected by outlier, median is robust
```

---

### What's the difference between dependence and correlation?

**Correlation:**
- Measures **linear** relationship between two variables
- Pearson correlation coefficient: ρ ∈ [-1, 1]
- ρ = 0 means no linear relationship

**Dependence:**
- General relationship (can be non-linear)
- Statistical dependence: P(X,Y) ≠ P(X)P(Y)
- If variables are independent, they are also uncorrelated
- **But uncorrelated ≠ independent!**

**Key Difference:**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Example 1: Linear relationship (correlated AND dependent)
np.random.seed(42)
x1 = np.random.normal(0, 1, 1000)
y1 = 2 * x1 + np.random.normal(0, 0.5, 1000)

corr1, _ = pearsonr(x1, y1)
print(f"Linear relationship - Pearson correlation: {corr1:.3f}")  # ~0.97

# Example 2: No relationship (uncorrelated AND independent)
x2 = np.random.normal(0, 1, 1000)
y2 = np.random.normal(0, 1, 1000)

corr2, _ = pearsonr(x2, y2)
print(f"Independent - Pearson correlation: {corr2:.3f}")  # ~0.0

# Example 3: Quadratic relationship (DEPENDENT but uncorrelated!)
x3 = np.random.uniform(-2, 2, 1000)
y3 = x3**2 + np.random.normal(0, 0.1, 1000)

corr3, _ = pearsonr(x3, y3)
print(f"Quadratic relationship - Pearson correlation: {corr3:.3f}")  # ~0.0
# y3 clearly depends on x3, but Pearson correlation is near 0!

# Spearman correlation (captures monotonic relationships)
spearman_corr3, _ = spearmanr(x3, y3)
print(f"Quadratic relationship - Spearman correlation: {spearman_corr3:.3f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(x1, y1, alpha=0.5)
axes[0].set_title(f'Linear (ρ={corr1:.2f})')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

axes[1].scatter(x2, y2, alpha=0.5)
axes[1].set_title(f'Independent (ρ={corr2:.2f})')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')

axes[2].scatter(x3, y3, alpha=0.5)
axes[2].set_title(f'Dependent but Uncorrelated (ρ={corr3:.2f})')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')

plt.tight_layout()
plt.show()
```

**Mathematical Definitions:**

```python
# Covariance
def covariance(x, y):
    return np.mean((x - np.mean(x)) * (y - np.mean(y)))

# Pearson correlation (normalized covariance)
def pearson_correlation(x, y):
    return covariance(x, y) / (np.std(x) * np.std(y))

# Independence test using mutual information
from sklearn.metrics import mutual_info_score

def mutual_information(x, y, bins=10):
    # Discretize continuous variables
    x_discrete = np.digitize(x, bins=np.histogram(x, bins=bins)[1])
    y_discrete = np.digitize(y, bins=np.histogram(y, bins=bins)[1])
    return mutual_info_score(x_discrete, y_discrete)

# MI = 0 → independent, MI > 0 → dependent
mi1 = mutual_information(x1, y1)
mi2 = mutual_information(x2, y2)
mi3 = mutual_information(x3, y3)

print(f"MI (linear): {mi1:.3f}")
print(f"MI (independent): {mi2:.3f}")
print(f"MI (quadratic): {mi3:.3f}")
# MI captures the quadratic dependence!
```

**Types of Correlation:**

```python
from scipy.stats import pearsonr, spearmanr, kendalltau

# Pearson: measures linear correlation
pearson_r, pearson_p = pearsonr(x, y)

# Spearman: measures monotonic correlation (rank-based)
spearman_r, spearman_p = spearmanr(x, y)

# Kendall: measures ordinal association
kendall_tau, kendall_p = kendalltau(x, y)

print(f"Pearson r: {pearson_r:.3f} (p={pearson_p:.3f})")
print(f"Spearman ρ: {spearman_r:.3f} (p={spearman_p:.3f})")
print(f"Kendall τ: {kendall_tau:.3f} (p={kendall_p:.3f})")
```

**Summary:**
- **Correlation = 0** only means no *linear* relationship
- **Dependence** can exist even with correlation = 0
- Always visualize data, don't rely on correlation alone
- Use mutual information for general dependence

---

## Part 2: Distributions

### Review common probability distributions

**Discrete Distributions:**

**1. Bernoulli Distribution:**

Single trial with two outcomes (success/failure).

```python
from scipy.stats import bernoulli

# P(X=1) = p, P(X=0) = 1-p
p = 0.3
rv = bernoulli(p)

# PMF
print(f"P(X=0) = {rv.pmf(0):.2f}")  # 0.70
print(f"P(X=1) = {rv.pmf(1):.2f}")  # 0.30

# Mean and variance
print(f"E[X] = {rv.mean()}")        # 0.3
print(f"Var[X] = {rv.var():.3f}")   # 0.21 = p(1-p)

# Sample
samples = rv.rvs(size=1000)
```

**2. Binomial Distribution:**

Number of successes in n independent Bernoulli trials.

```python
from scipy.stats import binom

# B(n, p): n trials, probability p
n, p = 10, 0.3
rv = binom(n, p)

# PMF
x = np.arange(0, n+1)
pmf = rv.pmf(x)

plt.bar(x, pmf)
plt.xlabel('Number of successes')
plt.ylabel('Probability')
plt.title(f'Binomial Distribution B({n}, {p})')
plt.show()

# E[X] = np, Var[X] = np(1-p)
print(f"E[X] = {rv.mean()}")      # 3.0
print(f"Var[X] = {rv.var():.2f}") # 2.10

# Example: 10 coin flips, probability of exactly 3 heads
prob_3_heads = rv.pmf(3)
print(f"P(X=3) = {prob_3_heads:.4f}")
```

**3. Poisson Distribution:**

Number of events in fixed interval (rare events).

```python
from scipy.stats import poisson

# Poisson(λ): λ = rate parameter (mean)
lambda_ = 3.5
rv = poisson(lambda_)

x = np.arange(0, 15)
pmf = rv.pmf(x)

plt.bar(x, pmf)
plt.xlabel('Number of events')
plt.ylabel('Probability')
plt.title(f'Poisson Distribution (λ={lambda_})')
plt.show()

# E[X] = Var[X] = λ
print(f"E[X] = {rv.mean()}")
print(f"Var[X] = {rv.var()}")

# Example: Number of emails per hour
# P(X ≤ 5) = ?
prob_at_most_5 = rv.cdf(5)
print(f"P(X ≤ 5) = {prob_at_most_5:.4f}")
```

**Continuous Distributions:**

**4. Uniform Distribution:**

All values equally likely in [a, b].

```python
from scipy.stats import uniform

# U(a, b): loc=a, scale=b-a
a, b = 0, 1
rv = uniform(loc=a, scale=b-a)

x = np.linspace(-0.5, 1.5, 1000)
pdf = rv.pdf(x)

plt.plot(x, pdf)
plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Uniform Distribution U({a}, {b})')
plt.show()

# E[X] = (a+b)/2, Var[X] = (b-a)²/12
print(f"E[X] = {rv.mean()}")
print(f"Var[X] = {rv.var():.4f}")
```

**5. Normal (Gaussian) Distribution:**

Most important distribution in statistics.

```python
from scipy.stats import norm

# N(μ, σ²)
mu, sigma = 0, 1
rv = norm(loc=mu, scale=sigma)

x = np.linspace(-4, 4, 1000)
pdf = rv.pdf(x)

plt.plot(x, pdf)
plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Normal Distribution N({mu}, {sigma}²)')
plt.fill_between(x[(x >= -1) & (x <= 1)], pdf[(x >= -1) & (x <= 1)], alpha=0.3)
plt.show()

# 68-95-99.7 rule
print(f"P(μ-σ ≤ X ≤ μ+σ) = {rv.cdf(mu+sigma) - rv.cdf(mu-sigma):.3f}")  # 0.683
print(f"P(μ-2σ ≤ X ≤ μ+2σ) = {rv.cdf(mu+2*sigma) - rv.cdf(mu-2*sigma):.3f}")  # 0.954

# Standard normal z-scores
z_score = (x - mu) / sigma

# Inverse CDF (quantile function)
# What value has 95% of data below it?
quantile_95 = rv.ppf(0.95)
print(f"95th percentile: {quantile_95:.3f}")
```

**6. Exponential Distribution:**

Time between events in Poisson process.

```python
from scipy.stats import expon

# Exp(λ): scale = 1/λ
lambda_ = 0.5
rv = expon(scale=1/lambda_)

x = np.linspace(0, 10, 1000)
pdf = rv.pdf(x)

plt.plot(x, pdf)
plt.xlabel('Time')
plt.ylabel('Density')
plt.title(f'Exponential Distribution (λ={lambda_})')
plt.show()

# E[X] = 1/λ, Var[X] = 1/λ²
print(f"E[X] = {rv.mean()}")
print(f"Var[X] = {rv.var()}")

# Memoryless property: P(X > s+t | X > s) = P(X > t)
s, t = 2, 3
prob_conditional = (1 - rv.cdf(s+t)) / (1 - rv.cdf(s))
prob_unconditional = 1 - rv.cdf(t)
print(f"Conditional: {prob_conditional:.4f}")
print(f"Unconditional: {prob_unconditional:.4f}")
```

**7. Beta Distribution:**

Distribution on [0, 1], useful for modeling probabilities.

```python
from scipy.stats import beta

# Beta(α, β)
alpha, beta_param = 2, 5
rv = beta(alpha, beta_param)

x = np.linspace(0, 1, 1000)
pdf = rv.pdf(x)

plt.plot(x, pdf)
plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Beta Distribution B({alpha}, {beta_param})')
plt.show()

# E[X] = α/(α+β)
print(f"E[X] = {rv.mean():.3f}")
```

**8. Gamma Distribution:**

Generalizes exponential distribution.

```python
from scipy.stats import gamma

# Gamma(k, θ): shape=k, scale=θ
k, theta = 2, 2
rv = gamma(a=k, scale=theta)

x = np.linspace(0, 20, 1000)
pdf = rv.pdf(x)

plt.plot(x, pdf)
plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Gamma Distribution Γ({k}, {theta})')
plt.show()
```

**9. Student's t-Distribution:**

Used when sample size is small and population σ unknown.

```python
from scipy.stats import t

# t(ν): ν = degrees of freedom
df = 5
rv = t(df=df)

x = np.linspace(-4, 4, 1000)
pdf_t = rv.pdf(x)
pdf_normal = norm(0, 1).pdf(x)

plt.plot(x, pdf_t, label=f't-distribution (df={df})')
plt.plot(x, pdf_normal, label='Normal(0,1)', linestyle='--')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('t-Distribution vs Normal')
plt.legend()
plt.show()

# Heavier tails than normal
# As df → ∞, t → N(0,1)
```

**10. Chi-Squared Distribution:**

Sum of squared standard normals.

```python
from scipy.stats import chi2

# χ²(k): k = degrees of freedom
k = 5
rv = chi2(df=k)

x = np.linspace(0, 20, 1000)
pdf = rv.pdf(x)

plt.plot(x, pdf)
plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Chi-squared Distribution χ²({k})')
plt.show()

# Used in hypothesis testing (goodness of fit, independence)
```

**Distribution Summary Table:**

| Distribution | Parameters | Support | Mean | Variance | Use Case |
|--------------|-----------|---------|------|----------|----------|
| Bernoulli | p | {0, 1} | p | p(1-p) | Single trial |
| Binomial | n, p | {0,...,n} | np | np(1-p) | # successes |
| Poisson | λ | {0,1,2,...} | λ | λ | Rare events |
| Uniform | a, b | [a, b] | (a+b)/2 | (b-a)²/12 | Random selection |
| Normal | μ, σ² | ℝ | μ | σ² | Natural phenomena |
| Exponential | λ | [0, ∞) | 1/λ | 1/λ² | Wait times |
| Beta | α, β | [0, 1] | α/(α+β) | Complex | Probabilities |
| t | ν | ℝ | 0 | ν/(ν-2) | Small samples |
| Chi² | k | [0, ∞) | k | 2k | Goodness of fit |

---

## Part 3: Bayesian Inference

### Explain the Bayes theorem

**Bayes' Theorem:**

Relates conditional probabilities:

P(A|B) = P(B|A) × P(A) / P(B)

Or in ML context:

P(hypothesis|data) = P(data|hypothesis) × P(hypothesis) / P(data)

**Components:**

- **P(A|B)**: Posterior probability (what we want)
- **P(B|A)**: Likelihood (probability of data given hypothesis)
- **P(A)**: Prior probability (initial belief)
- **P(B)**: Evidence (normalizing constant)

**Example: Medical Diagnosis**

```python
# Disease test
# Sensitivity: P(positive|disease) = 0.99
# Specificity: P(negative|no disease) = 0.95
# Prevalence: P(disease) = 0.01

p_disease = 0.01
p_no_disease = 1 - p_disease
p_pos_given_disease = 0.99
p_pos_given_no_disease = 1 - 0.95  # False positive rate

# Evidence: P(positive)
p_positive = (p_pos_given_disease * p_disease +
              p_pos_given_no_disease * p_no_disease)

# Posterior: P(disease|positive)
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_positive

print(f"P(disease|positive) = {p_disease_given_pos:.4f}")  # ~0.17
# Only 17% chance of actually having disease despite positive test!
```

**Intuition:**

Even with a highly accurate test, if the disease is rare (low prior), a positive result doesn't necessarily mean you have the disease.

**Bayesian Updating:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Example: Coin flip - estimate probability of heads
# Prior: Beta(α, β) - conjugate prior for Bernoulli

def update_beta(alpha, beta, data):
    """Update Beta prior with observed data"""
    n_heads = np.sum(data == 1)
    n_tails = np.sum(data == 0)
    return alpha + n_heads, beta + n_tails

# Start with uniform prior: Beta(1, 1)
alpha, beta = 1, 1

# Observe coin flips
observations = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]

x = np.linspace(0, 1, 100)

plt.figure(figsize=(15, 5))

for i, obs in enumerate(observations):
    alpha, beta = update_beta(alpha, beta, [obs])

    if i in [0, 4, 9]:  # Plot at different stages
        plt.subplot(1, 3, i//5 + 1)
        pdf = stats.beta(alpha, beta).pdf(x)
        plt.plot(x, pdf)
        plt.xlabel('Probability of Heads')
        plt.ylabel('Density')
        plt.title(f'After {i+1} observations')
        plt.axvline(alpha / (alpha + beta), color='r', linestyle='--',
                    label=f'Mean = {alpha/(alpha+beta):.2f}')
        plt.legend()

plt.tight_layout()
plt.show()

# As we see more data, posterior gets more concentrated
```

**Bayesian vs Frequentist:**

```python
# Frequentist approach: fixed unknown parameter
from scipy.stats import binom

n_trials = 100
n_heads = 60
p_hat = n_heads / n_trials  # Point estimate

# Confidence interval
from statsmodels.stats.proportion import proportion_confint

ci_lower, ci_upper = proportion_confint(n_heads, n_trials, alpha=0.05)
print(f"Frequentist CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# Bayesian approach: parameter is a distribution
alpha_post = 1 + n_heads
beta_post = 1 + (n_trials - n_heads)

# Credible interval (95% probability mass)
credible_lower = stats.beta(alpha_post, beta_post).ppf(0.025)
credible_upper = stats.beta(alpha_post, beta_post).ppf(0.975)
print(f"Bayesian Credible Interval: [{credible_lower:.3f}, {credible_upper:.3f}]")

# Mean of posterior
posterior_mean = alpha_post / (alpha_post + beta_post)
print(f"Posterior mean: {posterior_mean:.3f}")
```

---

### What is a conjugate prior?

**Definition:**
A prior distribution is conjugate to a likelihood if the posterior has the same family as the prior.

**Benefit**: Analytical solution for posterior (no need for MCMC).

**Common Conjugate Pairs:**

| Likelihood | Conjugate Prior | Posterior | Use Case |
|------------|----------------|-----------|----------|
| Bernoulli/Binomial | Beta | Beta | Coin flips, CTR |
| Poisson | Gamma | Gamma | Event counts |
| Normal (known σ²) | Normal | Normal | Gaussian data |
| Normal (known μ) | Inverse-Gamma | Inverse-Gamma | Variance estimation |
| Multinomial | Dirichlet | Dirichlet | Topic modeling |

**Example: Beta-Binomial Conjugacy**

```python
from scipy.stats import beta, binom

# Prior: Beta(α₀, β₀)
alpha_0, beta_0 = 2, 2  # Slightly favors p=0.5

# Likelihood: Binomial(n, p)
n_trials = 10
n_successes = 7

# Posterior: Beta(α₀ + successes, β₀ + failures)
alpha_post = alpha_0 + n_successes
beta_post = beta_0 + (n_trials - n_successes)

# Visualization
x = np.linspace(0, 1, 100)

prior_pdf = beta(alpha_0, beta_0).pdf(x)
posterior_pdf = beta(alpha_post, beta_post).pdf(x)

# Likelihood (normalized)
likelihood = np.array([binom(n_trials, p).pmf(n_successes) for p in x])
likelihood = likelihood / np.max(likelihood)  # Normalize for visualization

plt.figure(figsize=(10, 6))
plt.plot(x, prior_pdf, label='Prior: Beta(2, 2)', linestyle='--')
plt.plot(x, likelihood, label=f'Likelihood: Binomial(n={n_trials}, k={n_successes})', alpha=0.5)
plt.plot(x, posterior_pdf, label=f'Posterior: Beta({alpha_post}, {beta_post})', linewidth=2)
plt.xlabel('Probability of Success (p)')
plt.ylabel('Density')
plt.title('Bayesian Update with Conjugate Prior')
plt.legend()
plt.show()

# Posterior mean (combines prior and data)
prior_mean = alpha_0 / (alpha_0 + beta_0)
mle = n_successes / n_trials  # Maximum likelihood estimate
posterior_mean = alpha_post / (alpha_post + beta_post)

print(f"Prior mean: {prior_mean:.3f}")
print(f"MLE: {mle:.3f}")
print(f"Posterior mean: {posterior_mean:.3f}")
# Posterior is between prior and MLE
```

**Gamma-Poisson Conjugacy:**

```python
from scipy.stats import gamma, poisson

# Prior: Gamma(α, β)
alpha_0, beta_0 = 2, 1

# Observed data: counts from Poisson
data = [3, 5, 4, 6, 2, 4, 5, 3]
n = len(data)
sum_data = sum(data)

# Posterior: Gamma(α + Σxᵢ, β + n)
alpha_post = alpha_0 + sum_data
beta_post = beta_0 + n

# Posterior mean (estimate of λ)
prior_mean = alpha_0 / beta_0
posterior_mean = alpha_post / beta_post
mle = np.mean(data)

print(f"Prior mean: {prior_mean:.3f}")
print(f"MLE: {mle:.3f}")
print(f"Posterior mean: {posterior_mean:.3f}")
```

**Why use conjugate priors:**
1. Analytical tractability
2. Interpretable updates (simple counting)
3. Computational efficiency
4. Sequential updating

**When not to use:**
- If prior doesn't match domain knowledge
- When non-conjugate prior is more appropriate
- Modern MCMC methods (PyMC, Stan) make non-conjugate priors practical

---

## Part 4: Statistical Testing

### Explain hypothesis testing, p-value, and t-test

**Hypothesis Testing Framework:**

1. **Null hypothesis (H₀)**: Default assumption (e.g., no effect, no difference)
2. **Alternative hypothesis (H₁)**: What we want to prove
3. **Test statistic**: Computed from data
4. **p-value**: Probability of observing data (or more extreme) if H₀ is true
5. **Significance level (α)**: Threshold for rejection (typically 0.05)

**Decision Rule:**
- If p-value < α: Reject H₀ (statistically significant)
- If p-value ≥ α: Fail to reject H₀

**p-value:**

```python
import numpy as np
from scipy import stats

# Example: Is a coin fair?
# H₀: p = 0.5 (coin is fair)
# H₁: p ≠ 0.5 (coin is biased)

n_flips = 100
n_heads = 60

# Test statistic: number of heads
# Under H₀, X ~ Binomial(100, 0.5)

# Two-tailed p-value
from scipy.stats import binom

# P(X ≥ 60 or X ≤ 40 | p=0.5)
p_value = 2 * (1 - binom.cdf(n_heads - 1, n_flips, 0.5))
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject H₀: Coin is likely biased")
else:
    print("Fail to reject H₀: Not enough evidence of bias")
```

**t-test:**

Used to compare means when sample size is small or population variance unknown.

**One-sample t-test:**

```python
# Is the mean significantly different from a hypothesized value?
# H₀: μ = μ₀
# H₁: μ ≠ μ₀

data = [23, 25, 22, 24, 26, 21, 24, 23, 25, 22]
mu_0 = 20  # Hypothesized mean

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(data, mu_0)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Manual calculation
sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)  # Use sample std (n-1)
n = len(data)

t_manual = (sample_mean - mu_0) / (sample_std / np.sqrt(n))
print(f"Manual t-statistic: {t_manual:.4f}")

# Degrees of freedom
df = n - 1

# Critical value for α=0.05, two-tailed
critical_value = stats.t.ppf(0.975, df)
print(f"Critical value: ±{critical_value:.4f}")

if abs(t_stat) > critical_value:
    print("Reject H₀")
```

**Two-sample t-test:**

```python
# Compare means of two groups
# H₀: μ₁ = μ₂
# H₁: μ₁ ≠ μ₂

group1 = [23, 25, 22, 24, 26, 21, 24]
group2 = [20, 22, 19, 21, 23, 20, 22]

# Independent samples t-test (unequal variances)
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Effect size (Cohen's d)
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

d = cohens_d(group1, group2)
print(f"Cohen's d: {d:.4f}")
# |d| < 0.2: small, |d| ~ 0.5: medium, |d| > 0.8: large
```

**Paired t-test:**

```python
# Compare before/after measurements
# H₀: μ_diff = 0

before = [120, 125, 130, 128, 122, 135, 127]
after = [115, 120, 125, 122, 118, 128, 122]

# Paired samples t-test
t_stat, p_value = stats.ttest_rel(before, after)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Equivalent to one-sample t-test on differences
differences = np.array(before) - np.array(after)
t_stat_diff, p_value_diff = stats.ttest_1samp(differences, 0)
print(f"Via differences: t={t_stat_diff:.4f}, p={p_value_diff:.4f}")
```

**Type I and Type II Errors:**

```python
# Type I error (α): Rejecting H₀ when it's true (false positive)
# Type II error (β): Failing to reject H₀ when it's false (false negative)
# Power (1-β): Probability of correctly rejecting false H₀

# Example: Power analysis
from statsmodels.stats.power import ttest_power

# Calculate required sample size
from statsmodels.stats.power import tt_solve_power

# Given effect size, power, and α, find required n
effect_size = 0.5  # Cohen's d
alpha = 0.05
power = 0.8

n_required = tt_solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power,
    alternative='two-sided'
)
print(f"Required sample size per group: {int(np.ceil(n_required))}")
```

**Multiple Testing Correction:**

```python
from statsmodels.stats.multitest import multipletests

# When performing multiple tests, adjust p-values
p_values = [0.01, 0.04, 0.03, 0.08, 0.001, 0.06]

# Bonferroni correction (conservative)
rejected_bonf, p_adjusted_bonf, _, _ = multipletests(
    p_values, alpha=0.05, method='bonferroni'
)

# Benjamini-Hochberg (FDR control, less conservative)
rejected_bh, p_adjusted_bh, _, _ = multipletests(
    p_values, alpha=0.05, method='fdr_bh'
)

print("Original p-values:", p_values)
print("Bonferroni adjusted:", p_adjusted_bonf)
print("BH adjusted:", p_adjusted_bh)
```

---

### Explain the Central Limit Theorem

**Statement:**

The sum (or average) of a large number of independent, identically distributed (i.i.d.) random variables, regardless of their original distribution, will approximately follow a normal distribution.

**Mathematical Form:**

If X₁, X₂, ..., Xₙ are i.i.d. with mean μ and variance σ², then:

(X̄ - μ) / (σ/√n) → N(0, 1) as n → ∞

where X̄ = (X₁ + ... + Xₙ) / n

**Demonstration:**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def demonstrate_clt(distribution_func, n_samples_list, n_simulations=10000):
    """Demonstrate CLT for any distribution"""

    fig, axes = plt.subplots(2, len(n_samples_list), figsize=(15, 8))

    for i, n in enumerate(n_samples_list):
        # Generate sample means
        sample_means = []
        for _ in range(n_simulations):
            sample = distribution_func(n)
            sample_means.append(np.mean(sample))

        sample_means = np.array(sample_means)

        # Plot histogram
        axes[0, i].hist(sample_means, bins=50, density=True, alpha=0.7, edgecolor='black')

        # Overlay normal distribution
        mu = np.mean(sample_means)
        sigma = np.std(sample_means)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        axes[0, i].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
        axes[0, i].set_title(f'Sample size: {n}')
        axes[0, i].legend()

        # Q-Q plot
        stats.probplot(sample_means, dist="norm", plot=axes[1, i])
        axes[1, i].set_title(f'Q-Q Plot (n={n})')

    plt.tight_layout()
    plt.show()

# Test with different distributions

# 1. Uniform distribution
print("Uniform Distribution:")
demonstrate_clt(
    lambda n: np.random.uniform(0, 1, n),
    n_samples_list=[2, 5, 30, 100]
)

# 2. Exponential distribution (highly skewed)
print("Exponential Distribution:")
demonstrate_clt(
    lambda n: np.random.exponential(1, n),
    n_samples_list=[2, 5, 30, 100]
)

# 3. Binomial distribution
print("Binomial Distribution:")
demonstrate_clt(
    lambda n: np.random.binomial(10, 0.3, n),
    n_samples_list=[2, 5, 30, 100]
)
```

**Practical Implications:**

```python
# 1. Confidence intervals for means
def mean_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for mean using CLT"""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)  # Standard error of mean
    margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - margin, mean + margin

data = np.random.exponential(2, 100)  # Non-normal data
mean, ci_lower, ci_upper = mean_confidence_interval(data)
print(f"Mean: {mean:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# 2. Sample size determination
# CLT justifies using normal approximation for large n

# 3. A/B testing
def ab_test_z_test(group_a, group_b):
    """Use CLT to perform z-test on large samples"""
    n_a, n_b = len(group_a), len(group_b)
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)

    # Standard error of difference
    se_diff = np.sqrt(var_a/n_a + var_b/n_b)

    # Z-statistic (valid for large n due to CLT)
    z = (mean_a - mean_b) / se_diff

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_value

# Large samples from non-normal distributions
group_a = np.random.exponential(2, 1000)
group_b = np.random.exponential(2.2, 1000)

z, p = ab_test_z_test(group_a, group_b)
print(f"Z-statistic: {z:.4f}, p-value: {p:.4f}")
```

**When CLT Applies:**
- i.i.d. random variables
- Finite variance
- Sufficiently large n (rule of thumb: n ≥ 30)

**When CLT Doesn't Apply:**
- Heavy-tailed distributions (no finite variance)
- Strong dependence between observations
- Very small sample sizes

---

## Part 5: Bias-Variance Tradeoff

### Explain the bias-variance tradeoff

**Decomposition of Expected Test Error:**

Expected MSE = Bias² + Variance + Irreducible Error

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Generate data
np.random.seed(42)
n = 100
X = np.sort(np.random.uniform(0, 10, n))
true_function = lambda x: np.sin(x) + 0.5 * x
y = true_function(X) + np.random.normal(0, 0.5, n)

X = X.reshape(-1, 1)
X_test = np.linspace(0, 10, 300).reshape(-1, 1)
y_true = true_function(X_test.ravel())

# Models with different complexity
models = {
    'High Bias (Linear)': LinearRegression(),
    'Balanced (RF depth=5)': RandomForestRegressor(max_depth=5, n_estimators=100, random_state=42),
    'High Variance (Tree depth=20)': DecisionTreeRegressor(max_depth=20, random_state=42)
}

plt.figure(figsize=(15, 5))

for i, (name, model) in enumerate(models.items()):
    model.fit(X, y)
    y_pred = model.predict(X_test)

    plt.subplot(1, 3, i+1)
    plt.scatter(X, y, alpha=0.3, label='Training data')
    plt.plot(X_test, y_true, 'g-', label='True function', linewidth=2)
    plt.plot(X_test, y_pred, 'r-', label='Model prediction', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(name)
    plt.legend()

plt.tight_layout()
plt.show()
```

**Bias:**
- Error from wrong assumptions in the learning algorithm
- High bias → underfitting
- Model is too simple to capture underlying pattern

**Variance:**
- Error from sensitivity to small fluctuations in training set
- High variance → overfitting
- Model captures noise as if it were signal

**Empirical Demonstration:**

```python
def bias_variance_decomposition(model_class, model_params, X_train, y_train, X_test, y_true, n_iterations=100):
    """Empirically compute bias and variance"""
    predictions = []

    for i in range(n_iterations):
        # Resample training data
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_sample = X_train[indices]
        y_sample = y_train[indices]

        # Train model
        model = model_class(**model_params)
        model.fit(X_sample, y_sample)

        # Predict
        y_pred = model.predict(X_test)
        predictions.append(y_pred)

    predictions = np.array(predictions)

    # Average prediction across models
    mean_prediction = np.mean(predictions, axis=0)

    # Bias² = (E[f̂(x)] - f(x))²
    bias_squared = np.mean((mean_prediction - y_true)**2)

    # Variance = E[(f̂(x) - E[f̂(x)])²]
    variance = np.mean(np.var(predictions, axis=0))

    # Total error
    mse = np.mean((predictions - y_true)**2)

    return bias_squared, variance, mse

# Compare different model complexities
complexities = [1, 3, 5, 10, 20]
biases = []
variances = []
mses = []

for max_depth in complexities:
    bias_sq, var, mse = bias_variance_decomposition(
        DecisionTreeRegressor,
        {'max_depth': max_depth, 'random_state': 42},
        X, y, X_test, y_true,
        n_iterations=50
    )
    biases.append(bias_sq)
    variances.append(var)
    mses.append(mse)

# Plot bias-variance tradeoff
plt.figure(figsize=(10, 6))
plt.plot(complexities, biases, 'o-', label='Bias²', linewidth=2)
plt.plot(complexities, variances, 's-', label='Variance', linewidth=2)
plt.plot(complexities, mses, '^-', label='Total Error (MSE)', linewidth=2)
plt.xlabel('Model Complexity (max_depth)')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.grid(True)
plt.show()

print("Optimal complexity:", complexities[np.argmin(mses)])
```

**Key Insights:**

| Model Complexity | Bias | Variance | Total Error | Behavior |
|-----------------|------|----------|-------------|----------|
| Too Simple | High | Low | High | Underfitting |
| Optimal | Medium | Medium | Minimum | Good generalization |
| Too Complex | Low | High | High | Overfitting |

**Strategies to Balance:**

```python
# 1. Regularization (reduce variance)
from sklearn.linear_model import Ridge, Lasso

# L2 regularization
ridge = Ridge(alpha=1.0)  # Increase alpha to reduce variance

# 2. Ensemble methods (reduce variance)
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor

# Bagging reduces variance
bagging = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=10),
    n_estimators=100,
    random_state=42
)

# Boosting reduces bias
boosting = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# 3. Cross-validation (find optimal complexity)
from sklearn.model_selection import cross_val_score

depths = range(1, 21)
cv_scores = []

for depth in depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

optimal_depth = depths[np.argmin(cv_scores)]
print(f"Optimal depth via CV: {optimal_depth}")

# 4. Learning curves (diagnose bias vs variance)
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeRegressor(max_depth=5),
    X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='neg_mean_squared_error'
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, -train_scores.mean(axis=1), label='Training error')
plt.plot(train_sizes, -val_scores.mean(axis=1), label='Validation error')
plt.xlabel('Training Set Size')
plt.ylabel('MSE')
plt.title('Learning Curves')
plt.legend()
plt.show()

# High bias: both errors high and close
# High variance: large gap between train and val errors
```

---

This covers the comprehensive Probability & Statistics content for ML interviews. The file includes detailed explanations, code examples, and visualizations for all key statistical concepts needed for machine learning.
