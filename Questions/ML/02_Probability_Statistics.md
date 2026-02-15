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
- [[#Part 6: Information Theory & Divergence]]
  - [[#Explain KL Divergence (Relative Entropy)]]

---


## Part 1: Probability Fundamentals

### What is a random variable?

**Definition:**
A random variable $X$ is a function that maps outcomes from a random process to real numbers. It quantifies uncertainty.

**1. Discrete Random Variable**
*   **Definition**: Takes primarily countable values (e.g., integers).
*   **PMF (Probability Mass Function)**: $P(X=x)$. Probability that $X$ equals a specific value $x$.
*   **CDF (Cumulative Distribution Function)**: $F(x) = P(X \le x)$. Sum of probabilities for all outcomes $\le x$.
*   **Example**: Rolling a Fair Die.
    *   Sample Space: $\{1, 2, 3, 4, 5, 6\}$
    *   PMF: $P(X=k) = 1/6$ for all $k \in \{1..6\}$.

**2. Continuous Random Variable**
*   **Definition**: Takes values from an uncountably infinite range (e.g., real numbers, time, height).
*   **PDF (Probability Density Function)**: $f(x)$. Represents the *density* of probability at point $x$.
    *   *Crucial Note*: The probability of any single exact point is zero ($P(X=x) = 0$). Probabilities are only defined over intervals (area under the curve).
*   **CDF**: $F(x) = \int_{-\infty}^{x} f(t) dt$. The area under the PDF curve up to $x$.
*   **Example**: Standard Normal Distribution $Z \sim \mathcal{N}(0, 1)$.

**Key Properties:**

*   **Expected Value (Mean)**: The long-run average.
    *   Discrete: $E[X] = \sum x_i \cdot P(x_i)$
    *   Continuous: $E[X] = \int x \cdot f(x) dx$
    *   *Linearity*: $E[aX + bY] = aE[X] + bE[Y]$ (Works for dependent variables too!).

*   **Variance**: Measure of spread or dispersion around the mean.
    *   $\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$
    *   *Properties*: $\text{Var}(aX + b) = a^2 \text{Var}(X)$. (Adding a constant $b$ shifts the distribution but doesn't widen it; multiplying by $a$ scales the spread by $a^2$).

*   **Standard Deviation**: $\sigma = \sqrt{\text{Var}(X)}$. Reported in the same units as the original data.

**Transformations:**
If $Y = g(X)$, how does the distribution change?
*   **Linear**: If $X \sim \mathcal{N}(\mu, \sigma^2)$ and $Y = aX + b$, then $Y \sim \mathcal{N}(a\mu + b, a^2\sigma^2)$.
*   **Non-linear**: If $X \sim \mathcal{N}(0,1)$ and $Y = X^2$, then $Y$ follows a **Chi-Squared** distribution ($\chi^2_1$). This transformation is fundamental to many statistical tests (like the Chi-Square test).

---

### What is mean, median, and mode of a probability distribution?

**1. Definitions:**

*   **Mean ($\mu$)**: The weighted average or center of mass.
    *   *Pros*: Uses all data points; mathematically easy to handle.
    *   *Cons*: Higly sensitive to outliers.
*   **Median**: The middle value (50th percentile) where $F(m) = 0.5$.
    *   *Pros*: Robust to outliers.
    *   *Cons*: Harder to use in mathematical derivations.
*   **Mode**: The most frequent value (discrete) or peak density (continuous).
    *   *Pros*: Represents the "most likely" outcome.

**2. Comparison & Skewness:**

| Relationship | Distribution Shape | Example |
| :--- | :--- | :--- |
| **Mean = Median = Mode** | **Symmetric** | Normal Distribution |
| **Mode < Median < Mean** | **Right-Skewed** (Positive) | Income, Reaction Times (Long tail on right pulls Mean up) |
| **Mean < Median < Mode** | **Left-Skewed** (Negative) | Age at death (Long tail on left pulls Mean down) |

---

### What's the difference between dependence and correlation?

**1. Correlation (Pearson Correlation, $\rho$)**:
*   Measures statistical **Linear relationships** only.
*   Range: $[-1, 1]$.
*   $\rho = 0$ means *no linear relationship*, but variables could still be strongly dependent in a non-linear way (e.g., $Y=X^2$ over a symmetric range has $\rho = 0$).

**2. Dependence**:
*   General concept: Knowing $X$ gives *any* information about $Y$.
*   Formally: $P(X, Y) \neq P(X)P(Y)$.
*   Captures all relationships: linear, quadratic, sinusoidal, etc.

**Key Distinction:**
*   **Independent $\implies$ Uncorrelated**. (If they are unrelated, they certainly have no linear relation).
*   **Uncorrelated $\not\implies$ Independent**. (A circle or parabola is uncorrelated but dependent).

**Measuring Association:**

| Metric | Captures | Use Case |
| :--- | :--- | :--- |
| **Pearson Correlation** | Linear only | Standard continuous data, Gaussian assumptions. |
| **Spearman Rank** | Monotonic (Linear or curved) | **Rank-based**. Robust to outliers. Non-linear but monotonic. |
| **Mutual Information** | Any dependence | Complex, non-linear relationships. Based on entropy: $I(X;Y) = H(X) - H(X|Y)$. |

---

## Part 2: Distributions

### Review common probability distributions

**Discrete Distributions:**

1.  **Bernoulli ($p$)**: Single trial, binary outcome (Success/Failure). Mean $p$, Var $p(1-p)$.
2.  **Binomial ($n, p$)**: Number of successes in $n$ independent Bernoulli trials. Mean $np$.
3.  **Poisson ($\lambda$)**: Number of rare events in a fixed interval. Unique because Mean = Variance = $\lambda$.

**Continuous Distributions:**

4.  **Uniform ($a, b$)**: "Box" shape. Every value in $[a, b]$ is equally likely.
5.  **Normal / Gaussian ($\mu, \sigma^2$)**: The "Bell Curve". Central to statistics due to the CLT. Max entropy distribution for fixed variance. 68% of data is within $1\sigma$, 95% within $2\sigma$.
6.  **Exponential ($\lambda$)**: Time *between* events in a Poisson process. **Memoryless**: $P(T>t+s|T>s) = P(T>t)$.
7.  **Beta ($\alpha, \beta$)**: Defined on $[0,1]$. Used to model *probabilities* (e.g., the probability of heads). Conjugate prior for Bernoulli/Binomial.
8.  **Gamma ($k, \theta$)**: Generalizes Exponential. Sum of $k$ exponentials. Conjugate prior for Poisson.
9.  **Student's t ($\nu$)**: Like Normal but with heavier tails. Used for small samples ($n<30$). As $\nu \to \infty$, converges to Normal.
10. **Chi-Squared ($k$)**: Sum of squared standard normals ($Z^2$). Used in variance testing and goodness-of-fit.

**Summary Table:**

| Distribution | Type | Parameters | Support | Mean | Use Case |
|--------------|------|-----------|---------|------|----------|
| **Bernoulli** | Discrete | $p$ | $\{0, 1\}$ | $p$ | Single Coin Flip |
| **Binomial** | Discrete | $n, p$ | $\{0,...,n\}$ | $np$ | Count of Heads in $n$ flips |
| **Poisson** | Discrete | $\lambda$ | $\{0,1,...\}$ | $\lambda$ | Call center arrivals per hour |
| **Uniform** | Cont. | $a, b$ | $[a, b]$ | $\frac{a+b}{2}$ | Random number generation |
| **Normal** | Cont. | $\mu, \sigma^2$ | $\mathbb{R}$ | $\mu$ | Errors, Heights, IQ |
| **Exponential** | Cont. | $\lambda$ | $[0, \infty)$ | $1/\lambda$ | Time until next bus |
| **Beta** | Cont. | $\alpha, \beta$ | $[0, 1]$ | $\frac{\alpha}{\alpha+\beta}$ | Modeling prob. of success |
| **t-dist** | Cont. | $\nu$ | $\mathbb{R}$ | $0$ | Inference with small $n$ |
| **Chi-Squared** | Cont. | $k$ | $[0, \infty)$ | $k$ | Variance tests, Independence |



---

## Part 3: Bayesian Inference

### Explain the Bayes theorem

**Bayes' Theorem:**
Provides a way to update probabilities based on new evidence.

$$ P(H|D) = \frac{P(D|H) \cdot P(H)}{P(D)} $$

*   **$P(H|D)$ (Posterior)**: Probability of Hypothesis $H$ given Data $D$. (What we want).
*   **$P(D|H)$ (Likelihood)**: Probability of observing Data $D$ if Hypothesis $H$ is true.
*   **$P(H)$ (Prior)**: Initial belief about $H$ before seeing data.
*   **$P(D)$ (Evidence)**: Total probability of the data (normalizing constant).

**Example: Medical Diagnosis (The Base Rate Fallacy)**

*   **Scenario**: A test for a rare disease (prevalence 1%) is 99% sensitive and 95% specific.
*   **Prior**: $P(\text{Disease}) = 0.01$.
*   **Sensitivity**: $P(\text{Pos}|\text{Disease}) = 0.99$.
*   **False Positive Rate**: $P(\text{Pos}|\text{Healthy}) = 0.05$ (since Specificity is 0.95).

**Question**: If you test positive, what is the probability you have the disease?

**Calculation**:
$$ P(\text{Dis}|\text{Pos}) = \frac{P(\text{Pos}|\text{Dis})P(\text{Dis})}{P(\text{Pos})} $$

1.  **Numerator**: $0.99 \times 0.01 = 0.0099$ (True Positives)
2.  **Denominator (Evidence)**:
    $$ P(\text{Pos}) = P(\text{Pos}|\text{Dis})P(\text{Dis}) + P(\text{Pos}|\text{Healthy})P(\text{Healthy}) $$
    $$ P(\text{Pos}) = (0.99 \times 0.01) + (0.05 \times 0.99) \approx 0.0099 + 0.0495 = 0.0594 $$
3.  **Result**: $0.0099 / 0.0594 \approx \mathbf{16.7\%}$

**Intuition**: The disease is so rare (low prior) that the small number of false positives from the large healthy population overwhelms the true positives.

**Bayesian vs Frequentist Approach:**

*   **Frequentist**: Parameters are **fixed constants** (but unknown). Probability refers to the limit of relative frequency over many repetitions. (e.g., "If we repeated this experiment infinite times, the true parameter would be in this interval 95% of the time").
*   **Bayesian**: Parameters are **random variables**. Probability represents a degree of belief. We start with a prior belief and update it with data to get a posterior distribution. (e.g., "There is a 95% probability the true parameter is in this interval").

---

### What is a conjugate prior?

**Definition:**
A prior distribution is **conjugate** to a likelihood function if the resulting posterior distribution is in the same probability family as the prior.

**Benefit**:
*   **Analytical Tractability**: No need for complex integration or MCMC.
*   **Sequential Updating**: The posterior from today becomes the prior for tomorrow.
*   **Interpretability**: Parameters often act as "pseudo-counts".

**Common Conjugate Pairs:**

| Likelihood (Data) | Conjugate Prior | Posterior | Update Rule (Concept) |
| :--- | :--- | :--- | :--- |
| **Bernoulli / Binomial** | **Beta**($\alpha, \beta$) | **Beta**($\alpha', \beta'$) | $\alpha' = \alpha + \text{successes}$<br>$\beta' = \beta + \text{failures}$ |
| **Poisson** | **Gamma**($k, \theta$) | **Gamma**($k', \theta'$) | $k' = k + \text{total counts}$<br>Rate updates similarly |
| **Normal** (known $\sigma^2$) | **Normal**($\mu_0, \sigma_0^2$) | **Normal**($\mu_n, \sigma_n^2$) | Precision-weighted average of prior mean and data mean |
| **Multinomial** | **Dirichlet**($\alpha$) | **Dirichlet**($\alpha'$) | Add counts to corresponding category alphas |

**Example: Beta-Binomial Update**
1.  **Prior**: You believe a coin is fair, but aren't sure. $\text{Beta}(\alpha=2, \beta=2)$. (Equivalent to seeing 1 head, 1 tail conceptually).
2.  **Data**: You flip it 10 times and get 7 Heads, 3 Tails.
3.  **Posterior**:
    *   $\alpha_{new} = \alpha_{old} + 7 = 9$
    *   $\beta_{new} = \beta_{old} + 3 = 5$
    *   Result: $\text{Beta}(9, 5)$. The distribution shifts towards $0.64$ (mean).

---

## Part 4: Statistical Testing

### Explain hypothesis testing, p-value, and t-test

**1. Hypothesis Testing Framework:**
1.  **Null Hypothesis ($H_0$)**: The default assumption (e.g., "The coin is fair", "The drug has no effect").
2.  **Alternative Hypothesis ($H_1$)**: Accusation being tested (e.g., "The coin is biased", "The drug works").
3.  **Test Statistic**: A number calculated from sample data (e.g., t-score, z-score).
4.  **p-value**: The probability of seeing data *at least as extreme* as what was observed, **assuming $H_0$ is true**.
5.  **Significance level ($\alpha$)**: Threshold for rejection (typically 0.05).

**Decision Rule:**
*   If p-value $< \alpha$: Reject $H_0$ (Results are "statistically significant").
*   If p-value $\ge \alpha$: Fail to reject $H_0$ (Not enough evidence).

**2. Tests for Means (t-test):**
Used when sample size is small ($n < 30$) or population variance is unknown.

*   **One-Sample t-test**: Compares the mean of a single group against a known mean.
    *   *Example*: Is the average height of this class different from the national average (170cm)?
*   **Two-Sample t-test (Independent)**: Compares the means of two independent groups.
    *   *Example*: Do users on Design A spend more time than users on Design B?
*   **Paired t-test**: Compares means from the same group at different times (or matched pairs).
    *   *Example*: Weight before vs. Weight after for the *same* participants.

**3. Errors & Power:**
*   **Type I Error ($\alpha$)**: False Positive. Rejecting $H_0$ when it is actually true. (Convicting an innocent person).
*   **Type II Error ($\beta$)**: False Negative. Failing to reject $H_0$ when it is false. (Letting a guilty person go free).
*   **Power ($1 - \beta$)**: Probability of correctly rejecting $H_0$ when it is false. (Catching the guilty person).
    *   Power increases with: Larger sample size, larger effect size, higher $\alpha$.

---

### Explain the Central Limit Theorem (CLT)

**Statement:**
The sum (or average) of many independent, identically distributed (i.i.d.) random variables tends toward a **Normal Distribution**, regardless of the original distribution of the variables.

**Mathematical Form:**
If $X_1, ..., X_n$ are i.i.d. with mean $\mu$ and variance $\sigma^2$:
$$ \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \to \mathcal{N}(0, 1) \quad \text{as } n \to \infty $$

**Why is this magical?**
*   The original data could be Uniform, Exponential, Poisson, or completely weird/skewed.
*   But if you take samples of size $n$ (say, $n=30$) and calculate the mean, and repeat this many times, the **distribution of those means** will be a Bell Curve.

**Implications:**
1.  **Normality Assumption**: Justifies using methods that assume normality (like t-tests) even if the raw data isn't normal, provided sample size is large.
2.  **Confidence Intervals**: Allows us to construct error bars around estimates easily ($\bar{x} \pm 1.96 \cdot SE$).

**Conditions:**
1.  **Independence**: Samples must be independent.
2.  **Identically Distributed**: Drawn from same population.
3.  **Finite Variance**: Distributions with infinite variance (like Cauchy) do not converge to Normal.

---

## Part 5: Bias-Variance Tradeoff

### Explain the bias-variance tradeoff

**Decomposition of Expected Test Error:**
$$ E[\text{Error}] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error} $$

**1. Definitions:**

*   **Bias (Underfitting)**: Error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs.
    *   *Symptom*: High error on both training and test data (Underfitting).
    *   *Example*: Linear Regression on a quadratic dataset.

*   **Variance (Overfitting)**: Error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data rather than the intended outputs.
    *   *Symptom*: Low training error but high test error (Overfitting).
    *   *Example*: A Decision Tree with unlimited depth (memorizes data).

*   **Irreducible Error**: Noise inherent in the data itself. No model can reduce this.

**2. Model Complexity vs. Error:**

| Model Complexity | Bias | Variance | Total Error | State |
| :--- | :--- | :--- | :--- | :--- |
| **Low** (Simple) | High | Low | High | **Underfitting** |
| **Optimal** | Medium | Medium | **Minimum** | **Generalization** |
| **High** (Complex) | Low | High | High | **Overfitting** |

**3. Diagnostics (Learning Curves):**
*   **High Bias**: Training and Validation loss are both high and close together.
    *   *Fix*: Increase model complexity (add features, deeper method).
*   **High Variance**: Low Training loss, High Validation loss (large gap).
    *   *Fix*: Add more data, Regularization (L1/L2, Dropout), Ensemble methods.

---

## Part 6: Information Theory & Divergence

### Explain KL Divergence (Relative Entropy)

**Definition:**
Kullback-Leibler (KL) Divergence measures how one probability distribution $Q$ differs from a second, reference probability distribution $P$. Only defined if $P(x)=0 \implies Q(x)=0$.

$$D_{KL}(P || Q) = \sum_{x} P(x) \log \left( \frac{P(x)}{Q(x)} \right)$$

**Key Properties:**
1.  **Non-negative**: $D_{KL}(P || Q) \ge 0$.
2.  **Not Symmetric**: $D_{KL}(P || Q) \neq D_{KL}(Q || P)$. (It matters which one is the "true" distribution).
3.  **Zero**: $D_{KL}(P || Q) = 0$ if and only if $P = Q$.

**Intuition/Meaning:**
*   **Information Code**: The expected number of *extra* bits required to code samples from $P$ using a code optimized for $Q$.
*   **"Surprise"**: Expectation of the logarithmic difference between the probabilities.

**Usage in Machine Learning:**
1.  **Variational Autoencoders (VAEs)**: Regularizes the latent space. We minimize $D_{KL}(q(z|x) || p(z))$, forcing the learned posterior to be close to a unit Gaussian prior.
2.  **Classification (Cross-Entropy)**: Minimizing Cross-Entropy is equivalent to minimizing KL Divergence between true labels and predicted probs.
    *   $CE(P, Q) = H(P) + D_{KL}(P || Q)$. Since $H(P)$ is constant for fixed labels, minimizing $CE$ minimizes $D_{KL}$.
3.  **Reinforcement Learning (PPO, TRPO)**: Used as a constraint (trust region) to prevent the policy from changing too drastically between updates ($D_{KL}(\pi_{old} || \pi_{new}) < \delta$).


**Links to other Concepts:**
*   **[[04_Loss_Functions#Probabilistic (KL Divergence, ELBO)]]**: For its usage in loss functions.
*   **[[04_Loss_Functions#Classification (Cross-Entropy, Focal, Hinge, Label Smoothing)]]**: Connection to Cross-Entropy.
*   **[[08_Advanced_Deep_Learning#Part 2: Generative Models (VAE, GAN, Diffusion)]]**: Usage in VAEs.
*   **[[02_Probability_Statistics#Explain the Bayes theorem]]**: Underlying probabilistic inference.

---

This covers the comprehensive Probability & Statistics content for ML interviews. The file includes detailed explanations, code examples, and visualizations for all key statistical concepts needed for machine learning.
