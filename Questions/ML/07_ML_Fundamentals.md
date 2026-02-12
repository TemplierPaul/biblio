# Machine Learning Fundamentals - Interview Q&A

Comprehensive coverage of classical machine learning algorithms, evaluation metrics, loss functions, optimizers, and model debugging strategies.

---

## Part 1: Classification & Regression Models

### What is the difference between generative and non-generative (discriminative) models?

**Discriminative Models** learn the decision boundary directly by modeling P(y|x) - the probability of the label given the features.
- **Goal**: Separate classes or predict outputs directly
- **Examples**: Logistic Regression, SVM, Neural Networks, BERT
- **Advantages**: Usually better performance for classification tasks, require less data
- **Question answered**: "What is y given x?"

**Generative Models** learn the joint distribution P(x, y) = P(y)P(x|y), modeling how the data is generated.
- **Goal**: Understand the underlying data distribution
- **Examples**: Naive Bayes, Gaussian Mixture Models, VAEs, GANs, GPT
- **Advantages**: Can generate new samples, handle missing data, provide probability estimates
- **Question answered**: "How is the data generated?" and "What is the probability of this data?"

**Why BERT is not generative:**
BERT is trained with masked language modeling (predicting masked tokens given context) but uses bidirectional attention, making it unsuitable for sequential generation. It learns P(word|context) but cannot naturally generate sequences left-to-right like GPT.

**Code Example - Discriminative vs Generative:**
```python
# Discriminative: Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)  # Direct classification

# Generative: Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
# Can also compute P(X|y) and generate samples
```

---

### Explain Support Vector Machines (SVM). What are the fundamentals?

**Core Concept:**
SVM finds the optimal hyperplane that maximizes the margin between classes. The margin is the distance between the hyperplane and the nearest data points (support vectors).

**Key Components:**

1. **Hyperplane**: Decision boundary defined by w·x + b = 0
2. **Support Vectors**: Data points closest to the hyperplane that define the margin
3. **Margin**: Distance from hyperplane to nearest points (2/||w||)
4. **Kernel Trick**: Map data to higher dimensions for non-linear separation

**Mathematical Formulation:**

Minimize: (1/2)||w||² + C∑ξᵢ

Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0

where:
- w: weight vector (defines hyperplane)
- b: bias
- C: regularization parameter (tradeoff between margin and misclassification)
- ξᵢ: slack variables (allow some misclassification)

**Common Kernels:**

```python
# Linear kernel: K(x, x') = x·x'
# Good for linearly separable data

# RBF (Gaussian) kernel: K(x, x') = exp(-γ||x - x'||²)
# Most popular, works well for non-linear data
from sklearn.svm import SVC

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)

# RBF kernel
svm_rbf = SVC(kernel='rbf', gamma='scale', C=1.0)
svm_rbf.fit(X_train, y_train)

# Polynomial kernel: K(x, x') = (x·x' + c)^d
svm_poly = SVC(kernel='poly', degree=3, C=1.0)
svm_poly.fit(X_train, y_train)
```

**Advantages:**
- Effective in high-dimensional spaces
- Memory efficient (only stores support vectors)
- Versatile (different kernel functions)
- Works well with clear margin of separation

**Disadvantages:**
- Slow for large datasets (O(n²) to O(n³) complexity)
- Sensitive to feature scaling
- Requires careful kernel selection and parameter tuning
- Doesn't provide probability estimates directly

**When to use SVM:**
- Small to medium datasets
- High-dimensional data (text classification, bioinformatics)
- Clear separation between classes
- Need for robust classifier with good generalization

---

### Explain k-Nearest Neighbors (k-NN). What is the concept?

**Core Concept:**
k-NN is a non-parametric, instance-based learning algorithm that classifies new data points based on the majority vote of its k nearest neighbors in the feature space.

**Algorithm Steps:**

1. Choose k (number of neighbors)
2. Calculate distance between query point and all training points
3. Sort distances and select k nearest neighbors
4. Classification: majority vote; Regression: average of k neighbors

**Distance Metrics:**

```python
import numpy as np

# Euclidean distance (most common)
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Manhattan distance
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

# Minkowski distance (generalization)
def minkowski_distance(x1, x2, p):
    return np.sum(np.abs(x1 - x2)**p)**(1/p)

# Cosine similarity (for text/high-dim)
def cosine_similarity(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
```

**Implementation Example:**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Scale features (important for k-NN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train k-NN classifier
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',  # 'uniform' or 'distance'
    metric='euclidean',
    algorithm='auto'  # 'ball_tree', 'kd_tree', 'brute'
)
knn.fit(X_train_scaled, y_train)

# Predict
predictions = knn.predict(X_test_scaled)
probabilities = knn.predict_proba(X_test_scaled)
```

**Choosing k:**

```python
# Cross-validation to find optimal k
from sklearn.model_selection import cross_val_score

k_values = range(1, 31)
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    scores.append(score.mean())

# Plot k vs accuracy
import matplotlib.pyplot as plt
plt.plot(k_values, scores)
plt.xlabel('k')
plt.ylabel('Cross-validated accuracy')
plt.show()

# Rule of thumb: k = sqrt(n), where n is number of samples
# Use odd k for binary classification to avoid ties
```

**Advantages:**
- Simple and intuitive
- No training phase (lazy learning)
- Naturally handles multi-class problems
- Non-parametric (no assumptions about data distribution)
- Can learn complex decision boundaries

**Disadvantages:**
- Slow prediction for large datasets (O(nd) per query)
- Memory intensive (stores all training data)
- Sensitive to irrelevant features and feature scaling
- Curse of dimensionality (performance degrades in high dimensions)
- Need to choose k and distance metric

**When to use k-NN:**
- Small to medium datasets
- Low-dimensional data
- Non-linear decision boundaries
- No need for model interpretability
- Baseline model for comparison

**Optimization Techniques:**

```python
# Use efficient data structures
knn_kdtree = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='kd_tree',  # Fast for low-dim
    leaf_size=30
)

knn_balltree = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='ball_tree',  # Better for high-dim
    leaf_size=30
)

# Dimensionality reduction before k-NN
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

knn.fit(X_train_pca, y_train)
```

---

## Part 2: Evaluation Metrics

### Explain Precision, Recall, F1-Score, and ROC-AUC

**Confusion Matrix Foundation:**

```
                Predicted Positive    Predicted Negative
Actual Positive      TP                    FN
Actual Negative      FP                    TN
```

**Precision:**
- **Definition**: Of all predicted positive samples, how many are actually positive?
- **Formula**: Precision = TP / (TP + FP)
- **Use Case**: When false positives are costly (e.g., spam detection - don't mark important emails as spam)

**Recall (Sensitivity, True Positive Rate):**
- **Definition**: Of all actual positive samples, how many did we correctly identify?
- **Formula**: Recall = TP / (TP + FN)
- **Use Case**: When false negatives are costly (e.g., disease detection - don't miss sick patients)

**F1-Score:**
- **Definition**: Harmonic mean of precision and recall
- **Formula**: F1 = 2 × (Precision × Recall) / (Precision + Recall)
- **Use Case**: Balance between precision and recall, especially with imbalanced classes

**ROC-AUC (Receiver Operating Characteristic - Area Under Curve):**
- **ROC Curve**: Plots True Positive Rate (Recall) vs False Positive Rate at various thresholds
- **AUC**: Area under ROC curve (0.5 = random, 1.0 = perfect)
- **Interpretation**: Probability that model ranks random positive example higher than random negative
- **Use Case**: Evaluate model performance across all classification thresholds

**Implementation:**

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, average_precision_score
)
import matplotlib.pyplot as plt

# Binary classification
y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
y_pred = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
y_proba = [0.1, 0.9, 0.8, 0.3, 0.4, 0.85, 0.2, 0.6, 0.95, 0.15]

# Basic metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.3f}")  # 0.750
print(f"Recall: {recall:.3f}")        # 0.750
print(f"F1-Score: {f1:.3f}")          # 0.750

# ROC-AUC (requires probability scores)
auc = roc_auc_score(y_true, y_proba)
print(f"ROC-AUC: {auc:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Full classification report
print(classification_report(y_true, y_pred))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

**Multi-class Metrics:**

```python
# Multi-class classification
y_true_multi = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred_multi = [0, 2, 1, 0, 1, 2, 0, 1, 1]

# Macro average: unweighted mean (treats all classes equally)
precision_macro = precision_score(y_true_multi, y_pred_multi, average='macro')

# Weighted average: weighted by support (number of true instances)
precision_weighted = precision_score(y_true_multi, y_pred_multi, average='weighted')

# Micro average: aggregate contributions of all classes
precision_micro = precision_score(y_true_multi, y_pred_multi, average='micro')

print(f"Macro Precision: {precision_macro:.3f}")
print(f"Weighted Precision: {precision_weighted:.3f}")
print(f"Micro Precision: {precision_micro:.3f}")

# Multi-class ROC-AUC (one-vs-rest)
from sklearn.preprocessing import label_binarize
y_true_bin = label_binarize(y_true_multi, classes=[0, 1, 2])
y_score = [[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], ...]  # probability scores

auc_ovr = roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr')
```

**Precision-Recall Tradeoff:**

```python
# Adjust threshold to balance precision and recall
from sklearn.metrics import precision_recall_curve

precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_proba)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Find optimal threshold for F1
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold for F1: {optimal_threshold:.3f}")
```

**Which Metric to Use:**

| Scenario | Recommended Metric |
|----------|-------------------|
| Balanced classes | Accuracy, F1-Score |
| Imbalanced classes | Precision, Recall, F1, ROC-AUC |
| Cost of FP >> FN | Precision |
| Cost of FN >> FP | Recall |
| Need ranking quality | ROC-AUC, PR-AUC |
| Medical diagnosis | Recall (don't miss positives) |
| Spam detection | Precision (don't block good emails) |
| Fraud detection | Balance with F1, high Recall |

**Regression Metrics:**

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)

y_true_reg = [3.0, -0.5, 2.0, 7.0]
y_pred_reg = [2.5, 0.0, 2.0, 8.0]

# Mean Squared Error
mse = mean_squared_error(y_true_reg, y_pred_reg)
rmse = np.sqrt(mse)

# Mean Absolute Error
mae = mean_absolute_error(y_true_reg, y_pred_reg)

# R² Score (coefficient of determination)
r2 = r2_score(y_true_reg, y_pred_reg)

# Mean Absolute Percentage Error
mape = mean_absolute_percentage_error(y_true_reg, y_pred_reg)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")
print(f"MAPE: {mape:.3f}")
```

---

### What are BLEU, ROUGE, and Perplexity metrics used in NLP?

**BLEU (Bilingual Evaluation Understudy):**

**Purpose**: Evaluate machine translation and text generation quality
**How it works**: Measures n-gram overlap between generated and reference texts

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

reference = [['the', 'cat', 'is', 'on', 'the', 'mat']]
candidate = ['the', 'cat', 'is', 'on', 'the', 'rug']

# BLEU-1 (unigram), BLEU-2 (bigram), etc.
bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

print(f"BLEU-1: {bleu1:.3f}")
print(f"BLEU-2: {bleu2:.3f}")
print(f"BLEU-4: {bleu4:.3f}")
```

**Formula**:
BLEU = BP × exp(∑(wₙ × log(pₙ)))

where:
- pₙ: modified n-gram precision
- wₙ: weights (typically uniform)
- BP: brevity penalty (penalizes short outputs)

**Advantages**:
- Language-independent
- Fast to compute
- Correlates well with human judgment

**Limitations**:
- Doesn't capture semantic meaning
- Favors shorter sentences
- Requires reference translations

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**

**Purpose**: Evaluate summarization and text generation (focuses on recall)

```python
from rouge import Rouge

reference = "the cat is on the mat"
candidate = "the cat is on the rug"

rouge = Rouge()
scores = rouge.get_scores(candidate, reference)

print(scores)
# Output: [{'rouge-1': {'r': 0.83, 'p': 0.83, 'f': 0.83},
#           'rouge-2': {'r': 0.60, 'p': 0.60, 'f': 0.60},
#           'rouge-l': {'r': 0.83, 'p': 0.83, 'f': 0.83}}]
```

**Variants**:
- **ROUGE-N**: N-gram overlap (like BLEU but recall-focused)
- **ROUGE-L**: Longest Common Subsequence
- **ROUGE-S**: Skip-bigram matching

**When to use ROUGE vs BLEU**:
- BLEU: Machine translation (precision matters)
- ROUGE: Summarization (recall matters)

**Perplexity:**

**Purpose**: Measure language model quality (lower is better)
**Definition**: Inverse probability of the test set, normalized by sequence length

```python
import torch
import torch.nn.functional as F

def calculate_perplexity(model, text_ids, tokenizer):
    """
    Calculate perplexity for a language model
    """
    model.eval()
    with torch.no_grad():
        outputs = model(text_ids, labels=text_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

# Example with GPT-2
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors='pt')

perplexity = calculate_perplexity(model, inputs['input_ids'], tokenizer)
print(f"Perplexity: {perplexity:.2f}")
```

**Formula**:
Perplexity = exp(-1/N × ∑ log P(wᵢ|w₁...wᵢ₋₁))

where N is the number of tokens

**Interpretation**:
- Perplexity of 10: Model is "perplexed" between ~10 choices per token
- Lower perplexity = better model
- GPT-2: ~30-50 on common benchmarks
- Human-level: ~12 on some tasks

**Bits Per Character (BPC):**

```python
def calculate_bpc(loss):
    """
    Convert cross-entropy loss to bits per character
    """
    return loss / np.log(2)

# BPC is commonly used for character-level models
# Lower is better, typically 1.0-2.0 for good models
```

**Comparison Table:**

| Metric | Task | Focus | Range | Better |
|--------|------|-------|-------|--------|
| BLEU | Translation | Precision | 0-1 | Higher |
| ROUGE | Summarization | Recall | 0-1 | Higher |
| Perplexity | LM quality | Probability | 1-∞ | Lower |
| BPC | Char LM | Bits | 0-∞ | Lower |

---

## Part 3: Loss Functions

### Name different types of loss functions and their assumptions

**Classification Loss Functions:**

**1. Binary Cross-Entropy (Log Loss)**

```python
import torch
import torch.nn as nn

# Binary classification
bce_loss = nn.BCELoss()
bce_with_logits = nn.BCEWithLogitsLoss()  # More numerically stable

# Example
y_true = torch.tensor([1., 0., 1., 1., 0.])
y_pred = torch.tensor([0.9, 0.1, 0.8, 0.6, 0.2])

loss = bce_loss(y_pred, y_true)
print(f"BCE Loss: {loss.item():.4f}")
```

**Formula**: L = -1/N × ∑[yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

**Assumptions**:
- Binary labels (0 or 1)
- Output is probability (0 to 1)
- Penalizes confident wrong predictions heavily

**2. Categorical Cross-Entropy (Multi-class)**

```python
# Multi-class classification
ce_loss = nn.CrossEntropyLoss()  # Combines LogSoftmax + NLLLoss

# Example (no need to apply softmax, it's built-in)
logits = torch.tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]])
targets = torch.tensor([0, 1])  # Class indices

loss = ce_loss(logits, targets)
print(f"CE Loss: {loss.item():.4f}")
```

**Formula**: L = -∑ yᵢ log(ŷᵢ)

**Assumptions**:
- Mutually exclusive classes
- Output is probability distribution (softmax)
- One-hot encoded targets

**3. Focal Loss (for Imbalanced Data)**

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return focal_loss.mean()

# Focuses on hard examples, down-weights easy ones
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
```

**Formula**: FL = -α(1-p)^γ × log(p)

**Assumptions**:
- Handles class imbalance
- γ > 0 focuses on hard examples
- α balances positive/negative classes

**Regression Loss Functions:**

**1. Mean Squared Error (L2 Loss)**

```python
mse_loss = nn.MSELoss()

y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_pred = torch.tensor([1.1, 2.2, 2.8, 4.1])

loss = mse_loss(y_pred, y_true)
print(f"MSE Loss: {loss.item():.4f}")
```

**Formula**: L = 1/N × ∑(yᵢ - ŷᵢ)²

**Assumptions**:
- Gaussian noise
- Outliers heavily penalized (squared term)
- Continuous targets

**2. Mean Absolute Error (L1 Loss)**

```python
mae_loss = nn.L1Loss()

loss = mae_loss(y_pred, y_true)
print(f"MAE Loss: {loss.item():.4f}")
```

**Formula**: L = 1/N × ∑|yᵢ - ŷᵢ|

**Assumptions**:
- Laplacian noise
- More robust to outliers than MSE
- Constant gradient

**3. Huber Loss (Smooth L1)**

```python
huber_loss = nn.SmoothL1Loss(beta=1.0)

loss = huber_loss(y_pred, y_true)
print(f"Huber Loss: {loss.item():.4f}")
```

**Formula**:
```
L(x) = { 0.5 × x²        if |x| ≤ δ
       { δ × (|x| - 0.5δ) if |x| > δ
```

**Assumptions**:
- Quadratic for small errors (smooth gradient)
- Linear for large errors (robust to outliers)
- Best of both MSE and MAE

**Advanced Loss Functions:**

**4. Contrastive Loss (Metric Learning)**

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss
```

**Use**: Siamese networks, face verification

**5. Triplet Loss (Metric Learning)**

```python
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

anchor = torch.randn(32, 128)
positive = torch.randn(32, 128)
negative = torch.randn(32, 128)

loss = triplet_loss(anchor, positive, negative)
```

**Formula**: L = max(||a - p||² - ||a - n||² + margin, 0)

**Use**: Face recognition, embedding learning

**6. KL Divergence (Distribution Matching)**

```python
kl_loss = nn.KLDivLoss(reduction='batchmean')

# Input should be log-probabilities
input_log_probs = F.log_softmax(torch.randn(3, 5), dim=1)
target_probs = F.softmax(torch.randn(3, 5), dim=1)

loss = kl_loss(input_log_probs, target_probs)
```

**Formula**: KL(P||Q) = ∑ P(x) log(P(x)/Q(x))

**Use**: VAEs, knowledge distillation, distribution alignment

**7. Hinge Loss (SVM)**

```python
hinge_loss = nn.HingeEmbeddingLoss()

# Binary classification with labels {-1, +1}
predictions = torch.randn(10)
targets = torch.tensor([1, -1, 1, 1, -1, 1, -1, -1, 1, -1])

loss = hinge_loss(predictions, targets)
```

**Formula**: L = max(0, 1 - y × ŷ)

**Use**: SVMs, max-margin classifiers

**Loss Function Selection Guide:**

| Task | Loss Function | Why |
|------|---------------|-----|
| Binary Classification | BCE | Probabilistic, well-calibrated |
| Multi-class | Cross-Entropy | Standard, works well |
| Imbalanced Classes | Focal Loss | Focuses on hard examples |
| Regression | MSE | Smooth gradients, penalizes large errors |
| Robust Regression | Huber/MAE | Handles outliers |
| Metric Learning | Triplet/Contrastive | Learns embeddings |
| Generative Models | KL Divergence | Matches distributions |
| Ranking | Hinge Loss | Max-margin principle |

**Custom Loss Example:**

```python
class CustomLoss(nn.Module):
    def __init__(self, weight_mse=0.7, weight_mae=0.3):
        super().__init__()
        self.weight_mse = weight_mse
        self.weight_mae = weight_mae
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        loss_mae = self.mae(pred, target)
        return self.weight_mse * loss_mse + self.weight_mae * loss_mae

# Combine multiple objectives
custom_loss = CustomLoss(weight_mse=0.7, weight_mae=0.3)
```

---

## Part 4: Optimizers

### Name different types of optimizers and their intuitions (e.g., Adam, RMSProp)

**1. Stochastic Gradient Descent (SGD)**

```python
import torch.optim as optim

# Basic SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SGD with momentum
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True  # Nesterov momentum
)
```

**Update Rule**:
```
θ = θ - η × ∇L(θ)
```

**With Momentum**:
```
v_t = β × v_{t-1} + ∇L(θ)
θ = θ - η × v_t
```

**Intuition**:
- Basic: Follow negative gradient direction
- Momentum: Accumulate velocity, smooth out oscillations
- Nesterov: "Look ahead" before computing gradient

**Pros**:
- Simple, well-understood
- Works well with momentum
- Good for convex problems

**Cons**:
- Same learning rate for all parameters
- Struggles with saddle points
- Sensitive to learning rate

**2. AdaGrad (Adaptive Gradient)**

```python
optimizer = optim.Adagrad(
    model.parameters(),
    lr=0.01,
    eps=1e-10
)
```

**Update Rule**:
```
G_t = G_{t-1} + (∇L(θ))²
θ = θ - (η / sqrt(G_t + ε)) × ∇L(θ)
```

**Intuition**:
- Adapt learning rate for each parameter
- Larger updates for infrequent parameters
- Smaller updates for frequent parameters

**Pros**:
- Good for sparse data
- No manual learning rate tuning

**Cons**:
- Learning rate decays too aggressively
- Can stop learning too early

**3. RMSProp (Root Mean Square Propagation)**

```python
optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.001,
    alpha=0.99,  # Decay rate
    eps=1e-8
)
```

**Update Rule**:
```
E[g²]_t = α × E[g²]_{t-1} + (1-α) × (∇L(θ))²
θ = θ - (η / sqrt(E[g²]_t + ε)) × ∇L(θ)
```

**Intuition**:
- Fixes AdaGrad's aggressive decay
- Uses exponential moving average of squared gradients
- Adapts learning rate per parameter

**Pros**:
- Fixes AdaGrad's decay problem
- Works well with RNNs
- Good for non-stationary objectives

**Cons**:
- Still requires manual learning rate
- Can oscillate

**4. Adam (Adaptive Moment Estimation)**

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),  # β₁, β₂
    eps=1e-8,
    weight_decay=0.0
)
```

**Update Rule**:
```
m_t = β₁ × m_{t-1} + (1-β₁) × ∇L(θ)        # 1st moment (mean)
v_t = β₂ × v_{t-1} + (1-β₂) × (∇L(θ))²     # 2nd moment (variance)

m̂_t = m_t / (1 - β₁^t)  # Bias correction
v̂_t = v_t / (1 - β₂^t)

θ = θ - η × m̂_t / (sqrt(v̂_t) + ε)
```

**Intuition**:
- Combines momentum (1st moment) and RMSProp (2nd moment)
- Bias correction for initial estimates
- Adaptive learning rates

**Pros**:
- Works well in practice
- Little hyperparameter tuning needed
- Handles sparse gradients well
- Fast convergence

**Cons**:
- Can have poor generalization vs SGD
- May not converge to optimal solution
- Memory intensive (stores two moments)

**5. AdamW (Adam with Weight Decay)**

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # Decoupled weight decay
)
```

**Key Difference from Adam**:
- Decouples weight decay from gradient update
- More effective regularization
- Better generalization

**Update Rule**:
```
θ = θ - η × (m̂_t / (sqrt(v̂_t) + ε) + λ × θ)
```

**When to use**: Transformer models, modern deep learning (now preferred over Adam)

**6. RAdam (Rectified Adam)**

```python
optimizer = optim.RAdam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999)
)
```

**Intuition**:
- Fixes Adam's warmup problem
- Automatic warmup based on variance
- More robust to learning rate

**7. Lookahead**

```python
from torch_optimizer import Lookahead

# Wrap another optimizer
base_optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
```

**Intuition**:
- Maintains two sets of weights: fast and slow
- Fast weights explore, slow weights maintain stability
- Improves convergence and generalization

**8. LAMB (Layer-wise Adaptive Moments for Batch training)**

```python
from torch_optimizer import Lamb

optimizer = Lamb(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
```

**Intuition**:
- Designed for large batch training
- Layer-wise adaptation
- Used in BERT training

**Optimizer Comparison:**

| Optimizer | Learning Rate Needed | Memory | Speed | Generalization | Best For |
|-----------|---------------------|---------|-------|----------------|----------|
| SGD | Yes | Low | Fast | Excellent | Simple problems, CV |
| SGD + Momentum | Yes | Low | Fast | Excellent | CV, proven architectures |
| AdaGrad | Less | Medium | Medium | Good | Sparse data, NLP |
| RMSProp | Yes | Medium | Fast | Good | RNNs, non-stationary |
| Adam | Less | High | Fast | Good | General purpose, RL |
| AdamW | Less | High | Fast | Better | Transformers, modern DL |
| RAdam | Less | High | Fast | Better | Unstable training |
| LAMB | Less | High | Fast | Good | Large batch training |

**Practical Guidelines:**

```python
# Computer Vision
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)

# NLP / Transformers
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# Reinforcement Learning
optimizer = optim.Adam(
    model.parameters(),
    lr=3e-4,
    eps=1e-5
)

# Large Batch Training
optimizer = Lamb(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

**Learning Rate Scheduling:**

```python
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ReduceLROnPlateau,
    OneCycleLR, CosineAnnealingWarmRestarts
)

# Step decay
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

# Reduce on plateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# One-cycle policy (recommended for SGD)
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,
    total_steps=epochs * len(train_loader)
)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(batch)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

---

## Part 5: Regularization & Overfitting

### What is the difference between L1 and L2 regularization? When would you use each?

**L2 Regularization (Ridge)**

**Formula**: Loss_total = Loss_original + λ × ∑ᵢ wᵢ²

```python
import torch.nn as nn

# In PyTorch, use weight_decay parameter
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2 regularization
)

# Or add manually to loss
l2_lambda = 0.01
l2_reg = sum(param.pow(2.0).sum() for param in model.parameters())
loss = loss_original + l2_lambda * l2_reg
```

**Effects**:
- Penalizes large weights quadratically
- Weights shrink but rarely become exactly zero
- Spreads weight across all features
- Differentiable everywhere

**When to use**:
- Have many relevant features
- Want to reduce multicollinearity
- Prefer smooth, distributed weights
- Default choice for deep learning

**L1 Regularization (Lasso)**

**Formula**: Loss_total = Loss_original + λ × ∑ᵢ |wᵢ|

```python
# L1 not directly supported in PyTorch optimizers
# Add manually to loss
l1_lambda = 0.01
l1_reg = sum(param.abs().sum() for param in model.parameters())
loss = loss_original + l1_lambda * l1_reg
```

**Effects**:
- Penalizes weights linearly
- Drives weights to exactly zero (sparsity)
- Performs feature selection
- Not differentiable at zero

**When to use**:
- Have many irrelevant features
- Want automatic feature selection
- Prefer sparse, interpretable models
- Feature engineering/selection

**Elastic Net (Combine L1 and L2)**

```python
# Combine L1 and L2
alpha = 0.01  # Overall regularization strength
l1_ratio = 0.5  # Balance between L1 and L2

l1_reg = sum(param.abs().sum() for param in model.parameters())
l2_reg = sum(param.pow(2.0).sum() for param in model.parameters())

reg_loss = alpha * (l1_ratio * l1_reg + (1 - l1_ratio) * l2_reg)
loss = loss_original + reg_loss
```

**Comparison**:

| Aspect | L1 (Lasso) | L2 (Ridge) | Elastic Net |
|--------|------------|------------|-------------|
| Penalty | ∑\|w\| | ∑w² | α(ρ∑\|w\| + (1-ρ)∑w²) |
| Sparsity | Yes | No | Moderate |
| Feature Selection | Yes | No | Yes |
| Differentiable | No (at 0) | Yes | Mixed |
| Computational | Slower | Faster | Medium |
| Multi-collinearity | Picks one | Handles well | Handles well |

**Sklearn Example**:

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# L2 (Ridge)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# L1 (Lasso)
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

# Elastic Net
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train, y_train)

# Compare coefficients
print(f"Ridge non-zero: {np.sum(ridge.coef_ != 0)}")
print(f"Lasso non-zero: {np.sum(lasso.coef_ != 0)}")
```

---

### Explain overfitting and strategies to mitigate it

**What is Overfitting?**

Model learns training data too well, including noise and outliers, leading to poor generalization on unseen data.

**Symptoms**:
- High training accuracy, low validation accuracy
- Large gap between train and validation loss
- Model memorizes rather than learns patterns

**Detection**:

```python
import matplotlib.pyplot as plt

# Plot training and validation curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Curves')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves')

plt.show()

# If val_loss increases while train_loss decreases → overfitting
```

**Mitigation Strategies:**

**1. More Training Data**

```python
# Data augmentation (for images)
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# Synthetic data generation
# Use GANs, SMOTE, or other techniques
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**2. Regularization (L1, L2)**

```python
# L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# L1 regularization
l1_lambda = 0.01
l1_reg = sum(param.abs().sum() for param in model.parameters())
loss = criterion(output, target) + l1_lambda * l1_reg
```

**3. Dropout**

```python
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(0.3)  # Drop 30% of neurons
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Only active during training
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Model automatically handles train/eval mode
model.train()  # Dropout active
model.eval()   # Dropout inactive
```

**4. Early Stopping**

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Usage
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

**5. Simpler Model Architecture**

```python
# Complex model (prone to overfitting)
class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 10)

# Simpler model (better generalization)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 10)
```

**6. Batch Normalization**

```python
class NetworkWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
```

**7. Cross-Validation**

```python
from sklearn.model_selection import KFold, cross_val_score

# K-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    X_train_fold = X[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X[val_idx]
    y_val_fold = y[val_idx]

    # Train model on this fold
    model = create_model()
    train(model, X_train_fold, y_train_fold)
    score = evaluate(model, X_val_fold, y_val_fold)
    print(f"Fold {fold+1} score: {score:.4f}")
```

**8. Ensemble Methods**

```python
# Bagging (Bootstrap Aggregating)
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)

# Random Forest (built-in bagging + feature randomness)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Boosting
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
ada = AdaBoostClassifier(n_estimators=50)
```

**Complete Overfitting Prevention Pipeline:**

```python
class AntiOverfitModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        # Simpler architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Training with regularization and early stopping
model = AntiOverfitModel(input_dim=784, hidden_dim=256, output_dim=10)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # L2 reg
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
early_stopping = EarlyStopping(patience=10)

for epoch in range(100):
    # Training with data augmentation
    train_loss = train_with_augmentation(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Early stopping
    early_stopping(val_loss)
    if early_stopping.early_stop:
        break
```

**Summary of Strategies:**

| Strategy | Effectiveness | Computational Cost | When to Use |
|----------|---------------|-------------------|-------------|
| More Data | High | High | Always preferred |
| Data Augmentation | High | Low-Medium | Images, text |
| L2 Regularization | Medium | Low | Default choice |
| L1 Regularization | Medium | Low | Feature selection |
| Dropout | High | Low | Deep networks |
| Early Stopping | High | None | Always use |
| Simpler Model | High | Negative | Small datasets |
| Batch Normalization | Medium | Low | Deep networks |
| Cross-Validation | High | High | Model selection |
| Ensembles | Very High | High | Production models |

---

## Part 6: Model Debugging & Improvement

### You have a classification model with 80% accuracy. What steps would you take to improve it?

**Systematic Debugging Approach:**

**Step 1: Diagnose the Problem**

```python
# 1. Check for class imbalance
import numpy as np
from collections import Counter

class_distribution = Counter(y_train)
print(f"Class distribution: {class_distribution}")

# Calculate imbalance ratio
majority_class = max(class_distribution.values())
minority_class = min(class_distribution.values())
imbalance_ratio = majority_class / minority_class
print(f"Imbalance ratio: {imbalance_ratio:.2f}")

# 2. Analyze confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Detailed classification report
print(classification_report(y_true, y_pred))

# 3. Check for high bias vs high variance
train_score = model.score(X_train, y_train)
val_score = model.score(X_val, y_val)

print(f"Training accuracy: {train_score:.3f}")
print(f"Validation accuracy: {val_score:.3f}")
print(f"Gap: {train_score - val_score:.3f}")

if train_score < 0.85 and val_score < 0.85:
    print("High bias (underfitting) - model too simple")
elif train_score > 0.95 and val_score < 0.85:
    print("High variance (overfitting) - model too complex")
else:
    print("Model seems balanced")
```

**Step 2: Address Class Imbalance**

```python
# Method 1: Class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# For PyTorch
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float)
)

# For sklearn
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced')

# Method 2: Resampling
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# Oversample minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Combined over and undersampling
from imblearn.combine import SMOTETomek
resampler = SMOTETomek(random_state=42)
X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)

# Method 3: Threshold tuning
from sklearn.metrics import precision_recall_curve

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Adjust predictions
y_pred_adjusted = (y_proba >= optimal_threshold).astype(int)
```

**Step 3: Feature Engineering & Selection**

```python
# 1. Feature importance analysis
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(importances.head(20))

# Select top features
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=50)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# 2. Create interaction features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_train_poly = poly.fit_transform(X_train)

# 3. Domain-specific features
# Example for text: TF-IDF, n-grams
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5
)
X_train_tfidf = tfidf.fit_transform(text_data)

# 4. Remove highly correlated features
corr_matrix = pd.DataFrame(X_train, columns=feature_names).corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop = [column for column in upper_triangle.columns
           if any(upper_triangle[column] > 0.95)]
X_train_uncorr = X_train.drop(to_drop, axis=1)
```

**Step 4: Try Different Models/Ensembles**

```python
# 1. Try multiple algorithms
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(n_estimators=100),
    'LightGBM': LGBMClassifier(n_estimators=100),
    'SVM': SVC(probability=True),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50))
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    results[name] = score
    print(f"{name}: {score:.4f}")

# 2. Ensemble methods
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Voting ensemble (soft voting for probabilities)
voting = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('gb', GradientBoostingClassifier(n_estimators=100)),
        ('xgb', XGBClassifier(n_estimators=100))
    ],
    voting='soft'
)
voting.fit(X_train, y_train)

# Stacking ensemble
stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('gb', GradientBoostingClassifier(n_estimators=100)),
        ('xgb', XGBClassifier(n_estimators=100))
    ],
    final_estimator=LogisticRegression()
)
stacking.fit(X_train, y_train)
```

**Step 5: Hyperparameter Tuning**

```python
# 1. Grid search
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# 2. Random search (faster for large param spaces)
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [10, 20, 30, 50, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)

# 3. Bayesian optimization (most efficient)
from skopt import BayesSearchCV
from skopt.space import Real, Integer

search_spaces = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(10, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 1.0)
}

bayes_search = BayesSearchCV(
    RandomForestClassifier(),
    search_spaces,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    random_state=42
)
bayes_search.fit(X_train, y_train)
```

**Step 6: Deep Learning Specific Improvements**

```python
# 1. Learning rate tuning
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

# Learning rate finder
def find_lr(model, train_loader, optimizer, criterion):
    lrs = []
    losses = []
    model.train()

    for lr in np.logspace(-6, -1, 50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch[0]), batch[1])
            loss.backward()
            optimizer.step()

            lrs.append(lr)
            losses.append(loss.item())

    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()

# Use OneCycle policy
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,
    total_steps=len(train_loader) * epochs
)

# 2. Batch size tuning
batch_sizes = [16, 32, 64, 128, 256]
for bs in batch_sizes:
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    # Train and evaluate

# 3. Architecture search
# Try different layer sizes, depths, activation functions
```

**Step 7: Error Analysis**

```python
# Analyze misclassified examples
from sklearn.metrics import confusion_matrix

# Get misclassified indices
misclassified = np.where(y_pred != y_test)[0]

# Analyze patterns
misclassified_samples = X_test[misclassified]
true_labels = y_test[misclassified]
pred_labels = y_pred[misclassified]
pred_proba = model.predict_proba(misclassified_samples)

# Look at low-confidence predictions
low_confidence = np.max(pred_proba, axis=1) < 0.6
print(f"Low confidence predictions: {np.sum(low_confidence)}")

# Analyze feature distributions for errors
import pandas as pd
error_df = pd.DataFrame(misclassified_samples, columns=feature_names)
error_df['true_label'] = true_labels
error_df['pred_label'] = pred_labels

# Statistical analysis of errors
print(error_df.describe())
```

**Complete Improvement Pipeline:**

```python
def improve_model(X_train, y_train, X_test, y_test):
    """Complete pipeline to improve model from 80% to 85%+"""

    # 1. Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # 2. Feature engineering
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    X_train_poly = poly.fit_transform(X_train_balanced)
    X_test_poly = poly.transform(X_test)

    # 3. Feature selection
    selector = SelectKBest(f_classif, k=100)
    X_train_selected = selector.fit_transform(X_train_poly, y_train_balanced)
    X_test_selected = selector.transform(X_test_poly)

    # 4. Ensemble model
    models = [
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=20)),
        ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5)),
        ('xgb', XGBClassifier(n_estimators=200, max_depth=6))
    ]

    ensemble = VotingClassifier(estimators=models, voting='soft')
    ensemble.fit(X_train_selected, y_train_balanced)

    # 5. Threshold tuning
    y_proba = ensemble.predict_proba(X_test_selected)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    y_pred = (y_proba >= optimal_threshold).astype(int)

    # 6. Evaluate
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Final accuracy: {accuracy:.4f}")
    print(f"Final F1-score: {f1:.4f}")

    return ensemble, optimal_threshold

# Run the pipeline
model, threshold = improve_model(X_train, y_train, X_test, y_test)
```

**Priority Checklist for 80% → 85%+:**

1. ✅ Check class imbalance → Use SMOTE/class weights
2. ✅ Analyze confusion matrix → Focus on misclassified classes
3. ✅ Feature engineering → Add interactions, domain features
4. ✅ Try ensemble methods → Voting/stacking usually gives 2-5% boost
5. ✅ Hyperparameter tuning → Can gain 1-3%
6. ✅ Threshold tuning → Often overlooked, can help 1-2%
7. ✅ More/better data → Most effective if possible

---

(Continued in next parts...)
