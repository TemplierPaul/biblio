# Machine Learning Fundamentals - Interview Q&A

Comprehensive coverage of evaluation metrics and model debugging strategies.

---

## Table of Contents

- [[#Part 1: Evaluation Metrics]]
  - [[#Explain Accuracy, Precision, Recall, F1-Score, and ROC-AUC]]
  - [[#What are BLEU, ROUGE, and Perplexity metrics used in NLP?]]
- [[#Part 2: Model Debugging & Improvement]]
  - [[#You have a classification model with 80% accuracy. What steps would you take to improve it?]]

---

## Part 1: Evaluation Metrics

### Explain Accuracy, Precision, Recall, F1-Score, and ROC-AUC

**Confusion Matrix Foundation:**

```
                Predicted Positive    Predicted Negative
Actual Positive      TP (True Pos)         FN (False Neg/Type II)
Actual Negative      FP (False Pos/Type I) TN (True Neg)
```

**Accuracy:**
- **Definition**: The overall correctness of the model.
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Use Case**: Balanced datasets. **Avoid** for imbalanced datasets (e.g., 99% negative class).

**Precision:**
- **Definition**: Of all predicted positive samples, how many are actually positive?
- **Formula**: Precision = TP / (TP + FP)
- **Use Case**: When false positives are costly (e.g., spam detection - don't mark important emails as spam)

**Recall (Sensitivity, True Positive Rate):**
- **Definition**: Of all actual positive samples, how many did we correctly identify?
- **Formula**: Recall = TP / (TP + FN)
- **Use Case**: When false negatives are costly (e.g., disease detection - don't miss sick patients)

**Specificity (True Negative Rate):**
- **Definition**: Of all actual negative samples, how many did we correctly identify?
- **Formula**: Specificity = TN / (TN + FP)
- **Use Case**: When false positives are costly (e.g., email filtering - don't lose good emails)

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
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, average_precision_score
)
import matplotlib.pyplot as plt

# Binary classification
y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
y_pred = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
y_proba = [0.1, 0.9, 0.8, 0.3, 0.4, 0.85, 0.2, 0.6, 0.95, 0.15]

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Specificity (TN / (TN + FP)) - not directly in sklearn, derive from CM
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp)

print(f"Accuracy: {accuracy:.3f}")    # 0.800
print(f"Precision: {precision:.3f}")   # 0.750
print(f"Recall: {recall:.3f}")         # 0.750
print(f"Specificity: {specificity:.3f}") # 0.833
print(f"F1-Score: {f1:.3f}")           # 0.750

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

---

## Part 2: Model Debugging & Improvement

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
