# System Design for Machine Learning - Interview Q&A

Comprehensive breakdown of designing scalable ML systems, from data ingestion to model serving and monitoring.

---

## Table of Contents

- [[#Part 1: ML System Design Framework]]
  - [[#What are the key components of an ML system?]]
  - [[#How do you approach an ML system design interview? (The 5-Step Framework)]]
  - [[#Trade-offs: Batch vs Real-time Prediction?]]
  - [[#Trade-offs: Batch vs Streaming Data Ingestion?]]
- [[#Part 2: Model Serving & Deployment]]
  - [[#What are the common model serving patterns?]]
  - [[#How to handle high throughput/low latency serving?]]
  - [[#What is a Feature Store and why do we need it?]]
  - [[#Explain Shadow Mode vs Canary Deployment vs A/B Testing]]
- [[#Part 3: Monitoring & Maintenance]]
  - [[#What is training-serving skew?]]
  - [[#How to detect Data Drift and Concept Drift?]]
  - [[#What metrics should be monitored in production?]]
  - [[#How to handle retraining loops?]]
- [[#Part 4: Case Studies]]
  - [[#Design a Recommendation System (e.g., YouTube/Netflix)]]
  - [[#Design a Search Ranking System]]
  - [[#Design an Ad Click Prediction System]]

---

## Part 1: ML System Design Framework

### What are the key components of an ML system?

1.  **Data Ingestion**: Streaming (Kafka/Kinesis) vs Batch (Airflow / Data Warehouses).
2.  **Data Processing & Validation**: Cleaning, schema validation, feature engineering.
3.  **Feature Store**: Central repository for features (Online for serving, Offline for training).
4.  **Model Training**: Distributed training, hyperparameter tuning, experiment tracking.
5.  **Model Evaluation**: Offline metrics (AUC, RMSE) vs Online metrics (CTR, Conversion).
6.  **Model Serving**: REST/gRPC APIs, batch inference jobs, edge deployment.
7.  **Monitoring**: Drift detection, latency, error rates, business metrics.
8.  **Orchestration**: Managing the DAG of dependencies (Airflow, Kubeflow).

### How do you approach an ML system design interview? (The 5-Step Framework)

1.  **Clarify Requirements**:
    *   **Goal**: What are we maximizing? (CTR, Watch time, Revenue?)
    *   **Constraints**: Latency (<100ms?), Throughput (10k QPS?), Budget?
    *   **Scale**: Users, Items, Data volume?

2.  **Data Engineering**:
    *   What data is available? (User profile, Interaction logs, Item metadata)
    *   How is it labeled? (Implicit vs Explicit feedback)
    *   Feature Engineering: Categorical (Embedding), Numerical (Scaling), Temporal.

3.  **Model Selection**:
    *   Baseline vs Advanced. (Logistic Regression -> Two-Tower NN -> Deep Interactions)
    *   Loss functions (LogLoss, Triplet Loss).
    *   Why this model? (Handling sparsity, non-linearity).

4.  **Training & Evaluation**:
    *   Train/Test split (Time-based splitting!).
    *   Offline metrics (Recall@K, MAP).
    *   Online evaluation (A/B testing).

5.  **Serving & Architecture**:
    *   Architecture diagram (Data flow).
    *   Caching strategies (Redis, CDN).
    *   Handling potential bottlenecks.

### Trade-offs: Batch vs Real-time Prediction?

**Batch Prediction (Offline)**:
*   **How**: Pre-compute predictions for all users daily/hourly and store in DB (Key-Value store).
*   **Pros**: Low latency serving (just a lookup), high throughput, simpler infra.
*   **Cons**: Predictions can be stale, wasted compute for inactive users, can't react to immediate context.
*   **Use case**: Churn prediction, Weekly recommendations, Email marketing.

**Real-time Prediction (Online)**:
*   **How**: Compute prediction on-demand when request arrives.
*   **Pros**: Fresh predictions (uses current session data), handles long-tail/new items immediately.
*   **Cons**: Higher latency constraint, complex infra, harder to scale.
*   **Use case**: Search ranking, Fraud detection, High-frequency trading.

### Trade-offs: Batch vs Streaming Data Ingestion?

*   **Batch (e.g., Airflow + Spark)**: Robust, simpler to manage, easier to re-process historical data. Good for training data generation.
*   **Streaming (e.g., Kafka + Flink)**: Low latency data availability. Critical for "Time-travel" features (e.g., # of clicks in last 5 mins). More complex to maintain (watermarking, late data).

---

## Part 2: Model Serving & Deployment

### What are the common model serving patterns?

1.  **Model-as-a-Service (Microservice)**: Model wrapped in Flask/FastAPI container (Docker). Scaled via Kubernetes.
2.  **Embedded Model**: Model library loaded directly into the application process. (Fastest, but coupling issues).
3.  **Database-Integrated**: Database calls model (e.g., BigQuery ML). Good for batch analytics.

### How to handle high throughput/low latency serving?

1.  **Serialization**: Use efficient formats like Protocol Buffers (gRPC) instead of JSON.
2.  **Batching**: Group incoming requests into micro-batches to utilize GPU parallelism effectively.
3.  **Caching**: Cache common predictions (e.g., "Trending Items" for anonymous users).
4.  **Async**: Decouple request processing if immediate response isn't strictly blocking.
5.  **Quantization/Pruning**: Compress model to run faster (FP16/INT8).
6.  **Hardware**: Use specialized hardware (TensorRT on Nvidia GPUs, TPUs, Inferentia).

### What is a Feature Store and why do we need it?

A **Feature Store** (e.g., Feast, Tecton) solves the **Training-Serving Skew** problem by ensuring consistency between offline training and online serving.

*   **Offline Store** (e.g., S3/Parquet): Stores months/years of historical feature values for model training.
*   **Online Store** (e.g., Redis/DynamoDB): Stores the *latest* feature values for low-latency serving.
*   **Point-in-Time Correctness**: Ensures that when generating training data, we use the feature values *as they were* at the time of the event (preventing data leakage).

### Explain Shadow Mode vs Canary Deployment vs A/B Testing

*   **Shadow Mode**: Deploy new model alongside old one. New model makes predictions but they are *ignored*. Used to verify latency/errors without affecting users.
*   **Canary Deployment**: Roll out new model to a small % (1-5%) of users. Monitor metrics. If stable, ramp up to 100%. Safety mechanism.
*   **A/B Testing**: Randomized experiment. Group A sees Model v1, Group B sees Model v2. Statistical test to prove v2 is significantly better on business metrics.

---

## Part 3: Monitoring & Maintenance

### What is training-serving skew?

When performance in production is worse than offline results due to differences between training time and serving time.
*   **Schema Skew**: Input data format changes.
*   **Data Skew**: Distribution of production data differs from training data.
*   **Logic Skew**: Difference in feature engineering code (e.g., Python for training vs Java for serving).

### How to detect Data Drift and Concept Drift?

*   **Data Drift (Covariate Shift - P(X) changes)**: The distribution of input features changes. (e.g., Users get younger). Detected using statistical tests like **KS-test**, **KL-divergence**, or **PSI (Population Stability Index)** between training and serving/windowed data.
*   **Concept Drift (P(Y|X) changes)**: The relationship between input and target changes. (e.g., "Corona" beer searches suddenly mean virus, not drink). Detected by monitoring prediction accuracy/loss over time (requires delayed labels).

### What metrics should be monitored in production?

1.  **Service Metrics**: Latency (p50, p99), Throughput (QPS), Error Rate, CPU/Memory/GPU utilization.
2.  **Model Metrics**: Prediction distribution (Did we stop predicting class A?), Feature distribution (Drift).
3.  **Business Metrics**: CTR, Conversion Rate, Revenue. (Ultimate truth).

### How to handle retraining loops?

*   **Scheduled**: Retrain every night/week. Simple, good for stable domains.
*   **Trigger-based**: Retrain when `accuracy < threshold` or `drift > threshold`. Active maintenance.
*   **Online Learning**: Continuous updates (e.g., Bandits). High complexity/risk.

---

## Part 4: Case Studies

### Design a Recommendation System (e.g., YouTube/Netflix)

1.  **Architecture**: **Funnel Approach**.
    *   **Retrieval (Candidate Generation)**: Fast, coarse selection. Selects 100s from Millions.
        *   Methods: Collaborative Filtering (Matrix Factorization), Two-Tower Networks (Dot product search via FAISS/ScaNN).
        *   Objective: High Recall.
    *   **Ranking**: Slower, precise scoring. Scores the 100 candidates.
        *   Methods: Deep Learning (DCN, Wide & Deep), GBDT (XGBoost).
        *   Features: Dense interactions, context, user history.
        *   Objective: High Precision/Ranking Metric (NDCG).
    *   **Re-ranking**: Business rules.
        *   Diversity, freshness, removing bad content.

2.  **Key Challenges**:
    *   **Cold Start**: New users/items. (Use content-based features, bandits).
    *   **Feedback Loop**: Model bias reinforcing itself (Exploration vs Exploitation).

### Design a Search Ranking System

1.  **Inverted Index**: To quickly find documents containing query terms.
2.  **Signals**:
    *   **Query-dependent**: BM25, Semantic similarity (BERT embedding).
    *   **Query-independent**: PageRank, Quality score, Freshness.
3.  **Learning to Rank (LTR)**:
    *   Pointwise (Is this relevant?), Pairwise (Is A > B?), Listwise (Optimize entire list order).

### Design an Ad Click Prediction System

1.  **Scale**: Billions of events, massive sparse features (User ID, Ad ID).
2.  **Model**: Logistic Regression (baseline), FM (Factorization Machines), DeepFM.
3.  **Optimization**: **Calibration** is critical (Predicted probability must match real probability for bidding).
4.  **Data**: Highly imbalanced (Clicks are rare). Downsample negatives or use Focal Loss.
