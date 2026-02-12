# Resources
- [ ] Deep Learning Book
- [ ] Deep dive into **MuZero's architecture.**
- [ ] https://jax-ml.github.io/scaling-book/

# Questions
## Machine Learning
- [ ] difference between generative and non-generative models (BERT is not generative?)
- [ ] **Precision**, **Recall**, **F1-Score**, and the **ROC-AUC**
- [ ] Transformer
	- [ ] KV cache
	- [ ] Explain transformers in the context of LLMs
	- [ ] What is the process of training transformers, supervised finetuning, RL involved in it?
	- [ ] Explain the difference between PPO and GRPO.
- [ ] VAE
- [ ] GAN
- [ ] diffusion model --> latent diffusion model
- [ ] What are the caveats of neural networks as universal approximators
- [ ] How to debug a poorly performing model (e.g., high bias or variance)
- [ ] Tuning hyperparameters for improved model accuracy.
- [ ] You have a classification model with 80% accuracy. What steps would you take to improve it?
	- [ ] Check for data imbalances, refine features, and experiment with ensemble methods.
- [ ] **Supervised/Unsupervised Learning:** Mastery of regression models, clustering algorithms, and deep learning.
- [ ] **Reinforcement Learning (RL):** Essential for DeepMind, especially given its use in projects like AlphaZero.
- [ ] **Linear Algebra:** Understanding tensors, eigenvalues, and matrix decomposition.
- [ ] **Calculus:** Derivations for optimization algorithms like gradient descent.
- [ ] Name different types of loss functions and their assumptions.
- [ ] Name different types of optimizers and their intuitions (e.g., Adam, RMSProp)
- [ ] General questions about regularization
	- [ ] **What is the difference between L1 and L2 regularization? When would you use each?**
	    - [ ] **L1 (Lasso):** Adds the absolute value of coefficients to the loss function, encouraging sparsity in the model. Use it when you suspect many irrelevant features.
	    - [ ] **L2 (Ridge):** Adds the squared value of coefficients, reducing multicollinearity. Use it when you want to shrink coefficients but keep all features.
- [ ] **Explain overfitting and strategies to mitigate it.**
	- [ ] Overfitting occurs when a model performs well on training data but poorly on unseen data. Mitigation strategies include:
	    - [ ] Using regularization (L1, L2).
	    - [ ] Increasing training data (including data augmentation)
	    - [ ] Employing dropout in neural networks.
	    - [ ] Early stopping
	    - [ ] Simpler model
- [ ] **Design a scalable recommendation system for YouTube videos.**
	- [ ] Use collaborative filtering or content-based filtering.
	- [ ] Implement a distributed pipeline for training using Apache Spark or a similar framework.
	- [ ] Utilize caching and edge computing for latency-sensitive queries.
- [ ] **How would you scale an ML model to handle billions of queries per second?**
	- [ ] Employ a distributed architecture using microservices.
	- [ ] Use load balancers and caching layers.
	- [ ] Optimize the model with quantization or distillation.
- [ ] What are the fundamentals of SVM
- [ ] Can you explain k-nearest neighbors concept?
- [ ] What is Turing Machine
- [ ] Describe the Newton algorithm, where does it come from? How can it be adapted to find a minimum in a function?
- [ ] What is gradient descent? What are second-order optimisation algorithms? How can the 2nd derivative be used in an optimisation algorithm?
- [ ] BLEU, ROUGE (for text similarity), Perplexity (for language modeling), bpc
- [ ] Mixture of expert
- [ ] What is a jacobian?
- [ ] Discuss how you would evaluate the success of an AI model.
	- [ ] Evaluating an AI model's success depends on both quantitative metrics and real-world applicability
	- [ ] Use metrics like accuracy, precision, recall, F1 score, and ROC-AUC for classification.
	- [ ] For regression, consider Root Mean Square Error (RMSE) or Mean Absolute Error (MAE).
	- [ ] Evaluate interpretability, computational efficiency, and scalability.
	- [ ] Validate with real-world test cases.

## Probability
- [ ] Review common distribution
- [ ] Review Markov Process
- [ ] What's the difference between dependence and correlation?
- [ ] What is a conjugate prior?
- [ ] The Bayes theorem
- [ ] Central Limit Theorem
- [ ] What is a random variable
- [ ] What is mean, median and mode of a probability distribution?
- [ ] Statitic test, student, t-test
- [ ] p-value
- [ ] statistics in general
- [ ] bias-variance tradeoff

## Ethics
- [ ] What are the ethical considerations in AI research?
	- Ethical considerations ensure that AI development and deployment align with societal values and avoid harm
	- Avoid biases in training data to prevent discriminatory outcomes.
	- Ensure transparency and explainability in AI decisions.
	- Consider the social impact of deploying AI in sensitive domains like healthcare or law enforcement.
- [ ] Bias mitigation, model transparency, ethics
	- [ ] For example, you might be asked how you would ensure fairness in a predictive model or reduce bias in a dataset.
	- [ ] Explain strategies such as analyzing datasets for bias, applying fairness-aware algorithms, and evaluating metrics like disparate impact.


A discussion on how you see machine learning and other computational approaches being used as tools in scientific discovery.
1. ML excels at identifying subtle, complex patterns within this data, and extracting knowledge from data. In genomics, ML models can scan millions of genetic variations across thousands of individuals to identify patterns associated with diseases like cancer or Alzheimer's.
2. Reinforcement Learning --> AlphaTensor, AlphaFold can enable discovery that are not in the data out there. Because RL generates its own data. ML helps improve climate models by learning to represent complex, small-scale phenomena like cloud formation. Researchers use ML to predict the properties of novel materials before they are synthesized.
3. Meta-learning: Learned algorithms are discovered!
4. Perhaps the most exciting frontier is using **generative models** not just to analyze data, but to create and propose new scientific hypotheses.
5. ML can also optimize the process of experimentation itself. Automating science. AlphaEvolve