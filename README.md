# biblio: Machine Learning Knowledge Base & Interview Prep

Welcome to **biblio**, a comprehensive knowledge repository covering Machine Learning, Reinforcement Learning, Game Theory, Computer Science fundamentals, and AI Ethics. This vault serves as both a personal knowledge base for research and an extensive interview preparation resource.

---

## 📂 Repository Structure

### 1. `Learning/` - Knowledge Base & Reference Materials

**Purpose**: Personal knowledge base with both high-level overviews and in-depth technical references. Designed for research, study, and LLM agent grounding.

**Coverage**:
- **Game Theory**: Self-play algorithms, AlphaZero, PSRO, NeuPL, CFR, Nash Equilibrium
- **Reinforcement Learning**: World Models, RL algorithms (MPO, CrossQ, PPO, SAC), RNN policies
- **Evolutionary Optimization**: Quality-Diversity, MAP-Elites, NEAT, CMA-ES
- **JAX Ecosystem**: Comprehensive guides for JAX, Flax NNX, Flashbax, PGX, QDax
- **Machine Learning**: Transformers, LLMs, Attention mechanisms, Diffusion models, GNNs

**Format**:
- High-level overviews (`.md` files)
- In-depth technical deep-dives (`_detailed.md` files with implementation details)
- Code examples in JAX/Flax
- Best for conceptual understanding and research grounding

---

### 2. `Questions/` - Interview Preparation ⭐

**Purpose**: Extensive, structured collection of interview questions and answers for ML/AI and Computer Science roles.

#### `Questions/ML/` - Machine Learning & AI (11 Files, ~700+ Questions)

Comprehensive ML interview coverage from fundamentals to cutting-edge:

| File | Topics | Coverage |
|------|--------|----------|
| **01_ANN_Architecture** | LSTM, Attention, Transformers | Neural network architectures |
| **02_LLM_Training** | Pre-training, Fine-tuning, RLHF, LoRA | Large language models |
| **03_Deep_Learning** | GNNs, Diffusion Models, VLAs | Advanced deep learning |
| **04_Classical_ML** | Gaussian Processes | Bayesian methods |
| **05_Game_Theory** | Nash, MCTS, AlphaZero, PSRO, CFR | Multi-agent systems |
| **06_Reinforcement_Learning** ⭐ | MDPs, Q-learning, PPO, SAC, DQN | Complete RL (12 parts) |
| **07_ML_Fundamentals** ⭐ | SVM, k-NN, metrics, optimizers | Classical ML algorithms |
| **08_Probability_Statistics** ⭐ | Distributions, Bayes, hypothesis testing | Statistical foundations |
| **09_ML_Systems** | Scalability, RecSys, deployment | Production ML systems |
| **10_Ethics_AI** | Fairness, bias mitigation, transparency | Responsible AI |
| **11_Math_Foundations** | Optimization, linear algebra, VAE, GAN | Mathematical theory |

**Format**: Q&A style with detailed explanations, code examples, equations, and interview tips.

#### `Questions/CS/` - Computer Science Fundamentals (4 Files, ~200+ Questions)

Essential CS interview topics:

| File | Topics | Coverage |
|------|--------|----------|
| **01_Data_Structures** | Arrays, trees, graphs, hash tables | Core data structures |
| **02_Algorithms** | Sorting, DP, greedy, backtracking | Algorithm design |
| **03_Systems_Programming** | Memory, OS, concurrency, compilation | Low-level programming |
| **04_Programming_Languages** | Python vs C++, OOP, type systems | Language concepts |

**Format**: Q&A with implementations in Python and C++, complexity analysis, and practical examples.

**Study Tracking**:
- **[STUDY_TRACKER.md](Questions/STUDY_TRACKER.md)** ⭐: Complete checklist with all 900+ topics, progress tracking, and role-specific guidance

**See Also**:
- [Questions/ML/README.md](Questions/ML/README.md) for ML study paths and detailed organization
- [Questions/CS/README.md](Questions/CS/README.md) for CS study paths and interview tips

---

## 🎯 Use Cases

### For Personal Learning & Research
**Use**: `Learning/` folder
- Deep-dive into topics with detailed notes
- Understand algorithms at implementation level
- Reference for research projects
- Ground LLM agents with domain knowledge

### For Interview Preparation
**Use**: `Questions/` folder
- Structured Q&A for systematic prep
- Track progress with **STUDY_TRACKER.md** (complete checklist)
- Practice coding implementations
- Review key concepts by topic
- Follow role-specific study paths

### For Quick Reference
**Use**: Both folders
- `Learning/`: Conceptual understanding and research context
- `Questions/`: Quick lookup of interview topics and formulas

---

## 🚀 Getting Started

### Path 1: Learning & Research
1. **Explore Topics**: Browse Learning/ folder by research area
2. **Deep Dive**: Read `_detailed.md` files for implementation details
3. **Code**: Review JAX/Flax examples in topic folders

### Path 2: Interview Preparation

**Start Here**: Use [STUDY_TRACKER.md](Questions/STUDY_TRACKER.md) to track progress across all topics

**ML/AI Roles:**
1. **Fundamentals** (Week 1-2): Questions/ML/07, 08, 11
2. **Deep Learning** (Week 3-4): Questions/ML/01, 02, 03
3. **Specialization** (Week 5-6): Questions/ML/06 (RL), 05 (Game Theory), or 09 (Systems)
4. **Practice**: Code implementations, mock interviews

**SWE Roles:**
1. **Data Structures** (Week 1-2): Questions/CS/01
2. **Algorithms** (Week 3-4): Questions/CS/02
3. **Systems** (Week 5): Questions/CS/03, 04
4. **ML Basics** (Week 6): Questions/ML/07, 08

**Research Scientist:**
1. **Theory**: Questions/ML/08, 11, 04
2. **Core**: Questions/ML/06 (full RL), 05 (Game Theory)
3. **Advanced**: Questions/ML/02, 03
4. **Practice**: Implement algorithms from Learning/

See specific README files in Questions/ML and Questions/CS for detailed study paths.

---

## 📊 Content Statistics

### Learning Folder
- **Topics**: 20+ major areas (Game Theory, RL, JAX ecosystem, etc.)
- **Format**: High-level overviews + detailed implementations
- **Focus**: Research depth and conceptual understanding

### Questions Folder
- **Questions**: 900+ total (700+ ML, 200+ CS)
- **Code Examples**: 500+ implementations
- **Equations**: 400+ mathematical formulas
- **Coverage**: Fundamentals to PhD-level research
- **Languages**: Python, C++, PyTorch, JAX, sklearn

---

## 🛠️ Tech Stack

### Learning Materials
- **Framework**: JAX, Flax NNX for implementations
- **Tools**: PGX (games), QDax (quality-diversity), Flashbax (replay)
- **Viewer**: Obsidian recommended (supports WikiLinks `[[...]]`)

### Interview Prep
- **Languages**: Python, C++ (for CS topics)
- **ML Libraries**: PyTorch, scikit-learn, fairlearn
- **Topics**: Covers all major ML/CS interview areas

---

## 📂 Directory Tree

```
biblio/
├── Learning/                    # Knowledge base & research references
│   ├── Game Theory/             # Self-play, AlphaZero, PSRO, CFR
│   │   ├── Self-play/           # Self-play algorithms
│   │   └── *.md                 # High-level + _detailed files
│   ├── Reinforcement Learning/  # RL algorithms, world models
│   ├── Machine Learning/        # Transformers, LLMs, diffusion
│   ├── JAX/                     # JAX ecosystem guides
│   └── ...                      # Other research topics
│
└── Questions/                   # Interview Q&A collection
    ├── STUDY_TRACKER.md         # Progress tracking checklist ⭐
    │
    ├── ML/                      # ML/AI interviews (11 files)
    │   ├── README.md            # ML study paths & organization
    │   ├── 01_ANN_Architecture.md
    │   ├── 02_LLM_Training.md
    │   ├── 06_Reinforcement_Learning.md ⭐
    │   ├── 07_ML_Fundamentals.md ⭐
    │   ├── 08_Probability_Statistics.md ⭐
    │   ├── 09_ML_Systems.md
    │   ├── 10_Ethics_AI.md
    │   ├── 11_Math_Foundations.md
    │   └── ...
    │
    └── CS/                      # CS fundamentals (4 files)
        ├── README.md            # CS study paths & tips
        ├── 01_Data_Structures.md
        ├── 02_Algorithms.md
        ├── 03_Systems_Programming.md
        └── 04_Programming_Languages.md
```

---

## 🎓 Key Differences: Learning vs Questions

| Aspect | Learning/ | Questions/ |
|--------|-----------|------------|
| **Purpose** | Knowledge base, research | Interview preparation |
| **Format** | Essays, detailed notes | Q&A, structured answers |
| **Depth** | Variable (overview + deep-dives) | Comprehensive but focused |
| **Code** | JAX implementations | Python/C++ examples |
| **Audience** | Self, LLM agents, researchers | Interview candidates |
| **Organization** | By research topic | By interview domain |
| **Style** | Exploratory, detailed | Concise, practical |

**When to use Learning/**: Understanding concepts deeply, research projects, implementing algorithms
**When to use Questions/**: Interview prep, quick review, systematic coverage of topics

---

## 💡 Tips for Success

### Using the Learning Folder
- Start with high-level `.md` files for overview
- Dive into `_detailed.md` for implementation depth
- Use as reference while coding projects
- Great for grounding LLM agents with domain knowledge

### Using the Questions Folder
- Follow structured study paths in READMEs
- Practice coding examples from scratch
- Focus on understanding, not memorization
- Use role-specific checklists for targeted prep

### Combining Both
1. Learn concept in `Learning/` (deep understanding)
2. Practice with `Questions/` (interview format)
3. Implement algorithms from `Learning/` (hands-on)
4. Review Q&A in `Questions/` (reinforce)

---

## 🤝 Contributing

This is a personal knowledge vault, but feedback is welcome:
- **Errors**: If you spot mistakes, please flag them
- **Suggestions**: Ideas for additional topics or improvements
- **Format**: Keep Learning/ exploratory, Questions/ structured

---

## 📚 Recommended External Resources

### Books
- **RL**: Sutton & Barto - *Reinforcement Learning: An Introduction*
- **Deep Learning**: Goodfellow et al. - *Deep Learning*
- **Game Theory**: Shoham & Leyton-Brown - *Multiagent Systems*
- **Algorithms**: CLRS - *Introduction to Algorithms*

### Online
- **RL**: OpenAI Spinning Up, Berkeley CS285
- **ML**: Stanford CS229, Fast.ai
- **JAX**: Official JAX documentation
- **Interview Prep**: LeetCode, Project Euler

---

## 📄 License

Personal knowledge base - content compiled from research papers, textbooks, and original notes.

---

**Happy Learning & Good Luck with Interviews!** 🚀

*Remember: Deep understanding beats memorization. Use Learning/ for depth, Questions/ for breadth.*
