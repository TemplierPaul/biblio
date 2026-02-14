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
- **JAX Tools**: Comprehensive guides for JAX, Flax NNX, Flashbax, PGX, QDax
- **Machine Learning**: Transformers, LLMs, Attention mechanisms, Diffusion models, GNNs

**Format**:
- High-level overviews (`.md` files)
- In-depth technical deep-dives (`_detailed.md` files with implementation details)
- Code examples in JAX/Flax
- Best for conceptual understanding and research grounding

---

### 2. `Questions/` - Interview Preparation ⭐

**Purpose**: Extensive, structured collection of interview questions and answers for ML/AI and Computer Science roles.

#### `Questions/ML/` - Machine Learning & AI (13 Files, ~380 Questions)

Comprehensive ML interview coverage from fundamentals to cutting-edge:

| File | Questions | Topics |
|------|-----------|--------|
| [[01_Math_Foundations]] | 10 | Optimization, linear algebra |
| [[02_Probability_Statistics]] | 9 | Distributions, Bayes, hypothesis testing |
| [[03_ML_Fundamentals]] | 3 | Metrics, model debugging |
| [[04_Loss_Functions]] | 44 | Regression/Classification losses |
| [[05_Optimizers_Regularization]] | 3 | SGD, Adam, L1/L2, Dropout |
| [[06_Classical_ML]] | 13 | SVM, k-NN, Gaussian Processes |
| [[07_Neural_Architectures]] | 31 | CNNs, LSTMs, Attention, Transformers |
| [[08_Advanced_Deep_Learning]] | 17 | Diffusion, GNNs, VLAs |
| [[09_LLM_Training]] | 38 | Pre-training, RLHF, LoRA, Peft |
| [[10_Reinforcement_Learning]] | 118 | MDPs, PPO, SAC, DQN, MPO |
| [[11_Game_Theory]] | 74 | Nash, AlphaZero, PSRO, CFR |
| [[12_ML_Systems]] | 2 | Scalability, RecSys |
| [[13_Ethics_AI]] | 16 | Fairness, bias, transparency |

**Format**: Q&A style with detailed explanations, code examples, equations, and interview tips.

#### `Questions/CS/` - Computer Science Fundamentals (9 Files, ~200 Questions)

Essential CS interview topics:

| File | Questions | Topics |
|------|-----------|--------|
| [[01_Data_Structures]] | 40 | Arrays, trees, graphs, hash tables |
| [[02_Algorithms]] | 25 | Sorting, DP, greedy, backtracking |
| [[03_Systems_Programming]] | 25 | Memory, OS, concurrency |
| [[04_Programming_Languages]] | 20 | Python vs C++, OOP, type systems |
| [[05_Distributed_Systems]] | 28 | MapReduce, Spark, Ray, DDP |
| [[06_System_Design]] | 15 | ML Pipelines, Serving, Monitoring |
| [[07_Databases_SQL]] | 16 | SQL, NoSQL, Normalization |
| [[08_ML_Infrastructure]] | 15 | GPU, CUDA, MLOps, Quantization |
| [[09_Software_Engineering]] | 15 | Git, CI/CD, Testing, Profiling |

**Format**: Q&A with implementations in Python and C++, complexity analysis, and practical examples.

**Study Tracking**:
- **[STUDY_TRACKER.md](Questions/STUDY_TRACKER.md)** ⭐: Complete checklist with all 900+ topics, progress tracking, and role-specific guidance

**See Also**:
- [Questions/ML/README.md](Questions/ML/README.md) for ML study paths and detailed organization
- [Questions/CS/README.md](Questions/CS/README.md) for CS study paths and interview tips

---

### 3. `Sources/` - Grounding Data

**Purpose**: Raw and cleaned TeX sources from ArXiv papers, used for RAG and LLM grounding.

**Current Sources**:
- `omni_epic/`, `jedi/`, `extract_qd/`, `dns/`, etc.
- Each contains raw `.tex` files and a `_full_paper_context.txt` (cleaned text).

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

## 📥 Data Acquisition: ArXiv to Local

Papers in the `Sources/` directory are acquired using [arxiv_download](https://github.com/TemplierPaul/arxiv_download). This tool downloads ArXiv TeX sources and cleans them for LLM grounding.

### Prerequisite: Start the Server (Docker)
Ensure the output directory is set to `/Users/ptemplie/Documents/ICLVault/Sources` in `docker-compose.yml`.
```bash
docker compose up --build -d
```

### Usage
1. **Load Extension**: Load `extension/manifest.json` as a temporary add-on in Firefox (`about:debugging`).
2. **Download**: Go to an ArXiv page (e.g., [1901.01753](https://arxiv.org/abs/1901.01753)).
3. **Trigger**: Click the extension icon, type a folder name (e.g., `poet`), and hit **Enter**.
4. **Result**: A new folder in `Sources/` with raw TeX and a `_full_paper_context.txt` ready for LLM use.

---

## 📊 Content Statistics

### Learning Folder
- **Topics**: 20+ major areas (Game Theory, RL, JAX ecosystem, etc.)
- **Format**: High-level overviews + detailed implementations
- **Focus**: Research depth and conceptual understanding

### Questions Folder
- **Questions**: ~580 total (378 ML, 199 CS)
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
│   ├── Evolutionary Optimization/ # QD, MAP-Elites, NEAT
│   ├── Game Theory/             # Self-play, AlphaZero, PSRO, CFR
│   ├── JAX Tools/               # JAX ecosystem guides (Flax, Flashbax)
│   ├── Machine Learning/        # Transformers, LLMs, diffusion
│   └── Reinforcement Learning/  # RL algorithms, world models
│
├── Sources/                     # Grounding data (ArXiv sources)
│   ├── used_sources.md          # Registry of processed papers
│   ├── deepseek_grpo/           # DeepSeek GRPO
│   ├── omni_epic/               # Omni-EPIC sources
│   ├── poet/                    # POET
│   └── ...                      # 20+ other paper sources
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
**When to use Sources/**: Grounding LLMs with raw paper data, RAG applications, extracting specific text
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
