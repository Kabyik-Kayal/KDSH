# 🐉 KDSH 2026: Narrative Consistency Detection with Dragon Hatchling Architecture

> **Detecting Character Backstory Contradictions in 19th-Century Literature Using Biologically-Inspired Neural Networks**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)
[![Pathway 0.27](https://img.shields.io/badge/Pathway-0.27+-green.svg)](https://pathway.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains our **KDSH 2026 Track B** submission, implementing **TextPath** — a novel adaptation of the biologically-inspired [Dragon Hatchling (BDH)](https://arxiv.org/abs/2509.26507) architecture for automated narrative consistency verification in classical literature.

---

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Project Structure](#-project-structure)
4. [Installation](#-installation)
5. [Usage](#-usage)
6. [Configuration](#-configuration)
7. [Modules Reference](#-modules-reference)
8. [Visualizations](#-visualizations)
9. [Technical Details](#-technical-details)
10. [References](#-references)

---

## 🎯 Project Overview

### The Challenge

**Task**: Given a 19th-century novel and a character backstory, classify whether the backstory is **consistent** with or **contradicts** the original narrative.

**Dataset**: 80 labeled training examples from two novels:
- *The Count of Monte Cristo* by Alexandre Dumas
- *In Search of the Castaways* by Jules Verne

**Challenges**:
| Challenge | Description |
|-----------|-------------|
| 📖 Long-context processing | Novels contain tens of thousands of lines |
| 🎭 Complex relationships | Character relationships and plot arcs span entire novels |
| 🔍 Subtle contradictions | Requires deep narrative understanding |
| ⚖️ Plausible vs. impossible | Distinguishing alternate backstories from contradictions |
| 📊 Limited data | Only 80 examples for binary classification |

### Our Solution: TextPath with BDH Architecture

We employ a **novel-specific pipeline** that leverages the unique biological properties of the Dragon Hatchling (BDH) architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         KDSH Pipeline                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │   Pathway    │    │  Novel-BDH   │    │   Classification     │   │
│  │   RAG        │ →  │  Language    │ →  │   Head               │   │
│  │   Retrieval  │    │  Models      │    │   (MLP → Binary)     │   │
│  └──────────────┘    └──────────────┘    └──────────────────────┘   │
│                                                                     │
│  1. Chunk novels     2. Process with     3. Classify as             │
│     into passages       Hebbian-trained     Consistent/Contradict   │
│     + embed with        BDH model                                   │
│     sentence-transformers                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Key Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Pathway RAG** | Retrieve relevant passages from novels | `PathwayNovelRetriever` with sentence-transformers embeddings |
| **Novel-Specific BDH** | Learn narrative patterns per novel | Separate pretrained models for each novel |
| **Classification Head** | Binary prediction | MLP on pooled BDH representations |

---

## 🏗️ Architecture

### TextPath: BDH-Based Language Model

```
┌──────────────────────────────────────────────────────────────────────┐
│                      TextPath Architecture                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input: Token IDs [batch_size, seq_len]                              │
│           ↓                                                          │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Token Embedding Layer (vocab_size=16384, d_model=256)          │  │
│  └────────────────────────────────────────────────────────────────┘  │
│           ↓                                                          │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ BDH Layers (L=4)                                               │  │
│  │   ├─ Scale-Free Neuron Network (N=4096 neurons)                │  │
│  │   ├─ Sparse Activations (~5% active neurons)                   │  │
│  │   ├─ Multi-Head Attention (H=8 heads)                          │  │
│  │   └─ RoPE Positional Encoding (max_seq_len=4096)               │  │
│  └────────────────────────────────────────────────────────────────┘  │
│           ↓                                                          │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Classification Mode (when enabled):                            │  │
│  │   LayerNorm → Dropout → Linear(256→128) → GELU                 │  │
│  │   → Dropout → Linear(128→2) → [Contradict, Consistent]         │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Output: Logits [batch_size, 2] for classification                   │
│          OR Logits [batch_size, seq_len, vocab_size] for LM          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### BDH Biological Properties

The Dragon Hatchling architecture provides three key advantages for narrative understanding:

#### 1. Hebbian Learning
> *"Neurons that fire together, wire together"*

Pre-training on sequential novel passages builds causal circuits encoding:
- Character relationships (Dantès → Mercédès, Fernand → betrayal)
- Plot events (imprisonment → escape → revenge)
- Narrative logic (foreshadowing → resolution)

$$\Delta w_{ij} \propto x_i \cdot y_j$$

#### 2. Sparse Activations (~5%)
Each concept (character, location, event) activates distinct neuron groups:
- Creates **monosemantic representations** (one concept per neuron cluster)
- Makes contradictions detectable as conflicting activation patterns
- Enables interpretability of what the model "knows"

$$\|a\|_0 \approx 0.05 \cdot N$$

#### 3. Causal Circuits
Learned connectivity graph encodes "if A then B" reasoning:

$$G_x = E \cdot D_x$$

Where $E$ encodes edge weights and $D_x$ encodes input-dependent dynamics.

### Novel-Specific Routing

The `NovelSpecificClassifier` routes each sample to the appropriate pretrained model:

```python
# Automatic model selection based on book_name
NOVEL_MODEL_MAP = {
    'castaways': 'textpath_in_search_of_the_castaways.pt',
    'monte cristo': 'textpath_the_count_of_monte_cristo.pt',
}
```

This ensures each novel's unique narrative patterns are captured by a dedicated model.

---

## 📂 Project Structure

```
KDSH/
├── run_pipeline.py                    # 🚀 Main CLI entry point (187 lines)
├── requirements.txt                   # Dependencies (293 packages)
├── results.csv                        # Final predictions for submission
├── LICENSE                            # MIT License
│
├── Dataset/
│   ├── train.csv                      # 80 labeled training pairs
│   ├── test.csv                       # Unlabeled test set
│   └── Books/
│       ├── The Count of Monte Cristo.txt
│       └── In search of the castaways.txt
│
├── models/                            # Trained model checkpoints
│   ├── custom_tokenizer.json          # 16,384 vocab BPE tokenizer
│   ├── textpath_pretrained.pt         # Generic pretrained model
│   ├── textpath_the_count_of_monte_cristo.pt    # Monte Cristo BDH
│   ├── textpath_in_search_of_the_castaways.pt   # Castaways BDH
│   ├── textpath_classifier_best.pt              # Best classifier
│   ├── textpath_classifier_best_monte_cristo.pt # Novel-specific
│   └── textpath_classifier_best_castaways.pt    # Novel-specific
│
├── src/                               # Source code modules
│   ├── __init__.py                    # Package exports
│   ├── config.py                      # PipelineConfig dataclass
│   │
│   ├── data_processing/               # Data and RAG
│   │   ├── __init__.py
│   │   ├── retrieval.py               # PathwayNovelRetriever class
│   │   ├── classification_dataset.py # PyTorch Dataset
│   │   ├── build_retrievers.py        # Retriever factory functions
│   │   ├── ingest.py                  # Data ingestion utilities
│   │   └── train_tokenizer.py         # BPE tokenizer training
│   │
│   ├── models/                        # Neural network modules
│   │   ├── __init__.py
│   │   ├── textpath.py                # TextPath/BDH core
│   │   ├── textpath_classifier.py     # Classifier wrappers
│   │   ├── finetune_classifier.py     # train_epoch/validate functions
│   │   ├── pretrain_bdh_native.py     # Hebbian pretraining script
│   │   └── state_manager.py           # Synaptic state management
│   │
│   ├── training/                      # Training infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Trainer class
│   │   └── pretraining.py             # Pretraining runner
│   │
│   ├── evaluation/                    # Evaluation and prediction
│   │   ├── __init__.py
│   │   ├── evaluate.py                # Metrics, prediction
│   │   └── inference.py               # Inference utilities
│   │
│   ├── visualization/                 # Analysis and plots
│   │   ├── __init__.py
│   │   └── visualize.py               # All visualization functions
│   │
│   └── utils/                         # Helper functions
│       └── __init__.py
│
├── visualizations/                    # Generated plots
│   ├── consistency_embedding_space.png
│   ├── prediction_confidence.png
│   ├── accuracy_by_character.png
│   └── accuracy_by_book.png
│
├── repos/                             # External dependencies
│   ├── bdh_educational/               # Educational BDH implementation
│   │   ├── bdh.py                     # Core BDH module
│   │   └── utils/
│   └── bdh_official/                  # Official BDH repo reference
│
├── outputs/                           # Training artifacts
│   ├── optimal_config.json
│   ├── train_predictions.csv
│   └── tuning_retrieval_k.json
│
└── logs/                              # Training logs
```

---

## 🔧 Installation

### Prerequisites
- Python 3.11+
- conda (recommended) or pip
- ~8GB RAM (for embedding models)

### Setup

```bash
# Clone the repository
git clone https://github.com/kabyik-kayal/kdsh.git
cd kdsh

# Create conda environment
conda create -n kds python=3.11
conda activate kds

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.9.1 | Deep learning framework |
| `pathway` | 0.27.1 | RAG document indexing (Track B requirement) |
| `sentence-transformers` | 5.2.0 | Embedding model for retrieval |
| `tokenizers` | 0.22.2 | BPE tokenizer |
| `scikit-learn` | 1.8.0 | Metrics and evaluation |
| `pandas` | 2.3.3 | Data manipulation |
| `matplotlib` | 3.10.8 | Visualization |
| `tqdm` | 4.67.1 | Progress bars |

---

## 🚀 Usage

### Quick Start

```bash
# Run the complete pipeline (train → evaluate → predict)
python run_pipeline.py --mode full
```

### Pipeline Modes

| Mode | Description | Command |
|------|-------------|---------|
| `pretrain` | Hebbian pretraining on novel texts | `python run_pipeline.py --mode pretrain` |
| `train` | Train classification head | `python run_pipeline.py --mode train` |
| `evaluate` | Evaluate on validation split | `python run_pipeline.py --mode evaluate` |
| `predict` | Generate test predictions | `python run_pipeline.py --mode predict` |
| `full` | Train + Evaluate + Predict | `python run_pipeline.py --mode full` |

### Command-Line Options

```bash
python run_pipeline.py --help

options:
  --mode {pretrain,train,predict,evaluate,full}
                        Pipeline mode (default: full)
  --pretrain-epochs PRETRAIN_EPOCHS
                        BDH pretraining epochs (default: 50)
  --epochs EPOCHS       Classifier training epochs (default: 15)
  --batch-size BATCH_SIZE
                        Training batch size (default: 4)
  --lr LR               Learning rate (default: 1e-4)
```

### Examples

```bash
# Pretrain BDH models for 100 epochs
python run_pipeline.py --mode pretrain --pretrain-epochs 100

# Train classifier for 20 epochs with larger batch
python run_pipeline.py --mode train --epochs 20 --batch-size 8

# Just generate predictions (requires trained model)
python run_pipeline.py --mode predict

# Generate visualizations
python src/visualization/visualize.py
```

---

## ⚙️ Configuration

All configuration is centralized in `src/config.py` via the `PipelineConfig` dataclass:

### Paths

| Parameter | Default | Description |
|-----------|---------|-------------|
| `novels_dir` | `Dataset/Books/` | Directory containing novel .txt files |
| `train_csv` | `Dataset/train.csv` | Training data CSV |
| `test_csv` | `Dataset/test.csv` | Test data CSV |
| `tokenizer_path` | `models/custom_tokenizer.json` | BPE tokenizer |
| `models_dir` | `models/` | Directory for checkpoints |
| `output_model` | `models/textpath_classifier_best.pt` | Best model path |
| `output_predictions` | `results.csv` | Predictions output |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4 | Training batch size |
| `epochs` | 15 | Classifier training epochs |
| `learning_rate` | 1e-4 | Initial learning rate |
| `weight_decay` | 0.01 | AdamW weight decay |
| `max_tokens` | 512 | Maximum sequence length |

### Freezing Strategy

| Parameter | Default | Description |
|-----------|---------|-------------|
| `freeze_bdh` | True | Freeze BDH layers initially |
| `unfreeze_after_epoch` | 5 | Epoch to unfreeze BDH |
| `unfreeze_lr_multiplier` | 0.1 | LR multiplier after unfreezing |

### RAG Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 200 | Words per chunk (~250 tokens) |
| `overlap` | 50 | Overlapping words between chunks |
| `top_k_retrieval` | 2 | Number of passages to retrieve |

### Class Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `class_weight_inconsistent` | 1.7 | Weight for "Contradict" class |
| `class_weight_consistent` | 1.0 | Weight for "Consistent" class |

### Device

Automatically detected in order: CUDA → MPS (Apple Silicon) → CPU

---

## 📊 Visualizations

The pipeline generates analysis plots in `visualizations/`:

| Plot | Description |
|------|-------------|
| `consistency_embedding_space.png` | 2D t-SNE projection showing consistent vs contradictory samples |
| `prediction_confidence.png` | Distribution of model confidence (entropy) across predictions |
| `accuracy_by_character.png` | Per-character classification accuracy |
| `accuracy_by_book.png` | Per-novel classification accuracy |

### Generate Visualizations

```bash
python src/visualization/visualize.py
```

This analyzes the trained classifier and generates all plots.

---

## 🔬 Technical Details

### TextPath Configuration

```python
@dataclass
class TextPathConfig:
    vocab_size: int = 16384     # Custom BPE tokenizer vocabulary
    max_seq_len: int = 4096     # Maximum sequence length
    n_heads: int = 8            # Attention heads
    n_neurons: int = 4096       # BDH neurons (scale-free graph)
    d_model: int = 256          # Model embedding dimension
    n_layers: int = 4           # Number of BDH layers
    dropout: float = 0.1        # Dropout rate
    use_rope: bool = True       # Rotary position encoding
    sparsity_target: float = 0.05  # 5% neuron activation target
    classification_mode: bool = False  # Enable classification head
```

### Training Strategy

1. **Phase 1: BDH Frozen** (epochs 1-5)
   - Only train classification head
   - Learning rate: 1e-4
   - Preserves pretrained narrative knowledge

2. **Phase 2: Full Fine-tuning** (epochs 6-15)
   - Unfreeze BDH layers
   - Reduced learning rate: 1e-5 (0.1× multiplier)
   - Gentle adaptation to classification task

3. **Optimizer**: AdamW with weight decay 0.01
4. **Scheduler**: Cosine annealing over total epochs
5. **Class Weights**: [1.7, 1.0] to handle imbalance (~36% contradict, ~64% consistent)

### Pathway Integration

```python
# Creating Pathway table from chunks
self.chunks_table = pw.debug.table_from_rows(
    schema=pw.schema_from_dict({"text": str}),
    rows=[(chunk,) for chunk in self.chunks]
)

# Embedding with sentence-transformers via Pathway
from pathway.xpacks import llm
self.embedder = llm.embedders.SentenceTransformerEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Model Files

| File | Size | Description |
|------|------|-------------|
| `textpath_the_count_of_monte_cristo.pt` | ~50MB | Monte Cristo pretrained BDH |
| `textpath_in_search_of_the_castaways.pt` | ~50MB | Castaways pretrained BDH |
| `textpath_classifier_best.pt` | ~55MB | Best classifier checkpoint |
| `custom_tokenizer.json` | ~2MB | 16,384 vocab BPE tokenizer |

---

## 📚 References

- **Dragon Hatchling (BDH)**: [arXiv:2509.26507](https://arxiv.org/abs/2509.26507) - Kosowski et al. (2025)
- **Pathway**: [pathway.com](https://pathway.com/) - Real-time data processing framework
- **sentence-transformers**: [SBERT.net](https://www.sbert.net/) - Sentence embeddings
- **Rotary Position Embedding (RoPE)**: [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

---