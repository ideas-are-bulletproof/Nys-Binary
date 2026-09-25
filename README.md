# NyS-Binary: Nyström-based Binary Node Representations

This repository contains the official, anonymous implementation for the paper **NyS-Binary: Nyström-based Binary Node Representations**. It provides a fully reproducible environment for generating ultra-compact, high-performance binary node embeddings using Nyström approximations and structural signatures.

**Note on Anonymity:** This repository has been fully anonymized for double-blind peer review. All author names, affiliations, and institutional links have been removed.

---

## Overview

NyS-Binary achieves state-of-the-art performance in binary node representation by decoupling structural and semantic processing. It scales gracefully to large networks by utilizing Nyström approximations to avoid computing the full Singular Value Decomposition (SVD), achieving significant speed-ups with negligible drops in downstream classification accuracy.

---

## Repository Structure

### Core Codebase
* **`benchmark_new_algebraic.py`**: The core data loader and evaluation engine. It handles graph dataset acquisition (via PyTorch Geometric), sparse matrix normalization, and test/train splits.
* **`advanced_baselines.py`**: Implementation of competing continuous and binary graph representation baselines.
* **`rigorous_ablation_system.py`**: A comprehensive, highly configurable engine to dissect and ablate all core architectural components of NyS-Binary.

### Notebooks and Experiments
* **`Main.ipynb`**: The primary entry point. Contains the grand baseline benchmark pipeline evaluating NyS-Binary against continuous (GCN, MLP) and binary (NodeSketch, NodeSig, Bi-GCN, node2binary) baselines across 10 datasets.
* **`Comprehensive_Ablation_Study.ipynb`**: An exhaustive suite of ablation experiments, including the Nyström landmark scaling Pareto frontier, hyperparameter sensitivity, and execution latency profiling.

---

## Installation and Requirements

Ensure you have Python 3.8+ installed. You can install all necessary dependencies using the provided `requirements.txt` file:

    pip install -r requirements.txt

---

## Usage, Reproduction, and Configuration

### 1. Main Benchmark
To reproduce the primary 10-dataset benchmark results comparing NyS-Binary against all other baselines, run the cells in:

    jupyter notebook Main.ipynb

This notebook automatically downloads the standard datasets, evaluates representations across 4 distinct classifiers (GCN, GraphSAGE, MLP, LINKX), and generates latency and macro-average accuracy tables.

### 2. Ablation Studies
To reproduce the Nyström landmark scaling trade-offs, confidence gating, or structural vs. semantic budget splits, explore the detailed pipeline inside:

    jupyter notebook Comprehensive_Ablation_Study.ipynb

### 3. Custom Configuration
You can easily configure the system to test your own hypotheses. Inside `rigorous_ablation_system.py` or the Jupyter Notebooks, look for the `AblationConfig` class to modify:
* **`k_total`**: The total binary representation budget (default: 250).
* **`m_landmarks`**: The number of Nyström landmark nodes (default: 500).
* **`struct_ratio`**: The budget split ratio between structural and semantic features (default: 0.5).
* **`hops`**: Polynomial diffusion depth (default: 3).

---

## License
This project is released under the MIT License.
