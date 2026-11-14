# Reproduction & Execution Guide

This document details how to reproduce all benchmark tables, main manuscript figures (Figures 2–4), appendix heatmaps (Figures 5–6), and comprehensive ablation studies from the manuscript using Jupyter notebooks or command-line scripts.

---

## 1. Main Benchmark Pipeline (`Main.ipynb`)

`Main.ipynb` reproduces **Table 1** (overall mean accuracy across 9 downstream classifiers), **Table 2** (homophilous vs. heterophilous breakdown), **Table 3** (embedding generation latency across datasets), and **Table 6** (comprehensive per-dataset accuracy breakdown).

### Notebook Configuration
Open [`Main.ipynb`](Main.ipynb). In **Cell 2**, configure the evaluation parameters and datasets:

```python
# -------------------------------------------------------------
# Benchmark Evaluation Configuration
# -------------------------------------------------------------
emb_dim = 250                    # Bit budget (250 bits for Table 1-2)
train_ratio = 0.70               # 70% stratified training split
seeds = [42, 123, 77]            # 3 random seeds
load_previous_checkpoint = False # False = fresh evaluation from scratch

# --- Datasets ---
datasets = [
    'cora',
    'citeseer',
    'pubmed',
    'wikics',
    'chameleon',
    'squirrel',
    'photo',
    'computers',
    'amazon-ratings',
    'dblp',

    ##### Highly Heterophilic Datasets (Uncomment to run) #####
    # 'actor',
    # 'texas',
    # 'wisconsin',
    # 'minesweeper',

    ### Large Datasets (Uncomment to run) ###
    # 'ogbn-arxiv',
    # 'ogbn-products',
]
```

### Dataset Selection: Running All Datasets (OGB, Heterophilic, & Core Benchmarks)
In [`Main.ipynb`](Main.ipynb) **Cell 2**, all supported datasets are organized inside the `datasets` list:

1. **Default Run (10 Core Benchmark Graphs):**  
   The notebook comes pre-configured with the 10 benchmark datasets uncommented (`cora`, `citeseer`, `pubmed`, `wikics`, `chameleon`, `squirrel`, `photo`, `computers`, `amazon-ratings`, `dblp`).
2. **Running Large-Scale OGB Datasets (`ogbn-arxiv`, `ogbn-products`):**  
   To evaluate large-scale graphs (e.g. `ogbn-arxiv` with $N=169{,}343$ nodes or `ogbn-products` with $N=2{,}449{,}029$ nodes), **simply uncomment the dataset line in the `datasets` list**:
   ```python
   ### Large Datasets ###
   'ogbn-arxiv',     # <-- Uncomment to run ogbn-arxiv (Table 14)
   # 'ogbn-products' # <-- Uncomment to run ogbn-products
   ```
3. **Running Additional Heterophilic Datasets (`texas`, `wisconsin`, `actor`, `minesweeper`):**  
   To evaluate the additional heterophilic benchmarks reported in Appendix A.6, **uncomment the respective dataset lines** under `##### Highly Heterophilic Datasets #####`:
   ```python
   ##### Highly Heterophilic Datasets (Uncomment to run) #####
   'actor',
   'texas',
   'wisconsin',
   'minesweeper',
   ```
4. **Running a Single Dataset for Fast Validation:**  
   To run a quick single-dataset test (e.g., just `ogbn-arxiv` or `cora`), comment out the other datasets and leave only your target dataset:
   ```python
   datasets = [
       'ogbn-arxiv',  # Run solely on ogbn-arxiv
   ]
   ```

### Execution Steps
* **Cell 1**: Imports core dependencies and verifies PyTorch / CUDA device setup.
* **Cell 2**: Sets the evaluation hyperparameters and active dataset selection above.
* **Cell 3**: Runs the complete evaluation across the selected datasets and 9 downstream classifiers. Results are saved automatically to `./Results/final_benchmark_results_latest.csv`.
* **Cells 4 & 5**: Formats and prints the consolidated tables (mean ± std accuracy, macro-F1, and generation runtime).

### Alternative: Run via Command Line
The main benchmark can also be executed directly via terminal:

```bash
# Full 10-dataset, 9-classifier benchmark at 250 bits
python benchmark_new_algebraic.py --emb_dim 250 --train_ratio 0.70 --seeds 42 123 77

# Fast validation on citation networks (Cora, CiteSeer, PubMed)
python benchmark_new_algebraic.py --datasets cora citeseer pubmed --emb_dim 250

# Run on large-scale OGB datasets (Table 14)
python benchmark_new_algebraic.py --datasets ogbn-arxiv --emb_dim 250
python benchmark_new_algebraic.py --datasets ogbn-products --emb_dim 250

# Run on specialized heterophilic graphs (Appendix A.6)
python benchmark_new_algebraic.py --datasets texas wisconsin actor minesweeper --emb_dim 250
```

---

## 2. Main Manuscript Figures (Figures 2–4 and Appendix Heatmaps)

The figures in the manuscript visualize overall accuracy vs. latency trade-offs, empirical speedups, per-dataset downstream accuracy heatmaps, and geometric class separability.

These figures can be generated in two ways:

### Option A: Interactive Figure Generation (`Generate_Paper_Figures.ipynb`)
1. Open [`Generate_Paper_Figures.ipynb`](Generate_Paper_Figures.ipynb).
2. **Zero Predefined Values**: The notebook dynamically reads and parses the empirical evaluation data strictly from `./Results/` (including `final_benchmark_results_latest.csv` and all benchmark runs). No hardcoded fallback dictionaries are used.
3. Execute the notebook cells sequentially to preview and inspect each figure interactively:
   * **Cell 2**: Automatically detects, validates, and loads all empirical benchmark results from `./Results/`.
   * **Cell 3**: Generates **Figure 2** (Accuracy vs. Generation Latency Pareto frontier across 10 benchmark datasets).
   * **Cell 4**: Generates **Figure 3** (Embedding Generation Latency by Method & relative speedup bars on log scale).
   * **Cell 5**: Generates **Figure 5** (NyS-Binary Supervised Downstream Accuracy Heatmap) and **Figure 6** (NyS-Binary Unsupervised Downstream Accuracy Heatmap) in Appendix Section A.5.
   * **Cell 6**: Generates Appendix Figure (Grouped Downstream Accuracy across 9 downstream probe classifiers).
   * **Cell 7**: Generates **Figure 4** (Geometric Class Separation Ratio $R = D_{\mathrm{inter}}/D_{\mathrm{intra}}$ per dataset).

### Option B: Direct Command-Line Generation (`generate_figures.py`)
Run the figure generation script from the terminal:

```bash
python generate_figures.py
```

This reads `./Results/final_benchmark_results_latest.csv` and outputs publication-grade 300 DPI vector PDF and PNG figures in `./figures/` (mirrored to `../../paper/`):
* **Figure 2 (Main Text §4.3):**  
  `./figures/fig_pareto_latency_accuracy.pdf` (Accuracy vs. Latency Pareto frontier, highlighting supervised and unsupervised NyS-Binary).
* **Figure 3 (Main Text §4.3):**  
  `./figures/fig_generation_time.pdf` (Embedding Generation Latency by Method on log scale).
* **Figure 4 (Main Text §4.4):**  
  `./figures/fig_class_separation.pdf` (Geometric Class Separation Ratio $R = D_{\mathrm{inter}}/D_{\mathrm{intra}}$).
* **Figure 5 (Appendix §A.5):**  
  `./figures/fig_nys_heatmap.pdf` (NyS-Binary Supervised Downstream Accuracy Heatmap across 10 datasets and 7 probe architectures).
* **Figure 6 (Appendix §A.5):**  
  `./figures/fig_nys_unsupervised_heatmap.pdf` (NyS-Binary Unsupervised Downstream Accuracy Heatmap).
* **Appendix Classifier Breakdown:**  
  `./figures/fig_accuracy_by_classifier.pdf` (Grouped probe breakdown across 9 classifiers).

---

## 3. Ablation Studies & Sensitivity Sweeps (`Comprehensive_Ablation_Study.ipynb`)

> **Note for Reviewers & Viewers:**  
> To view and verify all ablation results, heatmaps, Pareto trade-off curves, and unit test sanity checks exactly as the authors intended, please open [`Comprehensive_Ablation_Study.ipynb`](Comprehensive_Ablation_Study.ipynb) and run the notebook **completely from start to finish**.

---

### Individual Ablation Study Configurations

If you wish to configure, modify, or re-run specific ablations individually, you can execute their respective sections in [`Comprehensive_Ablation_Study.ipynb`](Comprehensive_Ablation_Study.ipynb) or use the corresponding standalone Python code:

#### Figure 7 & Table 11: Diffusion Horizon Sensitivity ($H \in \{0, 1, 2, 3, 4, 6, 8\}$)
* **Notebook Location:** Section 2.2
* **CLI Command:**
  ```bash
  python rigorous_ablation_system.py --sweep hops_sweep --hops_list 0 1 2 3 4 6 8
  ```
* **Python Configuration:**
  ```python
  import rigorous_ablation_system as ras
  for h in [0, 1, 2, 3, 4, 6, 8]:
      cfg = ras.AblationConfig(hops=h, m_landmarks=125, k_total=250)
  ```

#### Figure 8: Factorial Grid Sweep ($\alpha \in [0, 1]$ vs. $\tau \in [0.4, 0.95]$)
* **Notebook Location:** Section 2.1
* **CLI Command:**
  ```bash
  python rigorous_ablation_system.py --sweep grid_sweep
  ```
* **Python Configuration:**
  ```python
  alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
  taus = [0.40, 0.50, 0.60, 0.70, 0.85, 0.95]
  ```

#### Figure 9 & Table 12: Landmark Budget Pareto Sweep ($m \in [64, 1000]$)
* **Notebook Location:** Section 2.3
* **CLI Command:**
  ```bash
  python rigorous_ablation_system.py --sweep landmark_sweep --m_list 64 125 250 500 1000
  ```
* **Python Configuration:**
  ```python
  for m in [64, 125, 250, 500, 1000]:
      cfg = ras.AblationConfig(m_landmarks=m, k_total=250)
  ```

#### Figure 10 & Table 13: Structural vs. Label Dimension Split ($k_{\text{struct}} : k_{\text{label}}$)
* **Notebook Location:** Section 2.4
* **CLI Command:**
  ```bash
  python rigorous_ablation_system.py --sweep budget_sweep
  ```
* **Python Configuration:**
  ```python
  for r in [0.0, 0.3, 0.5, 0.7, 1.0]:
      cfg = ras.AblationConfig(struct_ratio=r, k_total=250)
  ```

#### Table 10: Structural Backbone Comparison
* **Notebook Location:** Section 2.5
* **CLI Command:**
  ```bash
  python rigorous_ablation_system.py --sweep backbone_sweep
  ```
* **Python Configuration:**
  ```python
  for mode in ["nystrom", "exact_svd", "random_projection", "chebyshev"]:
      cfg = ras.AblationConfig(struct_mode=mode)
  ```

---

## 4. Direct Table & Figure Mapping

| Manuscript Item | Content | Generating Notebook / Script | Output / Saved Artifact |
| :--- | :--- | :--- | :--- |
| **Table 1** | Main Benchmark: Overall accuracy across 9 classifiers | `Main.ipynb` (Cells 3–4) or `benchmark_new_algebraic.py` | `./Results/final_benchmark_results_latest.csv` |
| **Table 2** | Homophilous vs. Heterophilous Macro-Averaged Accuracy | `Main.ipynb` (Cell 4) or `benchmark_new_algebraic.py` | Summary printed to console / CSV |
| **Table 3** | Generation Latency (seconds) across 10 datasets | `Main.ipynb` (Cell 5) | Latency column in `./Results/` |
| **Table 4** | Summary statistics of the 10 benchmark graph datasets | Standard dataset metadata (`Main.ipynb` Cell 2) | Formatted in manuscript |
| **Table 5** | Memory footprint and compression savings (Float32 vs. 1-bit) | Memory analysis routines | Formatted in manuscript |
| **Table 6** | Comprehensive Per-Dataset Classification Accuracy $\pm$ Std across 10 datasets | `Main.ipynb` (Cells 3–4) or `benchmark_new_algebraic.py` | `./Results/final_benchmark_results_latest.csv` |
| **Figure 1** | Framework Pipeline Architecture Diagram | TikZ source (`paper/nys_binary_model_detailed.tex`) | Rendered in manuscript PDF |
| **Figure 2** | Accuracy vs. Generation Latency Pareto Frontier (Main Text §4.3) | `generate_figures.py` or `Generate_Paper_Figures.ipynb` (Cell 3) | `./figures/fig_pareto_latency_accuracy.pdf` |
| **Figure 3** | Embedding Generation Latency by Method (Main Text §4.3) | `generate_figures.py` or `Generate_Paper_Figures.ipynb` (Cell 4) | `./figures/fig_generation_time.pdf` |
| **Figure 4** | Geometric Class Separation Ratio $R = D_{\text{inter}}/D_{\text{intra}}$ (Main Text §4.4) | `generate_figures.py` or `Generate_Paper_Figures.ipynb` (Cell 7) | `./figures/fig_class_separation.pdf` |
| **Figure 5** | NyS-Binary Supervised Downstream Accuracy Heatmap (Appendix §A.5) | `generate_figures.py` or `Generate_Paper_Figures.ipynb` (Cell 5) | `./figures/fig_nys_heatmap.pdf` |
| **Figure 6** | NyS-Binary Unsupervised Downstream Accuracy Heatmap (Appendix §A.5) | `generate_figures.py` or `Generate_Paper_Figures.ipynb` (Cell 5) | `./figures/fig_nys_unsupervised_heatmap.pdf` |
| **Figure 7 / Table 11** | Diffusion Horizon Sensitivity ($H$) | `Comprehensive_Ablation_Study.ipynb` (Sec. 2.2) | `./Results/ablation_hops_*.png` |
| **Figure 8** | Factorial Grid Sweep ($\alpha$ vs. $\tau$) | `Comprehensive_Ablation_Study.ipynb` (Sec. 2.1) | `./Results/ablation_alpha_threshold_4classifiers.png` |
| **Figure 9 / Table 12** | Landmark Budget Pareto Sweep ($m$) | `Comprehensive_Ablation_Study.ipynb` (Sec. 2.3) | `./Results/ablation_nystrom_pareto_4classifiers_no_svd.png` |
| **Figure 10 / Table 13** | Structural vs. Label Dimension Split ($k_{\text{struct}} : k_{\text{label}}$) | `Comprehensive_Ablation_Study.ipynb` (Sec. 2.4) | `./Results/ablation_budget_split_tradeoff.png` |
| **Table 10** | Structural Backbone Comparison | `Comprehensive_Ablation_Study.ipynb` (Sec. 2.5) | Summary comparison table |
| **Table 14** | Scalability on `ogbn-arxiv` ($N=169{,}343$) | `benchmark_new_algebraic.py --datasets ogbn-arxiv` | Large-graph benchmark output |
| **Appendix Figures** | Downstream Accuracy by Classifier (Grouped Bars) | `generate_figures.py` or `Generate_Paper_Figures.ipynb` (Cell 6) | `./figures/fig_accuracy_by_classifier.pdf` |
