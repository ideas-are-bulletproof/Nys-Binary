# NyS-Binary: An Algebraic Framework for Binary Node Representation Learning

This repository contains the official, anonymous implementation for the paper:  
**NyS-Binary: A Feature-Free, Algebraic Framework for Binary Node Representation Learning**.

NyS-Binary generates compact, 250-bit binary node representations directly from graph topology in closed form, requiring zero neural network training, backpropagation, or learnable parameters.

---

## Overview

Graph representation learning typically relies on high-dimensional floating-point embeddings (e.g., 128 to 512 dimensions in `float32`), creating substantial memory and computational bottlenecks for downstream inference. NyS-Binary resolves this by synthesizing compact binary codes ($\mathbf{B} \in \{0, 1\}^{N \times K}$) in Hamming space that achieve up to **$32\times$ memory reduction** while matching or exceeding the classification performance of dense models.

### Key Architectural Pillars

* **Decoupled Structural Channel:** Leverages randomized Nyström sketching on the normalized graph transition operator $\mathbf{P} = \tilde{\mathbf{D}}^{-1}\tilde{\mathbf{A}}$, capturing global spectral geometry via a small landmark core ($m \ll N$) without full singular value decomposition.
* **Leakage-Safe Semantic Channel:** Diffuses training label signals across graph neighborhoods and projects pseudo-labels through an orthogonal Gaussian QR codebook under strict confidence gating ($\tau$).
* **Multi-Scale Polynomial Diffusion:** Blends structural and semantic representations via higher-order polynomial random-walk propagation $\mathcal{D}_H(\mathbf{W}) \propto \mathbf{P}^H(\mathbf{W} + \mathbf{P}\mathbf{W} + \mathbf{P}^2\mathbf{W})$.
* **Scale-Invariant Quantization:** Binarizes continuous coordinates via adaptive column-wise median thresholding, provably preserving discrete codes under positive scalar scaling.
* **Unsupervised Mode:** Can be executed in a purely unsupervised setting ($k_{\mathrm{label}} = 0$) using solely graph topology.

---

## Repository Structure

```text
├── benchmark_new_algebraic.py       # Core evaluation engine & PyG dataset loaders
├── advanced_baselines.py            # Competing binary & continuous graph baselines
├── rigorous_ablation_system.py      # Configurable hyperparameter and ablation engine
├── generate_figures.py              # Publication vector PDF and PNG figure generator
├── Main.ipynb                       # Primary benchmark notebook (10 datasets, 9 classifiers)
├── Generate_Paper_Figures.ipynb     # Interactive figure generation and inspection notebook
├── Comprehensive_Ablation_Study.ipynb # Landmark Pareto frontiers & sensitivity suites
├── figures/                         # Publication-grade figures and ablation plots
├── REPRODUCE.md                     # Comprehensive table-to-command reproduction guide
└── requirements.txt                 # Python environment dependencies
```

---

## Installation

Ensure you have Python 3.8+ installed. Install all necessary dependencies via:

```bash
pip install -r requirements.txt
```

---

## Results and Reproducibility

> [!IMPORTANT]
> **Reproducing Paper Tables, Figures, and Ablations:**  
> All experimental results, execution commands, notebook walkthroughs, and table-to-command mappings are detailed in:  
> ➡️ **[REPRODUCE.md](REPRODUCE.md)**

Please consult [`REPRODUCE.md`](REPRODUCE.md) for:
* **Tables 1 & 2:** Main 10-dataset classification accuracy and Macro-F1 across 9 classifiers.
* **Table 3:** Representation generation latency and throughput benchmarks.
* **Figures 2–5:** Accuracy heatmaps, accuracy-latency Pareto curves, generation speedups, and class separability plots.
* **Ablation Studies:** Landmark budget scaling ($m$), diffusion hops ($H$), blend weighting ($\alpha$), and low-supervision benchmarks.

---

## Anonymity & License

This repository is fully anonymized for double-blind peer review. All institutional affiliations, author identities, and tracking tokens have been removed.

Released under the [MIT License](LICENSE).
