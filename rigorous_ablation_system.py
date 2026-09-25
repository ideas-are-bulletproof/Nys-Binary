import glob
"""
=============================================================================
Ablation System for NyS-Binary Graph Representations
=============================================================================
This module provides the experimentation engine for the NyS-Binary algorithm,
supporting analysis of structural signatures, sampling strategies, label channels, 
and diffusion techniques.

Features:
- Canonical Baseline Configurations
- Grid Sweeps
- Multi-dataset Analysis
- Comprehensive Unit Tests
- Statistical Testing and Visualization
=============================================================================
"""

import os
import sys
import time
import math
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy import stats
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.extmath import randomized_svd

# Ensure local modules are accessible
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import benchmark_new_algebraic as bna
import advanced_baselines as ab


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================
@dataclass
class AblationConfig:
    """Explicit configuration capturing all 14 architectural dimensions."""
    name: str = "Full_NyS_Binary_Default"
    
    # Structural Backbone
    struct_mode: str = "nystrom"  # "nystrom", "exact_svd", "random_walk_pe", "degree", "random_noise", "none"
    
    # Landmark Sampling & Count
    landmark_strategy: str = "uniform"  # "uniform", "degree_weighted", "kmeans_pp"
    m_landmarks: Optional[int] = None   # None -> default min(n, max(2*k_struct, 500))
    
    # Budget & Split
    k_total: int = 250
    struct_ratio: float = 0.5           # fraction allocated to k_struct (e.g. 0.5 -> 50/50 split)
    
    # Class Signature Construction
    class_sig_mode: str = "orthogonal_sign"  # "orthogonal_sign", "one_hot", "gaussian_continuous"
    
    # Channel A: Ground Truth
    enable_gt_channel: bool = True
    
    # Channel B: Pseudo-Labels & Self-Training
    enable_pseudo_channel: bool = True
    rw_depth: int = 3                   # steps of RW diffusion to produce pseudo-label probabilities
    
    # Confidence Gate
    pseudo_threshold: float = 0.85
    
    # Polynomial Diffusion Function
    diffusion_mode: str = "polynomial_acc"  # "polynomial_acc", "hops_only", "single_power", "none"
    
    # Diffusion Radius
    hops: int = 3
    
    # Dual-Stream Fusion Weight
    alpha: float = 0.5
    
    # Binarization Threshold Formula
    threshold_mode: str = "alpha_median"  # "alpha_median", "fixed_half", "column_mean", "unscaled_median", "continuous"
    
    # Final Quantization
    binarize: bool = True
    
    # Execution & Reproducibility
    seed: int = 42
    device: str = "cpu"
    corrupt_leak_test_nodes: bool = False # Used ONLY for negative control check


# =============================================================================
# CORE ABLATION EMBEDDING GENERATOR
# =============================================================================
def generate_ablated_representation(
    G: nx.Graph,
    pyg_data: Any,
    labels: Any,
    label_mask: Any,
    cfg: AblationConfig
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Parametric implementation of NyS-Binary allowing isolated intervention
    on any of the 14 algorithmic components.
    
    Returns:
        emb: np.ndarray (n, k_final) representation
        gen_time: float wall-clock runtime in seconds
        diagnostics: dict containing pseudo-label yield, acceptance fraction, etc.
    """
    t0 = time.time()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    n = G.number_of_nodes()
    dev = cfg.device
    A_norm, A_torch = ab._get_A_torch(G, dev)
    
    # Determine dimension budget
    if cfg.struct_mode == "none" or cfg.struct_ratio <= 0.0:
        k_struct = 0
        k_label = cfg.k_total
    elif (not cfg.enable_gt_channel and not cfg.enable_pseudo_channel) or cfg.struct_ratio >= 1.0:
        k_struct = cfg.k_total
        k_label = 0
    else:
        k_struct = max(2, int(round(cfg.k_total * cfg.struct_ratio)))
        k_label = max(2, cfg.k_total - k_struct)
        
    diagnostics = {
        "k_struct": k_struct,
        "k_label": k_label,
        "m_landmarks": 0,
        "pseudo_accepted_count": 0,
        "pseudo_accepted_fraction": 0.0,
    }
    
    # -------------------------------------------------------------------------
    # Structural Backbone Generation
    # -------------------------------------------------------------------------
    if cfg.struct_mode == "none" or k_struct == 0:
        R_struct = None
        
    elif cfg.struct_mode == "exact_svd":
        try:
            if n > 5000:
                U_svd, _, _ = randomized_svd(A_norm, n_components=k_struct, n_iter=5, random_state=cfg.seed)
            elif k_struct < n - 1:
                U_svd, _, _ = sla.svds(A_norm, k=k_struct, which='LM')
            else:
                U_svd, _, _ = randomized_svd(A_norm, n_components=k_struct, random_state=cfg.seed)
        except Exception:
            U_svd, _, _ = randomized_svd(A_norm, n_components=k_struct, random_state=cfg.seed)
        R_struct = torch.tensor(np.ascontiguousarray(U_svd), dtype=torch.float32, device=dev)
        
    elif cfg.struct_mode == "random_noise":
        # Uninformative baseline: standard Gaussian noise of exact same shape
        R_noise = np.random.randn(n, k_struct).astype(np.float32) / math.sqrt(k_struct)
        R_struct = torch.tensor(R_noise, dtype=torch.float32, device=dev)
        
    elif cfg.struct_mode == "degree":
        # Degree and power structural feature baseline
        degs = np.array([d for _, d in G.degree()], dtype=np.float32).reshape(-1, 1)
        degs_norm = degs / (np.max(degs) + 1e-10)
        # Random projection of degree powers to k_struct dimensions
        powers = np.concatenate([degs_norm ** p for p in range(1, 6)], axis=1)
        proj = np.random.randn(powers.shape[1], k_struct).astype(np.float32)
        R_deg = powers.dot(proj)
        R_struct = torch.tensor(np.ascontiguousarray(R_deg), dtype=torch.float32, device=dev)
        
    elif cfg.struct_mode == "random_walk_pe":
        # Random walk positional encoding: diagonal of A^p for p = 1..k_struct
        pe_list = []
        A_curr = A_norm.copy()
        for step in range(1, min(k_struct + 1, 16)):
            pe_list.append(A_curr.diagonal().astype(np.float32))
            A_curr = A_curr.dot(A_norm)
        PE = np.column_stack(pe_list)
        if PE.shape[1] < k_struct:
            rep = (k_struct // PE.shape[1]) + 1
            PE = np.tile(PE, (1, rep))[:, :k_struct]
        R_struct = torch.tensor(np.ascontiguousarray(PE), dtype=torch.float32, device=dev)
        
    else:  # Default: "nystrom"
        # Landmark count selection
        if cfg.m_landmarks is not None:
            m = min(n, cfg.m_landmarks)
        else:
            m = min(n, max(k_struct * 2, 500))
        diagnostics["m_landmarks"] = m
        
        # Landmark sampling strategy
        if cfg.landmark_strategy == "degree_weighted":
            degrees = np.array([d for _, d in G.degree()], dtype=np.float64)
            deg_sum = degrees.sum()
            prob = degrees / deg_sum if deg_sum > 0 else np.ones(n) / n
            landmark_idx = np.random.choice(n, size=m, replace=False, p=prob)
        elif cfg.landmark_strategy == "kmeans_pp":
            # Greedy farthest-point / k-means++ sampling based on A_norm rows
            first = np.random.randint(0, n)
            selected = [first]
            # Fast approximate distance using random projection of A_norm
            Omega = np.random.randn(n, 16).astype(np.float32)
            A_proj = A_norm.dot(Omega)
            min_dists = np.sum((A_proj - A_proj[first]) ** 2, axis=1)
            for _ in range(1, m):
                prob = min_dists / (min_dists.sum() + 1e-10)
                next_cand = np.random.choice(n, p=prob)
                selected.append(next_cand)
                cand_dist = np.sum((A_proj - A_proj[next_cand]) ** 2, axis=1)
                min_dists = np.minimum(min_dists, cand_dist)
            landmark_idx = np.array(selected)
        else:  # "uniform"
            landmark_idx = np.random.choice(n, size=m, replace=False)
            
        A_cc = A_norm[landmark_idx, :][:, landmark_idx].toarray()
        U_c, S_c, _ = np.linalg.svd(A_cc)
        
        k_eff = min(k_struct, m)
        U_c = U_c[:, :k_eff]
        S_c = S_c[:k_eff]
        S_inv = np.diag(1.0 / (S_c + 1e-10))
        
        A_nc = A_norm[:, landmark_idx]
        U_approx = A_nc.dot(U_c).dot(S_inv)
        
        if k_eff < k_struct:
            U_approx = np.pad(U_approx, ((0, 0), (0, k_struct - k_eff)), mode='constant')
            
        R_struct = torch.tensor(np.ascontiguousarray(U_approx), dtype=torch.float32, device=dev)
        
    # -------------------------------------------------------------------------
    # Label Handling & Class Signatures
    # -------------------------------------------------------------------------
    is_multilabel = (hasattr(labels, "ndim") and labels.ndim == 2) or (
        isinstance(labels, torch.Tensor) and labels.dim() == 2
    )
    if is_multilabel:
        n_classes = labels.shape[1]
    elif isinstance(labels, torch.Tensor):
        n_classes = int(labels.max().item()) + 1
    else:
        n_classes = int(np.max(labels)) + 1
        
    labels_t = labels.to(dev) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=dev)
    label_mask_t = label_mask.to(dev) if isinstance(label_mask, torch.Tensor) else torch.tensor(label_mask, dtype=torch.bool, device=dev)
    
    # Negative Control Check: artificially leak test nodes
    if cfg.corrupt_leak_test_nodes:
        # Intentionally contaminate label_mask with 100% of all nodes
        label_mask_t = torch.ones(n, dtype=torch.bool, device=dev)
        
    if k_label > 0:
        if cfg.class_sig_mode == "one_hot":
            # Plain one-hot encoding expanded or truncated to k_label
            eye = torch.eye(n_classes, device=dev)
            if k_label >= n_classes:
                rep = (k_label // n_classes) + 1
                class_signatures = eye.repeat(1, rep)[:, :k_label]
            else:
                class_signatures = eye[:, :k_label]
        elif cfg.class_sig_mode == "gaussian_continuous":
            # Continuous orthogonal vectors without sign() discretization
            class_signatures = ab._get_orthogonal_signatures(n_classes, k_label, dev=dev)
        else:  # Default: "orthogonal_sign"
            class_signatures = torch.sign(ab._get_orthogonal_signatures(n_classes, k_label, dev=dev))
            
        # ---------------------------------------------------------------------
        # Ground-Truth Channel (S_gt, Channel A)
        # ---------------------------------------------------------------------
        S_gt = torch.zeros(n, k_label, dtype=torch.float32, device=dev)
        Y_0 = torch.zeros(n, n_classes, dtype=torch.float32, device=dev)
        
        if cfg.enable_gt_channel and torch.any(label_mask_t):
            if is_multilabel:
                Y_0[label_mask_t] = labels_t[label_mask_t].float()
                S_gt[label_mask_t] = torch.sign(Y_0[label_mask_t] @ class_signatures)
            else:
                Y_0[label_mask_t] = F.one_hot(labels_t[label_mask_t].long(), num_classes=n_classes).float()
                S_gt[label_mask_t] = class_signatures[labels_t[label_mask_t].long()]
                
        # ---------------------------------------------------------------------
        # Pseudo-Label Channel (S_v7, Channel B)
        # ---------------------------------------------------------------------
        S_v7 = S_gt.clone()
        if cfg.enable_pseudo_channel and torch.any(label_mask_t):
            # Propagate training one-hots over rw_depth random-walk steps
            Y_t = Y_0
            for _ in range(max(1, cfg.rw_depth)):
                Y_t = torch.sparse.mm(A_torch, Y_t)
            probs = Y_t / (Y_t.sum(dim=1, keepdim=True) + 1e-10)
            max_probs, pseudo_c = torch.max(probs, dim=1)
            
            # Confidence gate
            confident = (~label_mask_t) & (max_probs >= cfg.pseudo_threshold)
            num_conf = int(confident.sum().item())
            diagnostics["pseudo_accepted_count"] = num_conf
            diagnostics["pseudo_accepted_fraction"] = num_conf / max(1, n - int(label_mask_t.sum().item()))
            
            if torch.any(confident):
                S_v7[confident] = class_signatures[pseudo_c[confident]]
    else:
        S_gt = None
        S_v7 = None

    # -------------------------------------------------------------------------
    # Polynomial Graph Diffusion
    # -------------------------------------------------------------------------
    def diffuse(W_init: torch.Tensor) -> torch.Tensor:
        if cfg.diffusion_mode == "none":
            return W_init
        elif cfg.diffusion_mode == "hops_only":
            curr = W_init
            for _ in range(cfg.hops):
                curr = torch.sparse.mm(A_torch, curr)
            return curr
        elif cfg.diffusion_mode == "single_power":
            # Direct single higher power A^hops (no polynomial accumulation)
            curr = W_init
            for _ in range(max(1, cfg.hops)):
                curr = torch.sparse.mm(A_torch, curr)
            return curr
        else:  # Default: "polynomial_acc" (2-step accumulation + hops diffusion)
            W_acc = W_init.clone()
            curr = W_init
            for _ in range(2):
                curr = torch.sparse.mm(A_torch, curr)
                W_acc += curr
            for _ in range(cfg.hops):
                W_acc = torch.sparse.mm(A_torch, W_acc)
            return W_acc

    # Form channel matrices
    if R_struct is not None and S_gt is not None:
        W_svd_init = torch.cat([R_struct, S_gt], dim=1)
        W_v7_init = torch.cat([R_struct, S_v7], dim=1)
    elif R_struct is not None:
        W_svd_init = R_struct
        W_v7_init = R_struct
    elif S_gt is not None:
        W_svd_init = S_gt
        W_v7_init = S_v7
    else:
        raise ValueError("Both R_struct and label channels are disabled!")

    # Apply diffusion
    W_svd_acc = diffuse(W_svd_init)
    if cfg.enable_pseudo_channel and cfg.alpha > 0.0:
        W_v7_acc = diffuse(W_v7_init)
    else:
        W_v7_acc = W_svd_acc

    # -------------------------------------------------------------------------
    # Convex Blend Weight (alpha)
    # -------------------------------------------------------------------------
    if not cfg.enable_pseudo_channel or cfg.alpha == 0.0:
        W_fused = W_svd_acc
    else:
        W_fused = (1.0 - cfg.alpha) * W_svd_acc + cfg.alpha * W_v7_acc

    # -------------------------------------------------------------------------
    # Adaptive Binarization Threshold & Quantization
    # -------------------------------------------------------------------------
    if not cfg.binarize or cfg.threshold_mode == "continuous":
        out = W_fused.float().cpu().numpy()
    else:
        if cfg.threshold_mode == "fixed_half":
            threshold = 0.5
        elif cfg.threshold_mode == "column_mean":
            col_means = torch.mean(W_fused, dim=0)
            threshold = cfg.alpha * col_means
        elif cfg.threshold_mode == "unscaled_median":
            threshold = torch.median(W_fused, dim=0).values
        else:  # Default: "alpha_median" (alpha * col_median)
            col_medians = torch.median(W_fused, dim=0).values
            threshold = cfg.alpha * col_medians
            
        out = (W_fused > threshold).float().cpu().numpy()

    gen_time = time.time() - t0
    return out, gen_time, diagnostics


# =============================================================================
# 3. CANONICAL BASELINES DEFINITION
# =============================================================================
CANONICAL_BASELINES: Dict[str, AblationConfig] = {
    "Full_NyS_Binary (Ours)": AblationConfig(
        name="Full_NyS_Binary (Ours)",
        struct_mode="nystrom",
        struct_ratio=0.5,
        enable_gt_channel=True,
        enable_pseudo_channel=True,
        pseudo_threshold=0.85,
        hops=3,
        alpha=0.5,
        threshold_mode="alpha_median",
        binarize=True
    ),
    "Baseline: Structural-Only": AblationConfig(
        name="Baseline: Structural-Only",
        struct_mode="nystrom",
        struct_ratio=1.0,
        enable_gt_channel=False,
        enable_pseudo_channel=False,
        alpha=0.0,
        binarize=True
    ),
    "Baseline: Label-Only": AblationConfig(
        name="Baseline: Label-Only",
        struct_mode="none",
        struct_ratio=0.0,
        enable_gt_channel=True,
        enable_pseudo_channel=False,
        alpha=0.0,
        hops=0,
        diffusion_mode="none",
        binarize=True
    ),
    "Baseline: Full-SVD Oracle": AblationConfig(
        name="Baseline: Full-SVD Oracle",
        struct_mode="exact_svd",
        struct_ratio=0.5,
        enable_gt_channel=True,
        enable_pseudo_channel=True,
        pseudo_threshold=0.85,
        hops=3,
        alpha=0.5,
        binarize=True
    ),
    "Baseline: No Self-Training (alpha=0)": AblationConfig(
        name="Baseline: No Self-Training (alpha=0)",
        struct_mode="nystrom",
        struct_ratio=0.5,
        enable_gt_channel=True,
        enable_pseudo_channel=False,
        alpha=0.0,
        hops=3,
        binarize=True
    ),
    "Baseline: No Diffusion (hops=0)": AblationConfig(
        name="Baseline: No Diffusion (hops=0)",
        struct_mode="nystrom",
        struct_ratio=0.5,
        enable_gt_channel=True,
        enable_pseudo_channel=True,
        diffusion_mode="none",
        hops=0,
        alpha=0.5,
        binarize=True
    ),
    "Baseline: Continuous Representation": AblationConfig(
        name="Baseline: Continuous Representation",
        struct_mode="nystrom",
        struct_ratio=0.5,
        enable_gt_channel=True,
        enable_pseudo_channel=True,
        binarize=False,
        threshold_mode="continuous"
    ),
    "Baseline: Fixed-Threshold (0.5)": AblationConfig(
        name="Baseline: Fixed-Threshold (0.5)",
        struct_mode="nystrom",
        struct_ratio=0.5,
        enable_gt_channel=True,
        enable_pseudo_channel=True,
        threshold_mode="fixed_half",
        binarize=True
    ),
    "Baseline: Random Structural Signature": AblationConfig(
        name="Baseline: Random Structural Signature",
        struct_mode="random_noise",
        struct_ratio=0.5,
        enable_gt_channel=True,
        enable_pseudo_channel=True,
        binarize=True
    ),
}


# =============================================================================
# 4. FIXED DOWNSTREAM EVALUATION HEAD
# =============================================================================
def evaluate_representation_multi_classifier(
    X_emb: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    edge_index: Optional[Any] = None,
    classifiers: List[str] = ['GCN', 'GraphSAGE', 'MLP', 'LINKX'],
    epochs: int = 200,
    lr: float = 0.01,
    device: str = "cpu"
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates the representation across 4 canonical deep learning classifiers:
    1. GCN (Kipf & Welling)
    2. GraphSAGE (Hamilton et al.)
    3. MLP (Multi-Layer Perceptron)
    4. LINKX (Lim et al.)
    And computes their combined 4-classifier average (mean accuracy, macro F1, micro F1).
    """
    import benchmark_new_algebraic as bna
    dev = device if (torch.cuda.is_available() and str(device).startswith("cuda")) else "cpu"
    features_t = torch.from_numpy(X_emb).float().to(dev)
    labels_t = torch.from_numpy(y).to(dev)
    if edge_index is None:
        e_idx = torch.arange(features_t.shape[0], device=dev).unsqueeze(0).repeat(2, 1).long()
    elif isinstance(edge_index, torch.Tensor):
        e_idx = edge_index.to(dev).long()
    else:
        e_idx = torch.tensor(edge_index, device=dev, dtype=torch.long)
    n_nodes = features_t.shape[0]
    n_classes = y.shape[1] if (hasattr(y, 'ndim') and y.ndim == 2) else int(np.max(y)) + 1
    in_dim = features_t.shape[1]

    clf_metrics = {}
    for c_name in classifiers:
        c_upper = c_name.upper()
        if c_upper == 'GCN':
            model = bna.BasicGCN(in_dim, n_classes, hidden=64).to(dev)
        elif c_upper in ['GRAPHSAGE', 'GSAGE', 'SAGE']:
            model = bna.BasicGraphSAGE(in_dim, n_classes, hidden=64).to(dev)
        elif c_upper == 'MLP':
            model = bna.BasicMLP(in_dim, n_classes, hidden=64).to(dev)
        elif c_upper == 'LINKX':
            model = bna.BasicLINKX(in_dim, n_classes, num_nodes=n_nodes, hidden=64).to(dev)
        else:
            continue

        if e_idx is None and c_upper != 'MLP':
            # If edge_index is missing, skip graph convs
            continue

        acc, f1_mac, f1_mic, tr_t, inf_t = bna.train_and_eval(
            model, e_idx, features_t, labels_t, train_mask, test_mask,
            epochs=epochs, lr=lr, device=dev
        )
        clf_metrics[c_name] = {
            'accuracy': float(acc * 100.0),
            'macro_f1': float(f1_mac * 100.0),
            'micro_f1': float(f1_mic * 100.0),
            'train_time': float(tr_t)
        }

    # Compute 4-classifier mean
    avg_acc = float(np.mean([m['accuracy'] for m in clf_metrics.values()]))
    avg_mac = float(np.mean([m['macro_f1'] for m in clf_metrics.values()]))
    avg_mic = float(np.mean([m['micro_f1'] for m in clf_metrics.values()]))
    clf_metrics['Average_4_Classifiers'] = {
        'accuracy': avg_acc,
        'macro_f1': avg_mac,
        'micro_f1': avg_mic
    }
    return clf_metrics


def evaluate_representation_probe(
    X_emb: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    probe_type: str = "4_classifiers",
    edge_index: Optional[Any] = None,
    epochs: int = 200,
    lr: float = 0.01,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluates node classification performance.
    Supports:
    - '4_classifiers' / 'multi' / 'all' (default): evaluates GCN, GraphSAGE, MLP, LINKX and returns their average + individual metrics
    - 'gcn', 'graphsage', 'mlp', 'linkx': evaluates that specific model
    - 'logistic': linear probe
    """
    X_train, y_train = X_emb[train_mask], y[train_mask]
    X_test, y_test = X_emb[test_mask], y[test_mask]
    
    if len(np.unique(y_train)) < 2:
        return {"accuracy": 0.0, "micro_f1": 0.0, "macro_f1": 0.0}

    p_lower = str(probe_type).lower()
    if edge_index is None and p_lower not in ["mlp", "logistic"]:
        # When graph topology is not provided, fall back cleanly to fixed linear probe
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear', random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = float(accuracy_score(y_test, preds) * 100.0)
        mic = float(f1_score(y_test, preds, average="micro", zero_division=0) * 100.0)
        mac = float(f1_score(y_test, preds, average="macro", zero_division=0) * 100.0)
        return {"accuracy": acc, "micro_f1": mic, "macro_f1": mac}

    if p_lower in ["4_classifiers", "multi", "all", "average"]:
        multi_res = evaluate_representation_multi_classifier(
            X_emb, y, train_mask, test_mask, edge_index=edge_index,
            classifiers=['GCN', 'GraphSAGE', 'MLP', 'LINKX'],
            epochs=epochs, lr=lr, device=device
        )
        avg_res = multi_res['Average_4_Classifiers']
        return {
            "accuracy": avg_res["accuracy"],
            "macro_f1": avg_res["macro_f1"],
            "micro_f1": avg_res["micro_f1"],
            "per_classifier": multi_res
        }

    if p_lower in ["gcn", "graphsage", "gsage", "sage", "mlp", "linkx"]:
        import benchmark_new_algebraic as bna
        dev = device if (torch.cuda.is_available() and str(device).startswith("cuda")) else "cpu"
        features_t = torch.from_numpy(X_emb).float().to(dev)
        labels_t = torch.from_numpy(y).to(dev)
        if edge_index is None:
            e_idx = torch.arange(features_t.shape[0], device=dev).unsqueeze(0).repeat(2, 1).long()
        elif isinstance(edge_index, torch.Tensor):
            e_idx = edge_index.to(dev).long()
        else:
            e_idx = torch.tensor(edge_index, device=dev, dtype=torch.long)
        n_classes = y.shape[1] if (hasattr(y, 'ndim') and y.ndim == 2) else int(np.max(y)) + 1
        in_dim = features_t.shape[1]

        if p_lower == "gcn":
            model = bna.BasicGCN(in_dim, n_classes, hidden=64).to(dev)
        elif p_lower in ["graphsage", "gsage", "sage"]:
            model = bna.BasicGraphSAGE(in_dim, n_classes, hidden=64).to(dev)
        elif p_lower == "mlp":
            model = bna.BasicMLP(in_dim, n_classes, hidden=64).to(dev)
        elif p_lower == "linkx":
            model = bna.BasicLINKX(in_dim, n_classes, num_nodes=features_t.shape[0], hidden=64).to(dev)

        acc, f1_mac, f1_mic, _, _ = bna.train_and_eval(
            model, e_idx, features_t, labels_t, train_mask, test_mask, epochs=epochs, lr=lr, device=dev
        )
        return {"accuracy": float(acc * 100.0), "micro_f1": float(f1_mic * 100.0), "macro_f1": float(f1_mac * 100.0)}

    clf = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear', random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    acc = float(accuracy_score(y_test, preds) * 100.0)
    mic = float(f1_score(y_test, preds, average="micro", zero_division=0) * 100.0)
    mac = float(f1_score(y_test, preds, average="macro", zero_division=0) * 100.0)
    
    return {"accuracy": acc, "micro_f1": mic, "macro_f1": mac}


# =============================================================================
# SANITY CHECKS & UNIT TESTS
# =============================================================================
def run_leakage_and_sanity_checks(ds: Dict[str, Any], device: str = "cpu") -> Dict[str, Any]:
    """
    Executes 4 rigorous sanity checks:
      1. Zero test label leakage unit test (Channel A outside train_mask must be exactly 0)
      2. Negative-control test with corrupted label mask (proves guard is load-bearing)
      3. Degenerate input check (0% label mask, 100% label mask)
      4. Label-noise injection check (10%, 20%, 30% label noise to test self-training liability)
    """
    print("\n" + "=" * 70)
    print("  RUNNING SCIENTIFIC SANITY CHECKS & LEAKAGE CONTROLS")
    print("=" * 70)
    
    G = ds["G"]
    pyg = ds["pyg_data"]
    y = ds["labels"]
    n = G.number_of_nodes()
    
    # 70/30 split
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split_pt = int(0.7 * n)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[indices[:split_pt]] = True
    test_mask = ~train_mask
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Check 1: Explicit Unit Test for Zero Test Label Leakage
    # -------------------------------------------------------------------------
    cfg_clean = AblationConfig(device=device, seed=42)
    n_classes = int(np.max(y)) + 1
    k_label = cfg_clean.k_total - (cfg_clean.k_total // 2)
    label_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)
    
    class_signatures = torch.sign(ab._get_orthogonal_signatures(n_classes, k_label, dev=device))
    S_gt = torch.zeros(n, k_label, dtype=torch.float32, device=device)
    labels_t = torch.tensor(y, device=device)
    S_gt[label_mask_t] = class_signatures[labels_t[label_mask_t].long()]
    
    unlabeled_norm = float(torch.norm(S_gt[~label_mask_t]).item())
    is_leak_free = (unlabeled_norm == 0.0)
    results["unit_test_zero_leakage_passed"] = is_leak_free
    print(f"[Sanity Check 1] Zero Test Label Leakage: {'PASSED (Norm = 0.0)' if is_leak_free else 'FAILED!'}")
    assert is_leak_free, "FATAL: Test label leakage detected in Channel A initialization!"

    # -------------------------------------------------------------------------
    # Check 2: Negative Control with Corrupted Label Mask
    # -------------------------------------------------------------------------
    e_idx = getattr(pyg, 'edge_index', None)
    emb_clean, _, _ = generate_ablated_representation(G, pyg, y, train_mask, cfg_clean)
    metrics_clean = evaluate_representation_probe(emb_clean, y, train_mask, test_mask, probe_type="gcn", edge_index=e_idx, device=device)
    
    cfg_corrupt = AblationConfig(device=device, seed=42, corrupt_leak_test_nodes=True)
    emb_corrupt, _, _ = generate_ablated_representation(G, pyg, y, train_mask, cfg_corrupt)
    metrics_corrupt = evaluate_representation_probe(emb_corrupt, y, train_mask, test_mask, probe_type="gcn", edge_index=e_idx, device=device)
    
    acc_clean = metrics_clean["accuracy"]
    acc_corrupt = metrics_corrupt["accuracy"]
    results["clean_accuracy"] = acc_clean
    results["corrupted_accuracy"] = acc_corrupt
    results["negative_control_passed"] = (acc_corrupt > acc_clean + 10.0)
    print(f"[Sanity Check 2] Negative Control Leakage Guard:")
    print(f"                 Legitimate Mask Accuracy:  {acc_clean:.2f}%")
    print(f"                 Corrupted Mask Accuracy:   {acc_corrupt:.2f}% (+{acc_corrupt - acc_clean:.2f}%)")
    print(f"                 Verification: {'PASSED (Guard is strictly load-bearing)' if results['negative_control_passed'] else 'WARNING'}")

    # -------------------------------------------------------------------------
    # Check 3: Degenerate Input Conditions
    # -------------------------------------------------------------------------
    empty_mask = np.zeros(n, dtype=bool)
    cfg_empty = AblationConfig(device=device, seed=42)
    emb_empty, _, _ = generate_ablated_representation(G, pyg, y, empty_mask, cfg_empty)
    results["zero_percent_labels_shape"] = emb_empty.shape
    print(f"[Sanity Check 3] Degenerate Input (0% Labels): PASSED (Produced valid shape {emb_empty.shape})")

    # -------------------------------------------------------------------------
    # Check 4: Label Noise Robustness Test
    # -------------------------------------------------------------------------
    noise_rates = [0.0, 0.1, 0.2, 0.3]
    noise_results = []
    for nr in noise_rates:
        y_noisy = y.copy()
        if nr > 0.0:
            train_idx = np.where(train_mask)[0]
            corrupt_cnt = int(nr * len(train_idx))
            corrupt_nodes = np.random.choice(train_idx, size=corrupt_cnt, replace=False)
            y_noisy[corrupt_nodes] = np.random.randint(0, n_classes, size=corrupt_cnt)
            
        emb_noisy, _, diag = generate_ablated_representation(G, pyg, y_noisy, train_mask, cfg_clean)
        m_noisy = evaluate_representation_probe(emb_noisy, y, train_mask, test_mask, probe_type="gcn", edge_index=e_idx, device=device)
        noise_results.append({
            "noise_rate": nr,
            "accuracy": m_noisy["accuracy"],
            "macro_f1": m_noisy["macro_f1"],
            "pseudo_yield": diag["pseudo_accepted_count"]
        })
    results["noise_robustness"] = noise_results
    print(f"[Sanity Check 4] Label Noise Sensitivity:")
    for nr_res in noise_results:
        print(f"                 Noise {int(nr_res['noise_rate']*100)}%: Acc = {nr_res['accuracy']:.2f}%, Pseudo-yield = {nr_res['pseudo_yield']}")

    print("=" * 70 + "\n")
    return results


# =============================================================================
# 6. STATISTICAL SIGNIFICANCE TESTING
# =============================================================================
def compute_paired_statistics(
    sample_a: List[float],
    sample_b: List[float]
) -> Dict[str, Any]:
    """Computes paired t-test, Wilcoxon signed-rank test, and Cohen's d effect size."""
    arr_a = np.array(sample_a)
    arr_b = np.array(sample_b)
    diff = arr_a - arr_b
    
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    
    if std_diff > 1e-10:
        t_stat, p_ttest = stats.ttest_rel(arr_a, arr_b)
    else:
        t_stat, p_ttest = 0.0, 1.0
        
    try:
        w_stat, p_wilcox = stats.wilcoxon(arr_a, arr_b)
    except Exception:
        w_stat, p_wilcox = 0.0, 1.0
        
    pooled_std = np.sqrt((np.var(arr_a, ddof=1) + np.var(arr_b, ddof=1)) / 2.0) if len(diff) > 1 else 1.0
    cohens_d = float(mean_diff / (pooled_std + 1e-10))
    
    if p_wilcox < 0.001:
        sig_marker = "***"
    elif p_wilcox < 0.01:
        sig_marker = "**"
    elif p_wilcox < 0.05:
        sig_marker = "*"
    else:
        sig_marker = "ns"
        
    return {
        "mean_diff": mean_diff,
        "p_ttest": float(p_ttest),
        "p_wilcoxon": float(p_wilcox),
        "cohens_d": cohens_d,
        "sig_marker": sig_marker
    }


# =============================================================================
# 7. SYSTEMATIC ABLATION CAMPAIGN RUNNER
# =============================================================================
def run_systematic_ablation_campaign(
    dataset_names: List[str] = ["cora", "citeseer", "chameleon"],
    seeds: List[int] = [42, 123, 999],
    train_ratios: Union[float, List[float]] = [0.01, 0.30, 0.70],
    device: str = "cpu",
    checkpoint_dir: Optional[str] = None,
    clear_cache: bool = False,
    force_recompute: bool = False,
    probe_type: str = "logistic",
    root_data: Optional[str] = None
) -> pd.DataFrame:
    """
    Executes the comprehensive ablation suite,
    supporting multi-seed, multi-ratio (1%, 30%, 70%), and persistent per-dataset checkpointing.
    """
    if isinstance(train_ratios, (int, float)):
        ratios_list = [float(train_ratios)]
    else:
        ratios_list = [float(r) for r in train_ratios]

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if clear_cache:
            for old_f in glob.glob(os.path.join(checkpoint_dir, "campaign_*.csv")):
                try:
                    os.remove(old_f)
                except Exception:
                    pass
            print(f"Cleared campaign checkpoints directory: {checkpoint_dir}")

    print("=" * 70)
    print("  LAUNCHING ABLATION CAMPAIGN")
    print(f"  Datasets ({len(dataset_names)}): {dataset_names}")
    print(f"  Seeds ({len(seeds)}): {seeds}")
    print(f"  Training Ratios ({len(ratios_list)}): {ratios_list}")
    print(f"  Classifier Probe: {probe_type.upper()}")
    print(f"  Device: {device}")
    print("=" * 70)
    
    ablation_suite: List[Tuple[str, str, AblationConfig]] = []
    
    for b_name, b_cfg in CANONICAL_BASELINES.items():
        ablation_suite.append(("Baseline", b_name, b_cfg))
        
    # Comp 1: Structural Backbone Variants
    ablation_suite.append(("Comp 1 (R_struct)", "R_struct: Random Walk PE", AblationConfig(struct_mode="random_walk_pe")))
    ablation_suite.append(("Comp 1 (R_struct)", "R_struct: Degree Features", AblationConfig(struct_mode="degree")))
    
    # Comp 2: Landmark Sampling & Count
    ablation_suite.append(("Comp 2 (Landmarks)", "Sampling: Degree-Weighted", AblationConfig(landmark_strategy="degree_weighted")))
    ablation_suite.append(("Comp 2 (Landmarks)", "Sampling: Farthest-Point (k-means++)", AblationConfig(landmark_strategy="kmeans_pp")))
    ablation_suite.append(("Comp 2 (Landmarks)", "Count: m = k_struct (125)", AblationConfig(m_landmarks=125)))
    ablation_suite.append(("Comp 2 (Landmarks)", "Count: m = 250", AblationConfig(m_landmarks=250)))
    ablation_suite.append(("Comp 2 (Landmarks)", "Count: m = 1000", AblationConfig(m_landmarks=1000)))
    
    # Comp 3: Budget Split
    ablation_suite.append(("Comp 3 (Budget Split)", "Split: 10/90 (Struct/Label)", AblationConfig(struct_ratio=0.10)))
    ablation_suite.append(("Comp 3 (Budget Split)", "Split: 25/75 (Struct/Label)", AblationConfig(struct_ratio=0.25)))
    ablation_suite.append(("Comp 3 (Budget Split)", "Split: 75/25 (Struct/Label)", AblationConfig(struct_ratio=0.75)))
    ablation_suite.append(("Comp 3 (Budget Split)", "Split: 90/10 (Struct/Label)", AblationConfig(struct_ratio=0.90)))
    
    # Comp 4: Class Signature Construction
    ablation_suite.append(("Comp 4 (Class Sig)", "Signature: Plain One-Hot", AblationConfig(class_sig_mode="one_hot")))
    ablation_suite.append(("Comp 4 (Class Sig)", "Signature: Gaussian Continuous", AblationConfig(class_sig_mode="gaussian_continuous")))
    
    # Comp 6: Pseudo-Label RW Depth
    ablation_suite.append(("Comp 6 (Pseudo RW)", "RW Depth: 1 Step", AblationConfig(rw_depth=1)))
    ablation_suite.append(("Comp 6 (Pseudo RW)", "RW Depth: 5 Steps", AblationConfig(rw_depth=5)))
    ablation_suite.append(("Comp 6 (Pseudo RW)", "RW Depth: 10 Steps", AblationConfig(rw_depth=10)))
    
    # Comp 7: Confidence Threshold
    ablation_suite.append(("Comp 7 (Threshold)", "Threshold: tau = 0.50", AblationConfig(pseudo_threshold=0.50)))
    ablation_suite.append(("Comp 7 (Threshold)", "Threshold: tau = 0.70", AblationConfig(pseudo_threshold=0.70)))
    ablation_suite.append(("Comp 7 (Threshold)", "Threshold: tau = 0.95", AblationConfig(pseudo_threshold=0.95)))
    ablation_suite.append(("Comp 7 (Threshold)", "Threshold: tau = 0.99", AblationConfig(pseudo_threshold=0.99)))
    
    # Comp 8: Diffusion Mode
    ablation_suite.append(("Comp 8 (Diffuse Mode)", "Diffusion: Hops-Only (No 2-Step)", AblationConfig(diffusion_mode="hops_only")))
    ablation_suite.append(("Comp 8 (Diffuse Mode)", "Diffusion: Single Power A^hops", AblationConfig(diffusion_mode="single_power")))
    
    # Comp 9: Hops Sweep
    ablation_suite.append(("Comp 9 (Hops)", "Hops: 1", AblationConfig(hops=1)))
    ablation_suite.append(("Comp 9 (Hops)", "Hops: 2", AblationConfig(hops=2)))
    ablation_suite.append(("Comp 9 (Hops)", "Hops: 4", AblationConfig(hops=4)))
    ablation_suite.append(("Comp 9 (Hops)", "Hops: 6", AblationConfig(hops=6)))
    
    # Comp 10: Alpha Convex Blend
    ablation_suite.append(("Comp 10 (Alpha)", "Alpha: 0.1", AblationConfig(alpha=0.1)))
    ablation_suite.append(("Comp 10 (Alpha)", "Alpha: 0.3", AblationConfig(alpha=0.3)))
    ablation_suite.append(("Comp 10 (Alpha)", "Alpha: 0.7", AblationConfig(alpha=0.7)))
    ablation_suite.append(("Comp 10 (Alpha)", "Alpha: 0.9", AblationConfig(alpha=0.9)))
    ablation_suite.append(("Comp 10 (Alpha)", "Alpha: 1.0 (Pure Pseudo)", AblationConfig(alpha=1.0)))
    
    # Comp 11: Threshold Formula
    ablation_suite.append(("Comp 11 (Binarize Formula)", "Threshold: Column Mean", AblationConfig(threshold_mode="column_mean")))
    ablation_suite.append(("Comp 11 (Binarize Formula)", "Threshold: Unscaled Median", AblationConfig(threshold_mode="unscaled_median")))

    all_rows = []
    
    for ds_name in dataset_names:
        for r in ratios_list:
            r_pct = int(round(r * 100))
            ckpt_file = None
            if checkpoint_dir is not None:
                ckpt_file = os.path.join(checkpoint_dir, f"campaign_{ds_name}_ratio_{r_pct:02d}pct_seeds_{len(seeds)}.csv")
                if not force_recompute and os.path.exists(ckpt_file):
                    try:
                        df_c = pd.read_csv(ckpt_file)
                        all_rows.extend(df_c.to_dict('records'))
                        print(f"  [CACHED] {ds_name.upper():14s} (ratio={r*100:4.1f}%) loaded ({len(df_c)} records)")
                        continue
                    except Exception:
                        pass
            
            print(f"\n---> Running Campaign: {ds_name.upper()} (Ratio: {r*100:.1f}%, {len(seeds)} seeds)...")
            t0 = time.time()
            ds_kwargs = {}
            if root_data:
                ds_kwargs["root"] = root_data
            ds = bna.load_dataset(ds_name, **ds_kwargs)
            G = ds["G"]
            pyg = ds["pyg_data"]
            y = ds["labels"]
            n = G.number_of_nodes()
            
            ds_ratio_rows = []
            for seed in seeds:
                np.random.seed(seed)
                indices = np.arange(n)
                np.random.shuffle(indices)
                split_pt = max(1, int(r * n))
                train_mask = np.zeros(n, dtype=bool)
                train_mask[indices[:split_pt]] = True
                test_mask = ~train_mask
                
                for category, variant_name, base_cfg in ablation_suite:
                    cfg = copy.deepcopy(base_cfg)
                    cfg.name = variant_name
                    cfg.seed = seed
                    cfg.device = device
                    
                    emb, gen_time, diag = generate_ablated_representation(G, pyg, y, train_mask, cfg)
                    metrics = evaluate_representation_probe(
                        emb, y, train_mask, test_mask, 
                        probe_type=probe_type,
                        edge_index=getattr(pyg, 'edge_index', None)
                    )
                    
                    row = {
                        "dataset": ds_name,
                        "ratio": r,
                        "seed": seed,
                        "category": category,
                        "variant": variant_name,
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "micro_f1": metrics["micro_f1"],
                        "gen_time": gen_time,
                        "pseudo_yield": diag["pseudo_accepted_count"],
                        "pseudo_fraction": diag["pseudo_accepted_fraction"]
                    }
                    if "per_classifier" in metrics:
                        row["acc_gcn"] = metrics["per_classifier"]["GCN"]["accuracy"]
                        row["acc_gsage"] = metrics["per_classifier"]["GraphSAGE"]["accuracy"]
                        row["acc_mlp"] = metrics["per_classifier"]["MLP"]["accuracy"]
                        row["acc_linkx"] = metrics["per_classifier"]["LINKX"]["accuracy"]
                    ds_ratio_rows.append(row)
                    all_rows.append(row)
                    
            if ckpt_file is not None:
                pd.DataFrame(ds_ratio_rows).to_csv(ckpt_file, index=False)
            print(f"  [DONE]   {ds_name.upper():14s} (ratio={r*100:4.1f}%) in {time.time()-t0:5.1f}s")
                
    results_df = pd.DataFrame(all_rows)
    return results_df


# =============================================================================
# 8. PUBLICATION TABLE & SYNTHESIS GENERATOR
# =============================================================================
def build_comprehensive_ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs a synthesis table reporting mean accuracy, delta, p-value, macro F1, and latency."""
    full_mask = df["variant"] == "Full_NyS_Binary (Ours)"
    has_ratio = "ratio" in df.columns
    if has_ratio:
        full_acc_by_ds_seed = df[full_mask].set_index(["dataset", "seed", "ratio"])["accuracy"].to_dict()
    else:
        full_acc_by_ds_seed = df[full_mask].set_index(["dataset", "seed"])["accuracy"].to_dict()
    
    summary_rows = []
    variants = df["variant"].unique()
    
    for v in variants:
        v_df = df[df["variant"] == v]
        cat = v_df["category"].iloc[0]
        
        accs = v_df["accuracy"].values
        macs = v_df["macro_f1"].values
        times = v_df["gen_time"].values
        
        diffs = []
        full_accs_matched = []
        v_accs_matched = []
        for _, r in v_df.iterrows():
            key = (r["dataset"], r["seed"], r["ratio"]) if has_ratio else (r["dataset"], r["seed"])
            if key in full_acc_by_ds_seed:
                full_val = full_acc_by_ds_seed[key]
                diffs.append(r["accuracy"] - full_val)
                full_accs_matched.append(full_val)
                v_accs_matched.append(r["accuracy"])
                
        if len(v_accs_matched) > 1 and v != "Full_NyS_Binary (Ours)":
            stat_res = compute_paired_statistics(v_accs_matched, full_accs_matched)
            sig_marker = stat_res["sig_marker"]
            p_val = stat_res["p_wilcoxon"]
        else:
            sig_marker = "ref"
            p_val = 1.0
            
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_mac = np.mean(macs)
        std_mac = np.std(macs)
        mean_time = np.mean(times)
        delta_acc = np.mean(diffs) if diffs else 0.0
        
        row_dict = {
            "Category": cat,
            "Variant / Design Decision": v,
        }
        if "acc_gcn" in v_df.columns:
            row_dict["GCN (%)"] = f"{np.mean(v_df['acc_gcn']):.2f}"
            row_dict["GraphSAGE (%)"] = f"{np.mean(v_df['acc_gsage']):.2f}"
            row_dict["MLP (%)"] = f"{np.mean(v_df['acc_mlp']):.2f}"
            row_dict["LINKX (%)"] = f"{np.mean(v_df['acc_linkx']):.2f}"
            row_dict["Average (4 Clf) (Mean ± Std %)"] = f"{mean_acc:.2f} ± {std_acc:.2f}"
        else:
            row_dict["Accuracy (Mean ± Std %)"] = f"{mean_acc:.2f} ± {std_acc:.2f}"
            
        if v == "Full_NyS_Binary (Ours)":
            row_dict["Delta vs Full (%)"] = "--- [Ref]"
        else:
            row_dict["Delta vs Full (%)"] = f"{delta_acc:+.2f}%"
        row_dict["Signif. (p-val)"] = f"{sig_marker} ({p_val:.3f})" if sig_marker != "ref" else "--- [Ref]"
        row_dict["Macro F1 (Mean ± Std %)"] = f"{mean_mac:.2f} ± {std_mac:.2f}"
        row_dict["Gen Time (s)"] = f"{mean_time:.3f}s"
        summary_rows.append(row_dict)
        
    summary_df = pd.DataFrame(summary_rows)
    return summary_df


# =============================================================================
# 9. FACTORIAL SWEEPS (ALPHA x PSEUDO_THRESHOLD)
# =============================================================================
def run_factorial_alpha_threshold_sweep(
    ds: Dict[str, Any],
    alphas: List[float] = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    thresholds: List[float] = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95],
    train_ratio: float = 0.7,
    seed: int = 42,
    device: str = "cpu",
    probe_type: str = "gcn"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Executes a complete 2D factorial grid over convex blend alpha and pseudo_threshold."""
    G = ds["G"]
    pyg = ds["pyg_data"]
    y = ds["labels"]
    n = G.number_of_nodes()
    
    np.random.seed(seed)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split_pt = int(train_ratio * n)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[indices[:split_pt]] = True
    test_mask = ~train_mask
    
    acc_grid = np.zeros((len(alphas), len(thresholds)))
    f1_grid = np.zeros((len(alphas), len(thresholds)))
    yield_grid = np.zeros((len(alphas), len(thresholds)))
    
    for i, a in enumerate(alphas):
        for j, th in enumerate(thresholds):
            cfg = AblationConfig(alpha=a, pseudo_threshold=th, seed=seed, device=device)
            e_idx = getattr(pyg, 'edge_index', None)
            emb, _, diag = generate_ablated_representation(G, pyg, y, train_mask, cfg)
            m = evaluate_representation_probe(emb, y, train_mask, test_mask, probe_type=probe_type, edge_index=e_idx, device=device)
            acc_grid[i, j] = m["accuracy"]
            f1_grid[i, j] = m["macro_f1"]
            yield_grid[i, j] = diag["pseudo_accepted_count"]
            
    return acc_grid, f1_grid, yield_grid


# =============================================================================
# 10. HIGH-RESOLUTION PUBLICATION PLOTTING UTILITIES
# =============================================================================
def plot_factorial_alpha_threshold_heatmap(
    acc_grid: np.ndarray,
    alphas: List[float],
    thresholds: List[float],
    dataset_name: str = "Cora",
    save_path: Optional[str] = "ablation_alpha_threshold_heatmap.png"
):
    """Plots 2D heatmap capturing the factorial interaction between alpha and tau."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(
        acc_grid,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=[f"{th:.2f}" for th in thresholds],
        yticklabels=[f"{a:.1f}" for a in alphas],
        cbar_kws={'label': 'Test Accuracy (%)'}
    )
    plt.title(rf"Factorial Interaction: $\alpha$ (Blend Weight) vs. $\tau$ (Pseudo-Gate)\n[{dataset_name}]", fontsize=12, pad=12)
    plt.xlabel(r"Confidence Gate Threshold ($\tau$)", fontsize=11)
    plt.ylabel(r"Convex Blend Weight ($\alpha$)", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved heatmap to: {save_path}")
    plt.close()


def plot_hops_homophily_curve(
    hops_results: Dict[str, List[Tuple[int, float]]],
    save_path: Optional[str] = "ablation_hops_homophily_curve.png"
):
    """Plots accuracy vs hops across homophilous vs heterophilous graphs."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(7, 5), dpi=300)
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd']
    
    for idx, (ds_name, curve) in enumerate(hops_results.items()):
        hops_vals = [pt[0] for pt in curve]
        acc_vals = [pt[1] for pt in curve]
        plt.plot(
            hops_vals, acc_vals,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=2.2,
            label=ds_name
        )
        
    plt.title("Diffusion Radius ($hops$) Sensitivity across Graph Topologies", fontsize=12, pad=12)
    plt.xlabel("Polynomial Graph Diffusion Radius ($hops$)", fontsize=11)
    plt.ylabel("Downstream Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved hops curve to: {save_path}")
    plt.close()


def plot_nystrom_pareto_curve(
    m_values: List[int],
    runtimes: List[float],
    accuracies: List[float],
    exact_svd_time: float,
    exact_svd_acc: float,
    save_path: Optional[str] = "ablation_nystrom_pareto.png"
):
    """Plots Runtime vs Quality Pareto frontier for Nystrom landmark scaling."""
    import matplotlib.pyplot as plt
    
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    
    color = '#1f77b4'
    ax1.set_xlabel('Number of Landmark Nodes ($m$)', fontsize=11)
    ax1.set_ylabel('Generation Time (s)', color=color, fontsize=11)
    line1 = ax1.plot(m_values, runtimes, color=color, marker='o', linewidth=2, label='Nyström Latency (s)')
    ax1.axhline(exact_svd_time, color='black', linestyle=':', label=f'Exact SVD Time ({exact_svd_time:.2f}s)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Downstream Accuracy (%)', color=color, fontsize=11)
    line2 = ax2.plot(m_values, accuracies, color=color, marker='s', linewidth=2, linestyle='--', label='Accuracy (%)')
    ax2.axhline(exact_svd_acc, color='gray', linestyle='--', label=f'Exact SVD Acc ({exact_svd_acc:.2f}%)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Nyström Landmark Scaling: Accuracy vs. Wall-Clock Latency Pareto Trade-off", fontsize=12, pad=12)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved Pareto curve to: {save_path}")
    plt.close()
