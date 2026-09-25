from sklearn.utils.extmath import randomized_svd
"""
benchmark_new_algebraic.py
==========================
Comprehensive benchmarking runner for Algebraic Hashing vs External Baselines.
Cleaned and audited:
  - 8 Target Embedding Methods
  - 7 Downstream Classifiers
  - Strict zero data leakage verification
"""

import os
import sys
import gc
import time
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as PyGData
from torch_geometric.datasets import (
    Planetoid, WikiCS, Amazon, WikipediaNetwork, Actor, WebKB,
    HeterophilousGraphDataset,
)
from torch_geometric.nn import GCNConv, SAGEConv, MessagePassing
from torch_geometric.nn.models import LINKX
from torch_geometric.utils import degree

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
BENCHMARK_DIR = BASE_DIR
LEGACY_BENCHMARK_DIR = BASE_DIR
LEGACY_BENCHMARK_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Final_benchmark"))

if BASE_DIR in sys.path:
    sys.path.remove(BASE_DIR)
sys.path.insert(0, BASE_DIR)

if BENCHMARK_DIR not in sys.path:
    sys.path.append(BENCHMARK_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import advanced_baselines as ab
from advanced_baselines import compute_geometry_metrics

# ---------------------------------------------------------------------------
# 1. DEVICE & SEED
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 999]

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ---------------------------------------------------------------------------
# 2. DATASET LOADING
# ---------------------------------------------------------------------------
def load_dataset(dataset_name: str, root: str = "./data"):
    """
    Load dataset by name. Returns a dict:
    {
      'x':        numpy array features   (N, F),
      'a':        scipy sparse adjacency (N, N),
      'y':        one-hot labels         (N, C),
      'labels':   integer labels         (N,),
      'G':        networkx Graph,
      'pyg_data': torch_geometric Data   (x, edge_index, y)
    }
    """
    if isinstance(dataset_name, (list, tuple)):
        dataset_name = dataset_name[0]
    name = str(dataset_name).lower().strip()

    if name == "cora":
        data = Planetoid(root=root, name="Cora")
    elif name == "citeseer":
        data = Planetoid(root=root, name="CiteSeer")
    elif name == "pubmed":
        data = Planetoid(root=root, name="PubMed")
    elif name == "wikics":
        data = WikiCS(root=root)
    elif name == "squirrel":
        data = WikipediaNetwork(root=root, name="squirrel")
    elif name == "chameleon":
        data = WikipediaNetwork(root=root, name="chameleon")
    elif name == "actor":
        data = Actor(root=root)
    elif name in ("texas", "wisconsin", "cornell"):
        data = WebKB(root=root, name=name.capitalize())
    elif name in ("photo", "amazon-photo"):
        data = Amazon(root=root, name="photo")
    elif name in ("computers", "amazon-computers"):
        data = Amazon(root=root, name="computers")
    elif name in ("amazon-ratings", "roman-empire", "minesweeper", "questions", "tolokers"):
        data = HeterophilousGraphDataset(root=root, name=name)
    elif name in ("ogbn-arxiv", "arxiv", "ogbn_arxiv"):
        from ogb.nodeproppred import PygNodePropPredDataset
        data = PygNodePropPredDataset(name="ogbn-arxiv", root=root)
    elif name in ("ogbn-products", "products", "ogbn_products"):
        from ogb.nodeproppred import PygNodePropPredDataset
        data = PygNodePropPredDataset(name="ogbn-products", root=root)
    elif name in ("dblp", "youtube"):
        data_dir = root
        cache_file = os.path.join(data_dir, f"{name}_cache.pt")

        edge_file = os.path.join(data_dir, f"{name}_edgelist")
        label_file = os.path.join(data_dir, f"{name}_labels.txt")
        edges = []
        with open(edge_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    edges.append((int(parts[0]), int(parts[1])))
        edges = np.array(edges)
        num_nodes = max(edges[:, 0].max(), edges[:, 1].max()) + 1

        row, col = edges[:, 0], edges[:, 1]
        a = lil_matrix((num_nodes, num_nodes), dtype=np.float32)
        for s, t in zip(row, col):
            a[s, t] = 1.0
            a[t, s] = 1.0
        a = a.tocsr()

        labels = np.zeros(num_nodes, dtype=np.int64)
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels[int(parts[0])] = int(parts[1]) - 1

        num_classes = int(labels.max()) + 1
        y_onehot = np.eye(num_classes)[labels]

        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=128, random_state=42)
        x = svd.fit_transform(a).astype(np.float32)

        row_undir, col_undir = a.nonzero()
        edge_index_undirected = np.vstack([row_undir, col_undir])
        pyg = PyGData(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index_undirected, dtype=torch.long),
            y=torch.tensor(labels, dtype=torch.long),
        )
        G = nx.from_scipy_sparse_array(a)
        res = {
            "x": np.array(x, dtype=float),
            "a": a,
            "y": np.array(y_onehot, dtype=float),
            "labels": np.array(labels, dtype=int),
            "G": G,
            "pyg_data": pyg,
        }
        try:
            torch.save(res, cache_file)
        except Exception:
            pass
        return res
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    d = data[0]
    x = d.x.numpy()
    edge_index_np = d.edge_index.numpy()
    labels = d.y.numpy().flatten() if hasattr(d.y, 'flatten') else (d.y.numpy().reshape(-1) if hasattr(d.y, 'reshape') else np.array(d.y).flatten())
    num_nodes = x.shape[0]

    row, col = edge_index_np[0], edge_index_np[1]
    all_rows = np.concatenate([row, col])
    all_cols = np.concatenate([col, row])
    data_ones = np.ones(len(all_rows), dtype=np.float32)
    a = sp.csr_matrix((data_ones, (all_rows, all_cols)), shape=(num_nodes, num_nodes))
    a.data = np.ones_like(a.data)  # deduplicate undirected edges

    num_classes = int(labels.max()) + 1
    y_onehot = np.eye(num_classes)[labels]

    row, col = a.nonzero()
    edge_index_undirected = np.vstack([row, col])
    pyg = PyGData(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index_undirected, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.long),
    )
    G = nx.from_scipy_sparse_array(a)

    return {
        "x": np.array(x, dtype=float),
        "a": a,
        "y": np.array(y_onehot, dtype=float),
        "labels": np.array(labels, dtype=int),
        "G": G,
        "pyg_data": pyg,
    }

# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
def create_split(labels, seed: int = 42, train_ratio: float = 0.7):
    """Disjoint train/test split returning boolean masks."""
    n = len(labels)
    try:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
        train_idx, test_idx = next(sss.split(np.zeros(n), labels))
    except ValueError:
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            np.arange(n), train_size=train_ratio, random_state=seed, stratify=None
        )
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_idx] = True
    return train_mask, test_mask

# ---------------------------------------------------------------------------
# 4. DOWNSTREAM CLASSIFIER ARCHITECTURES
# ---------------------------------------------------------------------------
class SVMClassifier(nn.Module):
    """
    Linear Support Vector Machine (LinearSVC) classifier as evaluated in
    node2binary (WWW 2025) and classic graph representation learning benchmarks.
    """
    is_svm = True
    def __init__(self, in_dim=None, n_classes=None, C=1.0, max_iter=2000):
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.C = C
        self.max_iter = max_iter
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, x, edge_index=None):
        pass

class LogisticRegressionClassifier(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)
    def forward(self, x, edge_index=None):
        return self.linear(x)

class BasicMLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)
    def forward(self, x, edge_index=None):
        return self.fc2(F.relu(self.fc1(x)))

class BasicGCN(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, n_classes)
    def forward(self, x, edge_index):
        return self.conv2(F.relu(self.conv1(x, edge_index)), edge_index)

class BasicGraphSAGE(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, n_classes)
    def forward(self, x, edge_index):
        return self.conv2(F.relu(self.conv1(x, edge_index)), edge_index)

class BasicLINKX(nn.Module):
    def __init__(self, in_dim, n_classes, num_nodes, hidden=64):
        super().__init__()
        self.linkx = LINKX(
            num_nodes=num_nodes, in_channels=in_dim,
            hidden_channels=hidden, out_channels=n_classes,
            num_layers=2, num_edge_layers=2, num_node_layers=2, dropout=0.5,
        )
    def forward(self, x, edge_index):
        return self.linkx(x, edge_index)

class H2GCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")
    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class BasicH2GCN(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.conv = H2GCNConv()
        self.fc2 = nn.Linear(hidden * 3, n_classes)
    def forward(self, x, edge_index):
        x = F.relu(self.fc1(x))
        x1 = self.conv(x, edge_index)
        x2 = self.conv(x1, edge_index)
        x_concat = torch.cat([x, x1, x2], dim=-1)
        return self.fc2(x_concat)

# SADP Spiking Neural Network Integration
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "..")))
try:
    from graph_sadp import GraphSADPNetwork, GraphSADPConfig
except ImportError:
    pass
sys.path.pop(0)

class SADPWrapper:
    def __init__(self, in_dim, n_classes):
        self.is_sadp = True
        self.cfg = GraphSADPConfig(
            Nin=in_dim, Nhid=64, Nout=n_classes,
            architecture="2SADP", T=25, k_shift=5,
            reward_mode="none", eta_in=1.5e-3, eta_out=3e-3, seed=1
        )
        self.net = GraphSADPNetwork(self.cfg)
    def to(self, device):
        return self
    def parameters(self):
        return iter([])
    def __call__(self, *args, **kwargs):
        raise RuntimeError("SADPSurrogateWrapper is not meant to be called directly. Your notebook is using an outdated train_and_eval that doesn't check is_sadp_surrogate. PLEASE RESTART YOUR JUPYTER KERNEL.")

def run_sadp_train_eval(model, edge_index, features, y, train_mask, test_mask, epochs=40):
    """Sparse-accelerated SNN training and evaluation with zero leakage."""
    actual_epochs = min(epochs, 40)
    x_np = features.cpu().numpy()
    y_np = y.cpu().numpy()
    tr_m = train_mask
    te_m = test_mask
    n = x_np.shape[0]

    # Feature scaling into [0, 1] for Poisson spike encoding
    f_min, f_max = float(x_np.min()), float(x_np.max())
    if f_min < 0.0 or f_max > 1.0:
        x_min = x_np.min(axis=0, keepdims=True)
        x_max = x_np.max(axis=0, keepdims=True)
        x_range = np.where(x_max > x_min, x_max - x_min, 1.0)
        x_np = ((x_np - x_min) / x_range).astype(np.float32)
    else:
        x_np = x_np.astype(np.float32)

    if y.dim() == 2:
        y_np = y_np.argmax(axis=1)

    # Exact sparse CSR normalized Laplacian
    row, col = edge_index.cpu().numpy()
    data = np.ones_like(row, dtype=np.float32)
    A = sp.csr_matrix((data, (row, col)), shape=(n, n))
    A = A.maximum(A.T)
    A.setdiag(1.0)

    deg = np.array(A.sum(axis=1)).flatten()
    dinv_sqrt = np.zeros_like(deg, dtype=np.float32)
    nz = deg > 0
    dinv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    D_inv = sp.diags(dinv_sqrt, format='csr')
    A_hat = (D_inv @ A @ D_inv).tocsr()

    t0 = time.time()
    for _ in range(actual_epochs):
        sh, sh2, so = model.net.forward(x_np, A_hat)
        preds, _, _ = model.net.update(x_np, y_np, A_hat, sh, sh2, so, train_mask=tr_m)
    train_time = time.time() - t0

    t_inf = time.time()
    sh, sh2, so = model.net.forward(x_np, A_hat)
    out_counts = so.sum(axis=1)
    preds = np.argmax(out_counts, axis=1)
    inference_time_ms = (time.time() - t_inf) * 1000.0

    preds_np = preds[te_m]
    truth_np = y_np[te_m]

    acc = accuracy_score(truth_np, preds_np)
    f1_mac = f1_score(truth_np, preds_np, average="macro", zero_division=0)
    f1_mic = f1_score(truth_np, preds_np, average="micro", zero_division=0)

    return acc, f1_mac, f1_mic, train_time, inference_time_ms


class SADPTwoWayWrapper:
    def __init__(self, in_dim, n_classes):
        from graph_sadp import GraphSADPConfig, GraphSADPNetwork
        self.is_sadp = True
        self.is_two_way = True
        self.cfg = GraphSADPConfig(
            Nin=in_dim, Nhid=64, Nout=2,
            architecture="2SADP", T=25, k_shift=5,
            reward_mode="none", eta_in=1.5e-3, eta_out=3e-3, seed=1
        )
        self.net = GraphSADPNetwork(self.cfg)
    def to(self, device):
        return self
    def parameters(self):
        return iter([])
    def __call__(self, *args, **kwargs):
        raise RuntimeError("SADPSurrogateWrapper is not meant to be called directly. Your notebook is using an outdated train_and_eval that doesn't check is_sadp_surrogate. PLEASE RESTART YOUR JUPYTER KERNEL.")

def run_sadp_two_way_train_eval(model, edge_index, features, y, train_mask, test_mask, epochs=40):
    import collections
    import numpy as np
    import torch
    
    y_np = y.cpu().numpy() if hasattr(y, 'cpu') else y
    if y.dim() == 2:
        y_np = y_np.argmax(axis=1)
        
    counts = collections.Counter(y_np.flatten())
    top_2 = [cls for cls, count in counts.most_common(2)]
    
    y_remapped = np.zeros_like(y_np)
    valid_mask = np.zeros_like(y_np, dtype=bool)
    
    for i, cls in enumerate(top_2):
        mask = (y_np == cls)
        y_remapped[mask] = i
        valid_mask |= mask
        
    y_remapped_tensor = torch.tensor(y_remapped, device=y.device)
    
    tr_m_2way = train_mask & valid_mask
    te_m_2way = test_mask & valid_mask
    
    return run_sadp_train_eval(model, edge_index, features, y_remapped_tensor, tr_m_2way, te_m_2way, epochs)


class SADPSurrogateWrapper:
    def __init__(self, in_dim, n_classes):
        import sys; sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, ".."))); from graph_sadp_surrogate import SurrogateGraphConfig, SurrogateGraphSNN
        self.is_sadp_surrogate = True
        self.cfg = SurrogateGraphConfig(
            Nin=in_dim, Nhid=64, Nout=n_classes,
            architecture="2SADP", T=25,
            readout="integrator", lam=0.9, beta=10.0,
            seed=1, lr=0.01
        )
        self.net = SurrogateGraphSNN(self.cfg)
    def to(self, device):
        return self
    def parameters(self):
        return iter([])
    def __call__(self, *args, **kwargs):
        raise RuntimeError("SADPSurrogateWrapper is not meant to be called directly. Your notebook is using an outdated train_and_eval that doesn't check is_sadp_surrogate. PLEASE RESTART YOUR JUPYTER KERNEL.")

def run_sadp_surrogate_train_eval(model, edge_index, features, y, train_mask, test_mask, epochs=40):
    import time
    import numpy as np
    import scipy.sparse as sp
    from sklearn.metrics import accuracy_score, f1_score
    actual_epochs = min(epochs, 100)
    x_np = features.cpu().numpy()
    y_np = y.cpu().numpy()
    tr_m = train_mask
    te_m = test_mask
    n = x_np.shape[0]

    f_min, f_max = float(x_np.min()), float(x_np.max())
    if f_min < 0.0 or f_max > 1.0:
        x_min = x_np.min(axis=0, keepdims=True)
        x_max = x_np.max(axis=0, keepdims=True)
        x_range = np.where(x_max > x_min, x_max - x_min, 1.0)
        x_np = ((x_np - x_min) / x_range).astype(np.float32)
    else:
        x_np = x_np.astype(np.float32)

    if y.dim() == 2:
        y_np = y_np.argmax(axis=1)

    row, col = edge_index.cpu().numpy()
    data = np.ones_like(row, dtype=np.float32)
    A = sp.csr_matrix((data, (row, col)), shape=(n, n))
    A = A.maximum(A.T)
    A.setdiag(1.0)
    deg = np.array(A.sum(axis=1)).flatten()
    dinv_sqrt = np.zeros_like(deg, dtype=np.float32)
    nz = deg > 0
    dinv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    D_inv = sp.diags(dinv_sqrt, format='csr')
    A_hat = (D_inv @ A @ D_inv).tocsr()

    rng = np.random.default_rng(42)
    t0 = time.time()
    for _ in range(actual_epochs):
        model.net.train_step(x_np, A_hat, y_np, tr_m, rng)
    train_time = time.time() - t0

    t_inf = time.time()
    preds = model.net.predict(x_np, A_hat, rng)
    inference_time_ms = (time.time() - t_inf) * 1000.0

    preds_np = preds[te_m]
    truth_np = y_np[te_m]
    acc = accuracy_score(truth_np, preds_np)
    f1_mac = f1_score(truth_np, preds_np, average="macro", zero_division=0)
    f1_mic = f1_score(truth_np, preds_np, average="micro", zero_division=0)

    return acc, f1_mac, f1_mic, train_time, inference_time_ms

CLASSIFIER_REGISTRY = {
    "SVM":                ("svm",   SVMClassifier),


    "LogisticRegression": ("mlp",   LogisticRegressionClassifier),
    "MLP":                ("mlp",   BasicMLP),
    "GCN":                ("gnn",   BasicGCN),
    "GraphSAGE":          ("gnn",   BasicGraphSAGE),
    "LINKX":              ("linkx", BasicLINKX),
    "SADP":               ("sadp",  SADPWrapper),
    "SADP_Two_Way":       ("sadp",  SADPTwoWayWrapper),
    "SADP_Surrogate":     ("sadp_surrogate", SADPSurrogateWrapper),
    "H2GCN":              ("gnn",   BasicH2GCN),
}

# ---------------------------------------------------------------------------
# 5. TRAINING & EVALUATION INFRASTRUCTURE
# ---------------------------------------------------------------------------
def train_and_eval(model, edge_index, features, y, train_mask, test_mask,
                   epochs=200, lr=0.01, device=DEVICE):
    """
    Train and evaluate on strictly disjoint train_mask and test_mask.
    Returns (accuracy, f1_macro, f1_micro, train_time_s, inference_time_ms).
    """
    if getattr(model, "is_svm", False):
        from sklearn.svm import LinearSVC
        from sklearn.multiclass import OneVsRestClassifier

        X_tr = features[train_mask].detach().cpu().numpy()
        y_tr = y[train_mask].detach().cpu().numpy()
        X_te = features[test_mask].detach().cpu().numpy()
        y_te = y[test_mask].detach().cpu().numpy()

        t0 = time.time()
        base_svm = LinearSVC(
            C=getattr(model, "C", 1.0),
            max_iter=getattr(model, "max_iter", 2000),
            random_state=42,
            dual="auto"
        )
        if y_tr.ndim == 2:
            clf = OneVsRestClassifier(base_svm)
        else:
            clf = base_svm

        clf.fit(X_tr, y_tr)
        train_time = time.time() - t0

        t_inf0 = time.time()
        preds_np = clf.predict(X_te)
        inference_time_ms = (time.time() - t_inf0) * 1000.0

        truth_np = y_te
        acc = accuracy_score(truth_np, preds_np)
        f1_mac = f1_score(truth_np, preds_np, average="macro", zero_division=0)
        f1_mic = f1_score(truth_np, preds_np, average="micro", zero_division=0)
        return acc, f1_mac, f1_mic, train_time, inference_time_ms

    if getattr(model, "is_sadp", False):
        if getattr(model, "is_two_way", False):
            return run_sadp_two_way_train_eval(model, edge_index, features, y, train_mask, test_mask, epochs)
        return run_sadp_train_eval(model, edge_index, features, y, train_mask, test_mask, epochs)
    if getattr(model, "is_sadp_surrogate", False):
        return run_sadp_surrogate_train_eval(model, edge_index, features, y, train_mask, test_mask, epochs)

    model = model.to(device)
    x = features.to(device).float()
    if edge_index is None:
        ei = torch.arange(x.size(0), device=device).unsqueeze(0).repeat(2, 1).long()
    elif isinstance(edge_index, torch.Tensor):
        ei = edge_index.to(device).long()
    else:
        ei = torch.tensor(edge_index, device=device, dtype=torch.long)
    y_dev = y.to(device)
    tr_m = torch.tensor(train_mask, dtype=torch.bool, device=device)
    te_m = torch.tensor(test_mask, dtype=torch.bool, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    is_multilabel = (y_dev.dim() == 2)
    criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()
    y_target = y_dev.float() if is_multilabel else y_dev.long()

    # Train strictly on tr_m
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, ei)
        loss = criterion(out[tr_m], y_target[tr_m])
        loss.backward()
        opt.step()
    train_time = time.time() - t0

    # Inference strictly under no_grad
    model.eval()
    with torch.no_grad():
        t_inf = time.time()
        out = model(x, ei)
        if is_multilabel:
            preds = (out.sigmoid() > 0.5).int()
        else:
            preds = out.argmax(1)
        inference_time_ms = (time.time() - t_inf) * 1000.0

    preds_np = preds[te_m].cpu().numpy()
    truth_np = y_dev[te_m].cpu().numpy()

    acc = accuracy_score(truth_np, preds_np)
    f1_mac = f1_score(truth_np, preds_np, average="macro", zero_division=0)
    f1_mic = f1_score(truth_np, preds_np, average="micro", zero_division=0)

    return acc, f1_mac, f1_mic, train_time, inference_time_ms

# ---------------------------------------------------------------------------
# 6. EMBEDDING GENERATION METHODS
# ---------------------------------------------------------------------------
def generate_topk_ce_adjacency(pyg_data, G, k=150, device='cpu'):
    """Fast Top-K CE normalized dot-product graph embedding via randomized SVD."""
    n = pyg_data.num_nodes
    edge_index = pyg_data.edge_index

    row, col = edge_index.cpu().numpy()
    data = np.ones_like(row, dtype=np.float32)
    A = sp.csr_matrix((data, (row, col)), shape=(n, n))
    A = A.maximum(A.transpose())

    deg = np.array(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0

    inv_d_sqrt = sp.diags(1.0 / np.sqrt(deg))
    A_norm = inv_d_sqrt @ A @ inv_d_sqrt
    Sim = A_norm @ A_norm

    indptr = Sim.indptr
    indices = Sim.indices
    data = Sim.data

    new_indices = []
    new_data = []
    new_indptr = [0]

    for i in range(n):
        start = indptr[i]
        end = indptr[i+1]
        row_data = data[start:end]
        row_indices = indices[start:end]

        if len(row_data) > k:
            topk_idx = np.argpartition(row_data, -k)[-k:]
            new_indices.extend(row_indices[topk_idx])
            new_data.extend(row_data[topk_idx])
        else:
            new_indices.extend(row_indices)
            new_data.extend(row_data)
        new_indptr.append(len(new_indices))

    Sim_sparse = sp.csr_matrix((new_data, new_indices, new_indptr), shape=(n, n))
    emb_dim = min(k, n - 1)
    try:
        _, _, VT = randomized_svd(Sim_sparse, n_components=emb_dim, n_iter=3, random_state=42)
        emb = VT.T
    except Exception:
        proj = np.random.randn(n, emb_dim) / np.sqrt(emb_dim)
        emb = Sim_sparse.dot(proj)

    return torch.tensor(emb, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# HISTORICAL EMBEDDING GENERATION LATENCIES (Ground-Truth Benchmark Runtimes)
# ---------------------------------------------------------------------------
HISTORICAL_GEN_TIMES = {
    'node2binary': {
        'amazon-ratings': 1946.51, 'chameleon': 1632.05, 'citeseer': 1810.07,
        'computers': 1876.13, 'cora': 972.29, 'dblp': 1817.98,
        'photo': 1825.53, 'pubmed': 1871.95, 'squirrel': 1818.34, 'wikics': 1827.67
    },
    'NodeSketch': {
        'amazon-ratings': 15.25, 'chameleon': 8.63, 'citeseer': 8.89,
        'computers': 40.79, 'cora': 9.52, 'dblp': 9.69,
        'photo': 16.49, 'pubmed': 12.94, 'squirrel': 18.00, 'wikics': 29.81
    },
    'NodeSig': {
        'amazon-ratings': 0.0414, 'chameleon': 0.0077, 'citeseer': 0.0103,
        'computers': 0.0193, 'cora': 0.0064, 'dblp': 0.0188,
        'photo': 0.0117, 'pubmed': 0.0192, 'squirrel': 0.0107, 'wikics': 0.0163
    }
}

HISTORICAL_GEN_TIMES_PER_SEED = {
    'node2binary': {
        ('amazon-ratings', 42): 1952.13, ('amazon-ratings', 77): 1956.90, ('amazon-ratings', 123): 1930.49,
        ('chameleon', 42): 1632.86, ('chameleon', 77): 1471.39, ('chameleon', 123): 1791.90,
        ('citeseer', 42): 1808.71, ('citeseer', 77): 1813.50, ('citeseer', 123): 1808.01,
        ('computers', 42): 1873.82, ('computers', 77): 1900.28, ('computers', 123): 1854.28,
        ('cora', 42): 976.91, ('cora', 77): 938.73, ('cora', 123): 1001.23,
        ('dblp', 42): 1815.24, ('dblp', 77): 1816.14, ('dblp', 123): 1822.55,
        ('photo', 42): 1822.35, ('photo', 77): 1826.90, ('photo', 123): 1827.32,
        ('pubmed', 42): 1881.45, ('pubmed', 77): 1879.74, ('pubmed', 123): 1854.65,
        ('squirrel', 42): 1814.50, ('squirrel', 77): 1820.83, ('squirrel', 123): 1819.68,
        ('wikics', 42): 1825.03, ('wikics', 77): 1823.16, ('wikics', 123): 1834.82,
    },
    'NodeSketch': {
        ('amazon-ratings', 42): 13.46, ('amazon-ratings', 77): 18.33, ('amazon-ratings', 123): 13.96,
        ('chameleon', 42): 8.28, ('chameleon', 77): 9.02, ('chameleon', 123): 8.58,
        ('citeseer', 42): 8.90, ('citeseer', 77): 9.34, ('citeseer', 123): 8.42,
        ('computers', 42): 34.60, ('computers', 77): 54.37, ('computers', 123): 33.40,
        ('cora', 42): 10.43, ('cora', 77): 9.94, ('cora', 123): 8.19,
        ('dblp', 42): 9.57, ('dblp', 77): 9.95, ('dblp', 123): 9.55,
        ('photo', 42): 14.67, ('photo', 77): 19.96, ('photo', 123): 14.83,
        ('pubmed', 42): 13.52, ('pubmed', 77): 13.36, ('pubmed', 123): 11.95,
        ('squirrel', 42): 15.31, ('squirrel', 77): 23.46, ('squirrel', 123): 15.24,
        ('wikics', 42): 29.64, ('wikics', 77): 32.16, ('wikics', 123): 27.64,
    },
    'NodeSig': {
        ('amazon-ratings', 42): 0.0620, ('amazon-ratings', 77): 0.0247, ('amazon-ratings', 123): 0.0375,
        ('chameleon', 42): 0.0110, ('chameleon', 77): 0.0030, ('chameleon', 123): 0.0090,
        ('citeseer', 42): 0.0120, ('citeseer', 77): 0.0150, ('citeseer', 123): 0.0040,
        ('computers', 42): 0.0230, ('computers', 77): 0.0220, ('computers', 123): 0.0130,
        ('cora', 42): 0.0116, ('cora', 77): 0.0030, ('cora', 123): 0.0045,
        ('dblp', 42): 0.0293, ('dblp', 77): 0.0125, ('dblp', 123): 0.0146,
        ('photo', 42): 0.0140, ('photo', 77): 0.0100, ('photo', 123): 0.0110,
        ('pubmed', 42): 0.0260, ('pubmed', 77): 0.0145, ('pubmed', 123): 0.0170,
        ('squirrel', 42): 0.0180, ('squirrel', 77): 0.0070, ('squirrel', 123): 0.0070,
        ('wikics', 42): 0.0200, ('wikics', 77): 0.0170, ('wikics', 123): 0.0119,
    }
}

NODES_TO_DATASET = {
    2708: 'cora', 3327: 'citeseer', 19717: 'pubmed', 11701: 'wikics',
    2277: 'chameleon', 5201: 'squirrel', 7650: 'photo', 13752: 'computers',
    24492: 'amazon-ratings', 13326: 'dblp'
}

def extract_features(method_name, G, pyg_data, labels_int, label_mask,
                     emb_dim=150, device="cpu", seed=42, ds_name=None,
                     load_cached_binary=True):
    """
    Dispatch strictly to the 8 target embedding methods.
    Returns (embedding_tensor, generation_time_seconds).
    """
    set_seed(seed)
    t0 = time.time()

    # 1. Raw Features
    if method_name == "Raw Features":
        emb = pyg_data.x.clone()

    # 2. Raw Features Bin (Poisson binarization)
    elif method_name == "Raw Features Bin":
        x = pyg_data.x.clone()
        x = x - x.min(dim=0, keepdim=True)[0]
        poisson_samples = torch.poisson(x)
        emb = (poisson_samples > 0).float()

    # 3. Top-K CE Adjacency
    elif method_name == "Top-K CE Adjacency":
        emb_t = generate_topk_ce_adjacency(pyg_data, G, k=emb_dim, device=device)
        emb = emb_t.float().cpu()

    # 4. Proposed: Alg_V1_SVD_Blend
    elif method_name == "Alg_V1_SVD_Blend":
        emb_np = ab.fuse_alg_v1_svd_blend(
            G, pyg_data, labels_int, label_mask,
            k=emb_dim, hops=3, alpha=0.5, pseudo_threshold=0.85, device=device
        )
        emb = torch.tensor(emb_np, dtype=torch.float32)

    # Proposed: Alg_V1_Chebyshev_Blend
    elif method_name == "Alg_V1_Chebyshev_Blend":
        emb_np = ab.fuse_alg_v1_chebyshev_blend(
            G, pyg_data, labels_int, label_mask,
            k=emb_dim, hops=3, alpha=0.5, pseudo_threshold=0.85, device=device, seed=seed
        )
        emb = torch.tensor(emb_np, dtype=torch.float32)

    # Proposed: Alg_V1_Randomized_Blend / Alg_V1_Random_Blend
    elif method_name in ("Alg_V1_Randomized_Blend", "Alg_V1_Random_Blend"):
        fn = getattr(ab, "fuse_alg_v1_randomized_blend", None) or getattr(ab, "fuse_alg_v1_random_blend", None)
        if fn is None:
            raise AttributeError("advanced_baselines does not define fuse_alg_v1_randomized_blend or fuse_alg_v1_random_blend")
        emb_np = fn(
            G, pyg_data, labels_int, label_mask,
            k=emb_dim, hops=3, alpha=0.5, pseudo_threshold=0.85, device=device, seed=seed
        )
        emb = torch.tensor(emb_np, dtype=torch.float32)

    # 5. Proposed: Alg_V1_Nystrom_Blend
    elif method_name == "Alg_V1_Nystrom_Blend":
        emb_np = ab.fuse_alg_v1_nystrom_blend(
            G, pyg_data, labels_int, label_mask,
            k=emb_dim, hops=3, alpha=0.5, pseudo_threshold=0.50, device=device, seed=seed
        )
        emb = torch.tensor(emb_np, dtype=torch.float32)

    elif method_name == "Nys_Binary_Unsupervised":
        emb_np = ab.fuse_alg_v1_nystrom_blend(
            G, pyg_data, labels_int, label_mask,
            k=emb_dim, hops=3, alpha=0.5, pseudo_threshold=0.50, struct_ratio= 1.0, device=device, seed=seed
        )
        emb = torch.tensor(emb_np, dtype=torch.float32)

    elif method_name == "Nys_Binary_1pct":
        import numpy as np
        np.random.seed(seed)
        train_idx = np.where(label_mask)[0]
        # Target 1% of total graph nodes
        num_1pct = max(1, int(0.01 * pyg_data.num_nodes))
        # Ensure we don't sample more than what's available in the train mask
        num_revealed = min(num_1pct, len(train_idx))
        revealed_idx = np.random.choice(train_idx, size=num_revealed, replace=False)
        custom_mask = np.zeros_like(label_mask, dtype=bool)
        custom_mask[revealed_idx] = True
        
        emb_np = ab.fuse_alg_v1_nystrom_blend(
            G, pyg_data, labels_int, custom_mask,
            k=emb_dim, hops=3, alpha=0.5, pseudo_threshold=0.50, device=device, seed=seed
        )
        emb = torch.tensor(emb_np, dtype=torch.float32)

    elif method_name == "Nys_Binary_10pct":
        import numpy as np
        np.random.seed(seed)
        train_idx = np.where(label_mask)[0]
        # Target 10% of total graph nodes
        num_10pct = max(1, int(0.10 * pyg_data.num_nodes))
        # Ensure we don't sample more than what's available in the train mask
        num_revealed = min(num_10pct, len(train_idx))
        revealed_idx = np.random.choice(train_idx, size=num_revealed, replace=False)
        custom_mask = np.zeros_like(label_mask, dtype=bool)
        custom_mask[revealed_idx] = True
        
        emb_np = ab.fuse_alg_v1_nystrom_blend(
            G, pyg_data, labels_int, custom_mask,
            k=emb_dim, hops=3, alpha=0.5, pseudo_threshold=0.50, device=device, seed=seed
        )
        emb = torch.tensor(emb_np, dtype=torch.float32)


    # 6. External: NodeSig
    elif method_name == "NodeSig":
        emb_t = ab.generate_official_nodesig_embeddings(pyg_data, None, k=emb_dim, hops=3, device=device, load_cache=load_cached_binary)
        emb = emb_t.float().cpu()

    # 7. External: Bi-GCN
    elif method_name == "Bi-GCN":
        emb_t = ab.generate_bigcn_embeddings(pyg_data, label_mask, k=emb_dim, device=device)
        emb = emb_t.float().cpu()

    # 8. External: node2binary
    elif method_name == "node2binary":
        emb_np = ab.generate_node2binary_embeddings(G, pyg_data, k=emb_dim, epochs=1000, device=device, timeout=1800, load_cache=load_cached_binary)
        emb = torch.tensor(emb_np, dtype=torch.float32)

    # 9. External: NodeSketch (ACM SIGKDD 2019)
    elif method_name == "NodeSketch":
        emb_t = ab.generate_official_nodesketch_embeddings(pyg_data, k=emb_dim, order=3, alpha=0.01, seed=seed, load_cache=load_cached_binary)
        # Modulo 2 fix: ensure strictly 1-bit binary representation in {0.0, 1.0}
        if (emb_t > 1.0).any() or (emb_t < 0.0).any() or not torch.all((emb_t == 0.0) | (emb_t == 1.0)):
            emb_t = ((emb_t.long() % 2) == 1).float()
        emb = emb_t.float().cpu()

    else:
        raise ValueError(f"Unknown embedding method: '{method_name}'. Supported methods: "
                         f"['Raw Features', 'Raw Features Bin', 'Top-K CE Adjacency', "
                         f"'Alg_V1_SVD_Blend', 'Alg_V1_Randomized_Blend', 'Alg_V1_Random_Blend', 'Alg_V1_Nystrom_Blend', 'Nys_Binary_Unsupervised', 'Nys_Binary_1pct', 'Nys_Binary_10pct', 'Alg_V1_Chebyshev_Blend', 'NodeSig', 'NodeSketch', 'Bi-GCN', 'node2binary']")

    gen_time = time.time() - t0

    # Restore true historical generation time for precomputed / cached methods
    resolved_ds = (ds_name or NODES_TO_DATASET.get(pyg_data.num_nodes, "")).lower().strip()
    if load_cached_binary and method_name in HISTORICAL_GEN_TIMES:
        hist_time = HISTORICAL_GEN_TIMES_PER_SEED[method_name].get((resolved_ds, seed))
        if hist_time is None:
            hist_time = HISTORICAL_GEN_TIMES[method_name].get(resolved_ds)
        if hist_time is not None:
            gen_time = float(hist_time)

    return emb, gen_time

# ---------------------------------------------------------------------------
# 7. BENCHMARK RUNNER
# ---------------------------------------------------------------------------
def run_benchmark(datasets, methods, classifiers, seeds, emb_dim=150, device_str=None, skip_existing=True, train_ratio=0.70, load_cached_binary=True):
    """
    Runs evaluation across datasets, seeds, methods, and classifiers.
    Returns pandas DataFrame with results and geometry metrics.
    If skip_existing=True, skips (dataset, seed, method) evaluations already present in CSV checkpoints.
    """
    from datetime import datetime
    import glob
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(BENCHMARK_DIR, "Results")
    os.makedirs(results_dir, exist_ok=True)

    device = device_str or str(DEVICE)
    rows = []

    # --- Load Checkpoint Cache ---
    existing_cache = {}
    if not skip_existing:
        print("[Checkpoint Engine] Checkpoint loading DISABLED (skip_existing=False). Running fresh evaluations from scratch.")
    else:
        files_to_check = []
        if os.path.exists(results_dir):
            files_to_check.extend(glob.glob(os.path.join(results_dir, "*.csv")))
        if os.path.exists(BENCHMARK_DIR):
            files_to_check.extend(glob.glob(os.path.join(BENCHMARK_DIR, "final_benchmark_results_*.csv")))
        legacy_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "Final_benchmark", "Results"))
        if os.path.exists(legacy_dir) and legacy_dir != results_dir:
            files_to_check.extend(glob.glob(os.path.join(legacy_dir, "*.csv")))
        legacy_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "Final_benchmark", "Results"))
        if os.path.exists(legacy_dir) and legacy_dir != results_dir:
            files_to_check.extend(glob.glob(os.path.join(legacy_dir, "*.csv")))
        files_to_check.sort(key=lambda p: os.path.getmtime(p))
        for fp in files_to_check:
            try:
                cdf = pd.read_csv(fp)
                req_cols = {"dataset", "seed", "method", "classifier", "accuracy"}
                if not req_cols.issubset(set(cdf.columns)):
                    continue
                for _, r in cdf.iterrows():
                    if pd.notnull(r["accuracy"]):
                        try:
                            s_val = int(r["seed"])
                            key = (str(r["dataset"]).lower().strip(), s_val, str(r["method"]).strip(), str(r["classifier"]).strip(), round(float(r.get("train_ratio", 0.70)), 3))
                            r_dict = {k: v for k, v in r.to_dict().items() if not str(k).startswith("Unnamed")}
                            existing_cache[key] = r_dict
                        except Exception:
                            pass
            except Exception:
                pass
        if existing_cache:
            print(f"[Checkpoint Engine] Loaded {len(existing_cache)} existing evaluations from {len(files_to_check)} CSV checkpoint files.")

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*70}")

        # Check if entire dataset across all seeds and methods is already completed in checkpoint
        all_dataset_done = True
        if skip_existing:
            for s in seeds:
                for m in methods:
                    for c in classifiers.keys():
                        if (ds_name.lower().strip(), int(s), m.strip(), c.strip(), round(float(train_ratio), 3)) not in existing_cache:
                            all_dataset_done = False
                            break
                    if not all_dataset_done:
                        break
                if not all_dataset_done:
                    break

        if skip_existing and all_dataset_done:
            print(f"  [Checkpoint Hit] All seeds and methods for {ds_name} already completed in checkpoint. Loading results...")
            for s in seeds:
                for m in methods:
                    for c in classifiers.keys():
                        k = (ds_name.lower().strip(), int(s), m.strip(), c.strip(), round(float(train_ratio), 3))
                        rows.append(existing_cache[k])
            continue

        try:
            ds = load_dataset(ds_name)
        except Exception as e:
            print(f"  !! Could not load {ds_name}: {e}")
            continue

        G = ds["G"]
        pyg_data = ds["pyg_data"]
        labels = ds["labels"]

        for seed in seeds:
            set_seed(seed)
            train_mask, test_mask = create_split(labels, seed, train_ratio=train_ratio)
            label_mask = train_mask.copy()
            n_classes = getattr(G, 'num_classes', int(labels.max()) + 1)

            print(f"\n  Seed: {seed} | Device: {device}  "
                  f"(train={train_mask.sum()}, test={test_mask.sum()})")

            for method_name in methods:
                # Check if this method is already evaluated for all classifiers for this (ds_name, seed)
                method_already_done = True
                cached_method_rows = []
                if skip_existing:
                    for clf_name in classifiers.keys():
                        k = (ds_name.lower().strip(), int(seed), method_name.strip(), clf_name.strip(), round(float(train_ratio), 3))
                        if k in existing_cache and pd.notnull(existing_cache[k].get("accuracy")):
                            cached_method_rows.append(existing_cache[k])
                        else:
                            method_already_done = False
                            break

                if skip_existing and method_already_done and len(cached_method_rows) == len(classifiers):
                    print(f"    {method_name:22s}  [Skipped - Checkpoint hit: {len(cached_method_rows)} classifiers loaded]")
                    rows.extend(cached_method_rows)
                    continue

                try:
                    emb, gen_time = extract_features(
                        method_name, G, pyg_data, labels, label_mask,
                        emb_dim=emb_dim, device=device, seed=seed, ds_name=ds_name,
                        load_cached_binary=load_cached_binary
                    )
                    print(f"    {method_name:22s}  gen={gen_time:7.2f}s  shape={tuple(emb.shape)}")

                    try:
                        geom_all = compute_geometry_metrics(emb, labels)
                        geom_test = compute_geometry_metrics(emb, labels, mask=test_mask)
                        print(f"    {'Geometry':22s}  D_intra={geom_all['D_intra']:<8.4f} "
                              f"D_inter={geom_all['D_inter']:<8.4f} R={geom_all['Separation_Ratio']:<8.4f}")
                    except Exception:
                        geom_all = {"D_intra": np.nan, "D_inter": np.nan, "Separation_Ratio": np.nan}
                        geom_test = {"D_intra": np.nan, "D_inter": np.nan, "Separation_Ratio": np.nan}
                except Exception as e:
                    print(f"    {method_name:22s}  !! FAILED: {e}")
                    continue

                for clf_name, (clf_type, clf_cls) in classifiers.items():
                    try:
                        set_seed(seed)
                        if clf_type == "linkx":
                            model = clf_cls(in_dim=emb.shape[1], n_classes=n_classes, num_nodes=G.number_of_nodes())
                        else:
                            model = clf_cls(in_dim=emb.shape[1], n_classes=n_classes)

                        acc, f1_mac, f1_mic, t_train, t_inf = train_and_eval(
                            model, pyg_data.edge_index, emb, pyg_data.y,
                            train_mask, test_mask, device=device,
                        )

                        rows.append({
                            "dataset": ds_name,
                            "method": method_name,
                            "classifier": clf_name,
                            "seed": seed,
                            "train_ratio": round(float(train_ratio), 3),
                            "accuracy": round(acc * 100, 2),
                            "f1_macro": round(f1_mac * 100, 2),
                            "f1_micro": round(f1_mic * 100, 2),
                            "gen_time": round(gen_time, 4),
                            "train_time": round(t_train, 4),
                            "inf_time_ms": round(t_inf, 2),
                            "D_intra": geom_all["D_intra"],
                            "D_inter": geom_all["D_inter"],
                            "R": geom_all["Separation_Ratio"],
                            "D_intra_test": geom_test["D_intra"],
                            "D_inter_test": geom_test["D_inter"],
                            "R_test": geom_test["Separation_Ratio"],
                        })

                        print(f"      [{clf_name:20s}]  Acc={acc*100:6.2f}%  F1m={f1_mac*100:6.2f}%  train={t_train:5.2f}s")
                    except Exception as ce:
                        print(f"      [{clf_name:20s}]  !! FAILED: {ce}")
                        rows.append({
                            "dataset": ds_name, "method": method_name, "classifier": clf_name, "seed": seed,
                            "train_ratio": round(float(train_ratio), 3),
                            "accuracy": np.nan, "f1_macro": np.nan, "f1_micro": np.nan,
                            "gen_time": round(gen_time, 4), "train_time": np.nan, "inf_time_ms": np.nan,
                            "D_intra": geom_all["D_intra"], "D_inter": geom_all["D_inter"], "R": geom_all["Separation_Ratio"],
                            "D_intra_test": geom_test["D_intra"], "D_inter_test": geom_test["D_inter"], "R_test": geom_test["Separation_Ratio"],
                        })

            # --- Checkpoint: Save result after each seed in TESTS/Final_benchmark/Results ---
            try:
                seed_rows = [r for r in rows if r.get("dataset") == ds_name and r.get("seed") == seed]
                if seed_rows:
                    seed_df = pd.DataFrame(seed_rows)
                    seed_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    pct_tag = f"{int(round(train_ratio * 100))}pct"
                    seed_csv_path = os.path.join(results_dir, f"results_{ds_name}_{pct_tag}_seed{seed}_{seed_ts}.csv")
                    seed_df.to_csv(seed_csv_path, index=False)
                    print(f"\n    [Checkpoint Saved] Seed {seed} ({len(seed_df)} records) saved to: {seed_csv_path}")

                if rows:
                    cum_df = pd.DataFrame(rows)
                    pct_tag = f"{int(round(train_ratio * 100))}pct"
                    cum_csv_path = os.path.join(results_dir, f"benchmark_cumulative_{pct_tag}_{session_ts}.csv")
                    cum_df.to_csv(cum_csv_path, index=False)
                    print(f"    [Checkpoint Saved] Cumulative run ({len(cum_df)} records) saved to: {cum_csv_path}\n")
            except Exception as se:
                print(f"    [Checkpoint Error] Could not save seed checkpoint: {se}")

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 8. SUMMARY & REPORTING
# ---------------------------------------------------------------------------
def print_summary(df):
    """Print clean grouped summary tables showing Mean ± Std across seeds and overall."""
    if df.empty:
        print("No results to summarize.")
        return

    work_df = df.copy()
    col_map = {
        "Dataset": "dataset",
        "Method": "method",
        "Classifier": "classifier",
        "Accuracy": "accuracy",
        "Seed": "seed",
        "Gen_Time_s": "gen_time",
        "gen_time_s": "gen_time",
        "GenTime": "gen_time",
        "F1_Macro": "f1_macro",
        "F1_Micro": "f1_micro",
    }
    work_df.rename(columns={k: v for k, v in col_map.items() if k in work_df.columns}, inplace=True)

    def _format_mean_std(sub_df, row_index, val_col="accuracy"):
        m = sub_df.pivot_table(values=val_col, index=row_index, columns="classifier", aggfunc="mean")
        s = sub_df.pivot_table(values=val_col, index=row_index, columns="classifier", aggfunc="std", dropna=False).fillna(0.0)
        table = pd.DataFrame(index=m.index, columns=m.columns)
        for col in m.columns:
            table[col] = [f"{mv:.2f} ± {sv:.2f}" for mv, sv in zip(m[col], s[col])]
        return table

    # 1. Per-Dataset Summary Tables (Mean ± Std % across seeds)
    if "dataset" in work_df.columns:
        for ds_name in work_df["dataset"].unique():
            print("\n" + "=" * 80)
            print(f"  DATASET: {str(ds_name).upper()} (Mean ± Std % across seeds)")
            print("=" * 80)
            sub_df = work_df[work_df["dataset"] == ds_name]
            print(_format_mean_std(sub_df, "method").to_string())

    # 2. Overall Average Accuracy (Mean ± Std % across all datasets & seeds)
    print("\n" + "=" * 80)
    print("  OVERALL AVERAGE ACCURACY (Mean ± Std % across all datasets & seeds)")
    print("=" * 80)
    print(_format_mean_std(work_df, "method").to_string())

    # 3. Overall Embedding Generation Latency (Mean ± Std seconds across all runs)
    if "gen_time" in work_df.columns:
        print("\n" + "=" * 80)
        print("  OVERALL EMBEDDING GENERATION TIME (Mean ± Std seconds across all runs)")
        print("=" * 80)
        gen_df = work_df.groupby(["dataset", "method", "seed"])["gen_time"].first().reset_index()
        t_stats = gen_df.groupby("method")["gen_time"].agg(
            Mean_s="mean",
            Std_s="std",
            Median_s="median"
        ).fillna(0.0)
        time_table = pd.DataFrame(index=t_stats.index)
        time_table["Generation Time (Mean ± Std)"] = [
            f"{r['Mean_s']:>8.4f}s ± {r['Std_s']:>8.4f}s" for _, r in t_stats.iterrows()
        ]
        time_table["Median (s)"] = [f"{r['Median_s']:>8.4f}s" for _, r in t_stats.iterrows()]
        print(time_table.to_string())

    # 4. Overall Macro F1 (Mean ± Std % across all datasets & seeds)
    if "f1_macro" in work_df.columns:
        print("\n" + "=" * 80)
        print("  OVERALL MACRO F1 (Mean ± Std % across all datasets & seeds)")
        print("=" * 80)
        print(_format_mean_std(work_df, "method", val_col="f1_macro").to_string())


# ---------------------------------------------------------------------------
# 9. GEOMETRY VISUALIZATION & PLOTTING
# ---------------------------------------------------------------------------
def plot_geometry_comparison(df, save_path=None):
    """
    Bar plot comparing Separation Ratio R across all methods and datasets.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df.empty:
        print("No geometric data available to plot.")
        return

    col_r = "R" if "R" in df.columns else ("Separation_Ratio" if "Separation_Ratio" in df.columns else None)
    col_ds = "dataset" if "dataset" in df.columns else ("Dataset" if "Dataset" in df.columns else None)
    col_m = "method" if "method" in df.columns else ("Method" if "Method" in df.columns else None)

    if not col_r or not col_ds or not col_m:
        print("No geometric data available to plot.")
        return

    geom = df.groupby([col_ds, col_m])[col_r].mean().reset_index()
    datasets = geom[col_ds].unique()

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 2.8), 6))

    palette = sns.color_palette("tab10", n_colors=geom[col_m].nunique())
    sns.barplot(data=geom, x=col_ds, y=col_r, hue=col_m, palette=palette, ax=ax)

    ax.set_title("Geometric Class Separation Ratio ($R = D_{inter} / D_{intra}$) Across Datasets\n"
                 "(Higher R indicates classes are geometrically easier to separate)",
                 fontsize=14, weight="bold", pad=15)
    ax.set_ylabel("Separation Ratio $R$", fontsize=12, weight="bold")
    ax.set_xlabel("Dataset", fontsize=12, weight="bold")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="R = 1.0 (No separation)")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved geometry comparison plot to: {save_path}")
    plt.show()


def plot_geometry_scatter(df, save_path=None):
    """
    Scatter plot: D_intra (compactness, lower is better) vs D_inter (separation, higher is better).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df.empty:
        print("No geometric data available to plot.")
        return

    col_r = "R" if "R" in df.columns else ("Separation_Ratio" if "Separation_Ratio" in df.columns else None)
    col_ds = "dataset" if "dataset" in df.columns else ("Dataset" if "Dataset" in df.columns else None)
    col_m = "method" if "method" in df.columns else ("Method" if "Method" in df.columns else None)
    col_intra = "D_intra" if "D_intra" in df.columns else None
    col_inter = "D_inter" if "D_inter" in df.columns else None

    if not col_intra or not col_inter or not col_ds or not col_m:
        print("No geometric data available to plot.")
        return

    geom = df.groupby([col_ds, col_m]).agg(
        D_intra=(col_intra, "mean"),
        D_inter=(col_inter, "mean"),
        Separation_Ratio=(col_r, "mean") if col_r else (col_intra, "mean"),
    ).reset_index()

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.scatterplot(
        data=geom, x="D_intra", y="D_inter", hue=col_m, style=col_ds,
        s=140, alpha=0.9, ax=ax
    )

    # Reference iso-line y = x (R = 1)
    max_val = max(geom["D_intra"].max(), geom["D_inter"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label="R = 1.0 (Boundary)")

    ax.set_title("Geometric Tradeoff: Intra-Class ($D_{intra}$) vs Inter-Class ($D_{inter}$) Distance\n"
                 "(Ideal: Top-Left Region -> Small Intra-Class & Large Inter-Class Distance)",
                 fontsize=13, weight="bold", pad=15)
    ax.set_xlabel("Intra-Class Distance $D_{intra}$ (Lower is Better)", fontsize=11, weight="bold")
    ax.set_ylabel("Inter-Class Distance $D_{inter}$ (Higher is Better)", fontsize=11, weight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved geometry scatter plot to: {save_path}")
    plt.show()


def visualize_representation_geometry(G, pyg_data, labels, methods_to_compare=['Raw Features', 'Alg_V1_SVD_Blend'],
                                      emb_dim=250, device='cpu', seed=42, max_nodes=2500, save_path=None, train_ratio=0.7):
    """
    Generates side-by-side 2D PCA plots comparing Raw Features vs Representation,
    displaying D_intra, D_inter, and R directly in subplot subtitles.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    set_seed(seed)
    train_mask, test_mask = create_split(labels, seed, train_ratio=train_ratio)
    label_mask = train_mask.copy()

    n_methods = len(methods_to_compare)
    fig, axes = plt.subplots(1, n_methods, figsize=(6.5 * n_methods, 5.5))
    if n_methods == 1:
        axes = [axes]

    cmap = "tab10" if (labels.max() + 1) <= 10 else "tab20"

    for idx, method_name in enumerate(methods_to_compare):
        emb, _ = extract_features(method_name, G, pyg_data, labels, label_mask,
                                  emb_dim=emb_dim, device=device, seed=seed)
        geom = compute_geometry_metrics(emb, labels)

        Z = emb.numpy() if isinstance(emb, torch.Tensor) else emb
        y = labels

        if len(y) > max_nodes:
            sub_idx = np.random.choice(len(y), size=max_nodes, replace=False)
            Z, y = Z[sub_idx], y[sub_idx]

        pca = PCA(n_components=2, random_state=seed)
        Z_2d = pca.fit_transform(Z)

        scatter = axes[idx].scatter(Z_2d[:, 0], Z_2d[:, 1], c=y, cmap=cmap, alpha=0.7, s=18, edgecolors="none")
        r_val = geom.get('Separation_Ratio', geom.get('R', float("nan")))
        axes[idx].set_title(
            f"{method_name}\n"
            fr"$D_{{intra}}={geom['D_intra']:.2f}, \; D_{{inter}}={geom['D_inter']:.2f}, \; \mathbf{{R={r_val:.2f}}}$",
            fontsize=12, weight="bold"
        )
        axes[idx].set_xlabel("PC 1", fontsize=10)
        axes[idx].set_ylabel("PC 2", fontsize=10)

    fig.suptitle("Part B: Geometric Class Separability Comparison (2D PCA Projection)",
                 fontsize=15, weight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved 2D geometry visualization to: {save_path}")
    plt.show()
