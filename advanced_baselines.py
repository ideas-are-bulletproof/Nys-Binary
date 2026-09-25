"""
Advanced Baselines & Proposed Methods (Cleaned for Final Grand Benchmark)
Contains:
- Alg_V1_SVD_Blend
- Alg_V1_Nystrom_Blend
- Alg_V1_SVD_Ablation (for Scientific Ablation)
- Bi-GCN
- NodeSig
- NodeSketch
- node2binary
- Geometry Metrics
"""

from __future__ import annotations
import os
import sys
import time
import subprocess
from typing import Optional, Tuple, Dict, Any

import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import scipy.sparse.linalg as sla
from sklearn.utils.extmath import randomized_svd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_geometric
    import torch_geometric.typing
except ImportError:
    pass

# Path to cloned external repos
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
BENCHMARK_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Final_benchmark"))
BIGCN_DIR = os.path.join(BENCHMARK_DIR, "Bi-GCN")
NODESIG_DIR = os.path.join(BENCHMARK_DIR, "NodeSig")
N2B_DIR = os.path.join(BENCHMARK_DIR, "node2binary")

# =============================================================================
# Helper: Graph Adjacency Torch Tensor
# =============================================================================
def _get_A_torch(G: nx.Graph, device: str = "cpu"):
    """Returns row-stochastic normalized adjacency (SciPy CSR & Torch Sparse COO)."""
    A_scipy = nx.to_scipy_sparse_array(G, format="csr")
    A_scipy.setdiag(1.0)
    degrees = np.array(A_scipy.sum(axis=1)).flatten()
    D_inv = 1.0 / (degrees + 1e-10)
    D_inv_mat = sp.diags(D_inv)
    A_norm = D_inv_mat.dot(A_scipy)

    A_norm_coo = A_norm.tocoo()
    indices = torch.tensor(
        np.vstack((A_norm_coo.row, A_norm_coo.col)), dtype=torch.long, device=device
    )
    values = torch.tensor(A_norm_coo.data, dtype=torch.float32, device=device)
    A_torch = torch.sparse_coo_tensor(
        indices, values, torch.Size(A_norm_coo.shape), device=device
    )
    return A_norm, A_torch

def _get_orthogonal_signatures(num_items: int, dim: int, dev: str = "cpu") -> torch.Tensor:
    """Generates orthogonal or normalized random signatures for class coding."""
    if dim <= 0:
        return torch.empty((num_items, 0), device=dev)
    X = torch.randn(num_items, dim, device=dev)
    if num_items >= dim:
        Q, _ = torch.linalg.qr(X)
    else:
        Q = X / (torch.norm(X, dim=1, keepdim=True) + 1e-10)
    return Q

# =============================================================================
# Core Blend Logic for Proposed Algebraic Hashing
# =============================================================================
def get_v1_blend_from_r_struct(
    R_struct: Optional[torch.Tensor],
    G: nx.Graph,
    pyg_data: Any,
    labels: Any,
    label_mask: Any,
    k: int = 250,
    hops: int = 2,
    alpha: float = 0.5,
    pseudo_threshold: float = 0.5,
    device: str = "cpu",
    A_torch: Optional[torch.Tensor] = None
) -> np.ndarray:
    """
    Blends structural signatures with training label diffusion and confident pseudo-labels.
    Supports arbitrary structural vs. label bit ratios (struct_ratio in [0.0, 1.0]).
    Guarantees strict zero test label leakage.
    """
    n = G.number_of_nodes()
    k_struct = R_struct.shape[1] if R_struct is not None else 0
    k_label = max(0, k - k_struct)

    # Safe class count for both 1D and 2D multi-label tensors
    is_multilabel = False
    if hasattr(labels, "ndim") and labels.ndim == 2:
        is_multilabel = True
        n_classes = labels.shape[1]
    elif isinstance(labels, torch.Tensor) and labels.dim() == 2:
        is_multilabel = True
        n_classes = labels.shape[1]
    elif isinstance(labels, torch.Tensor):
        n_classes = int(labels.max().item()) + 1
    else:
        n_classes = int(np.max(labels)) + 1

    labels_t = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)
    label_mask_t = label_mask if isinstance(label_mask, torch.Tensor) else torch.tensor(label_mask, dtype=torch.bool, device=device)

    if k_label > 0 and torch.any(label_mask_t):
        class_signatures = torch.sign(_get_orthogonal_signatures(n_classes, k_label, dev=device))

        # Channel A: Ground-truth training labels strictly on label_mask
        S_gt = torch.zeros(n, k_label, dtype=torch.float32, device=device)
        Y_0 = torch.zeros(n, n_classes, dtype=torch.float32, device=device)
        
        if is_multilabel:
            Y_0[label_mask_t] = labels_t[label_mask_t].float()
            S_gt[label_mask_t] = torch.sign(Y_0[label_mask_t] @ class_signatures)
        else:
            Y_0[label_mask_t] = F.one_hot(labels_t[label_mask_t].long(), num_classes=n_classes).float()
            S_gt[label_mask_t] = class_signatures[labels_t[label_mask_t].long()]

        # Channel B: Pseudo-labels via 3-step random walk diffusion from training seeds
        S_v7 = S_gt.clone()
        if alpha > 0.0:
            Y_t = Y_0
            for _ in range(3):
                Y_t = torch.sparse.mm(A_torch, Y_t)
            probs = Y_t / (Y_t.sum(dim=1, keepdim=True) + 1e-10)
            max_probs, pseudo_c = torch.max(probs, dim=1)
            confident = (~label_mask_t) & (max_probs >= pseudo_threshold)

            if torch.any(confident):
                S_v7[confident] = class_signatures[pseudo_c[confident]]
    else:
        S_gt = None
        S_v7 = None

    # Channel Assembly
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
        raise ValueError("Both R_struct and label channel are empty. Check k, struct_ratio, and labels.")

    # Polynomial diffusion over graph topology
    def diffuse(W_init: torch.Tensor) -> torch.Tensor:
        W_acc = W_init.clone()
        curr = W_init
        for _ in range(2):
            curr = torch.sparse.mm(A_torch, curr)
            W_acc += curr
        for _ in range(hops):
            W_acc = torch.sparse.mm(A_torch, W_acc)
        return W_acc

    W_svd_acc = diffuse(W_svd_init)
    if S_v7 is not None and alpha > 0.0:
        W_v7_acc = diffuse(W_v7_init)
        W_fused = (1.0 - alpha) * W_svd_acc + alpha * W_v7_acc
    else:
        W_fused = W_svd_acc

    # Convex combination and adaptive binarization threshold
    col_medians = torch.median(W_fused, dim=0).values
    threshold = alpha * col_medians

    return (W_fused > threshold).float().cpu().numpy()



# =============================================================================
# 2. Proposed Method: Alg_V1_Nystrom_Blend [Nys Binary]
# =============================================================================

def fuse_alg_v1_nystrom_blend(
    G: nx.Graph,
    pyg_data: Any,
    labels: Any,
    label_mask: Any,
    k: int = 250,
    hops: int = 3,
    alpha: float = 0.5,
    pseudo_threshold: float = 0.5,
    device: str = "cpu",
    seed: int = 42,
    m_landmarks: int = 125,
    struct_ratio: float = 0.5,
    k_struct: Optional[int] = None
) -> np.ndarray:
    """Nystrom Landmark Structural Signatures + Label Diffusion Blend.
    
    Parameters:
    -----------
    struct_ratio : float, default=0.5
        Fraction of total budget k allocated to structural features (e.g. 0.5 for 50/50, 0.7 for 70/30).
    k_struct : int, optional
        Explicit number of structural bits. If provided, overrides struct_ratio.
    """
    np.random.seed(seed)
    A_norm, A_torch = _get_A_torch(G, device)

    n = G.number_of_nodes()

    if k_struct is None:
        if struct_ratio >= 1.0:
            k_struct = k
        elif struct_ratio <= 0.0:
            k_struct = 0
        else:
            k_struct = int(round(k * struct_ratio))
    k_struct = max(0, min(k, k_struct))

    if k_struct > 0:
        m = min(n, max(k_struct, m_landmarks))
        landmark_idx = np.random.choice(n, m, replace=False)

        A_cc = A_norm[landmark_idx, :][:, landmark_idx].toarray()
        U_c, S_c, _ = np.linalg.svd(A_cc)
        U_c = U_c[:, :k_struct]
        S_c = S_c[:k_struct]
        S_inv = np.diag(1.0 / (S_c + 1e-10))

        A_nc = A_norm[:, landmark_idx]
        U_approx = A_nc.dot(U_c).dot(S_inv)
        R_struct = torch.tensor(np.ascontiguousarray(U_approx), dtype=torch.float32, device=device)
        if R_struct.shape[1] < k_struct:
            pad = torch.zeros(n, k_struct - R_struct.shape[1], device=device)
            R_struct = torch.cat([R_struct, pad], dim=1)
    else:
        R_struct = None

    return get_v1_blend_from_r_struct(
        R_struct, G, pyg_data, labels, label_mask,
        k=k, hops=hops, alpha=alpha, pseudo_threshold=pseudo_threshold,
        device=device, A_torch=A_torch
    )

# =============================================================================
#  Other Proposed Method: Alg_V1_SVD_Blend 
# =============================================================================
def fuse_alg_v1_svd_blend(
    G: nx.Graph,
    pyg_data: Any,
    labels: Any,
    label_mask: Any,
    k: int = 250,
    hops: int = 2,
    alpha: float = 0.5,
    pseudo_threshold: float = 0.5,
    device: str = "cpu",
    struct_ratio: float = 0.5,
    k_struct: Optional[int] = None
) -> np.ndarray:
    """Exact SVD Structural Signatures + Label Diffusion Blend.
    
    Parameters:
    -----------
    struct_ratio : float, default=0.5
        Fraction of total budget k allocated to structural features (e.g. 0.5 for 50/50).
    k_struct : int, optional
        Explicit number of structural bits. If provided, overrides struct_ratio.
    """
    n = G.number_of_nodes()
    if k_struct is None:
        if struct_ratio >= 1.0:
            k_struct = k
        elif struct_ratio <= 0.0:
            k_struct = 0
        else:
            k_struct = int(round(k * struct_ratio))
    k_struct = max(0, min(k, k_struct))

    A_norm, A_torch = _get_A_torch(G, device)

    if k_struct > 0:
        try:
            if k_struct < n - 1:
                U, _, _ = sla.svds(A_norm, k=k_struct, which='LM')
            else:
                U, _, _ = randomized_svd(A_norm, n_components=k_struct, random_state=42)
        except Exception:
            U, _, _ = randomized_svd(A_norm, n_components=k_struct, random_state=42)
        R_struct = torch.tensor(np.ascontiguousarray(U), dtype=torch.float32, device=device)
    else:
        R_struct = None

    return get_v1_blend_from_r_struct(
        R_struct, G, pyg_data, labels, label_mask,
        k=k, hops=hops, alpha=alpha, pseudo_threshold=pseudo_threshold,
        device=device, A_torch=A_torch
    )
    
# =============================================================================
#  Other Proposed Method: Alg_V1_randomized_blend 
# =============================================================================
def fuse_alg_v1_randomized_blend(
    G: nx.Graph,
    pyg_data: Any,
    labels: Any,
    label_mask: Any,
    k: int = 250,
    hops: int = 3,
    alpha: float = 0.5,
    pseudo_threshold: float = 0.5,
    device: str = "cpu",
    seed: int = 42,
    struct_ratio: float = 0.5,
    k_struct: Optional[int] = None
) -> np.ndarray:
    """Random Projection Structural Signatures + Label Diffusion Blend."""
    np.random.seed(seed)
    n = G.number_of_nodes()
    if k_struct is None:
        if struct_ratio >= 1.0:
            k_struct = k
        elif struct_ratio <= 0.0:
            k_struct = 0
        else:
            k_struct = int(round(k * struct_ratio))
    k_struct = max(0, min(k, k_struct))

    A_norm, A_torch = _get_A_torch(G, device)

    if k_struct > 0:
        # Gaussian random projection: A_norm @ Omega  (single sparse-dense multiply)
        Omega = np.random.randn(n, k_struct).astype(np.float32) / np.sqrt(k_struct)
        Y = A_norm.dot(Omega)

        # QR orthogonalization for numerical stability
        Q, _ = np.linalg.qr(Y)
        R_struct = torch.tensor(np.ascontiguousarray(Q), dtype=torch.float32, device=device)
    else:
        R_struct = None

    return get_v1_blend_from_r_struct(
        R_struct, G, pyg_data, labels, label_mask,
        k=k, hops=hops, alpha=alpha, pseudo_threshold=pseudo_threshold,
        device=device, A_torch=A_torch
    )

#=============================================================================
# 3. External Baseline: Bi-GCN (CVPR 2021 / TPAMI 2024)
# =============================================================================
# Import official layers directly from cloned Bi-GCN repo
sys.path.insert(0, BIGCN_DIR)
try:
    from layers import BiGCNConv as OfficialBiGCNConv
    from function import BinActive as OfficialBinActive
except ImportError:
    class OfficialBinActive(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.sign(input)
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input.abs() > 1] = 0
            return grad_input
    import torch_geometric.nn as geom_nn
    class OfficialBiGCNConv(nn.Module):
        def __init__(self, in_channels, out_channels, cached=True, bi=True):
            super().__init__()
            self.conv = geom_nn.GCNConv(in_channels, out_channels, cached=cached)
        def forward(self, x, edge_index):
            bw = OfficialBinActive.apply(self.conv.lin.weight)
            self.conv.lin.weight.data = bw
            bx = OfficialBinActive.apply(x)
            return self.conv(bx, edge_index)
sys.path.pop(0)

def generate_bigcn_embeddings(
    pyg_data: Any,
    train_mask: Any,
    k: int = 250,
    epochs: int = 200,
    lr: float = 0.01,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Trains official Bi-GCN on train_mask and extracts the intermediate binary embeddings.
    """
    n = pyg_data.num_nodes
    edge_index = pyg_data.edge_index.to(device)

    # Use actual node features if present, otherwise structural features
    if getattr(pyg_data, 'x', None) is not None and pyg_data.x.numel() > 0:
        x = pyg_data.x.to(device).float()
    else:
        # Fallback to random orthogonal features if graph is featureless
        x = _get_orthogonal_signatures(n, k, dev=device)

    in_dim = x.shape[1]
    y = pyg_data.y.to(device)
    tr_m = train_mask if isinstance(train_mask, torch.Tensor) else torch.tensor(train_mask, dtype=torch.bool, device=device)

    is_multilabel = (y.dim() == 2)
    if is_multilabel:
        n_classes = int(y.shape[1])
        criterion = nn.BCEWithLogitsLoss()
        y_target = y.float()
    else:
        n_classes = int(y.max().item()) + 1
        criterion = nn.CrossEntropyLoss()
        y_target = y.long()

    conv1 = OfficialBiGCNConv(in_dim, k, cached=True, bi=True).to(device)
    conv2 = OfficialBiGCNConv(k, n_classes, cached=True, bi=True).to(device)
    opt = torch.optim.Adam(list(conv1.parameters()) + list(conv2.parameters()), lr=lr)

    bin_active = OfficialBinActive()
    conv1.train()
    conv2.train()
    for _ in range(epochs):
        opt.zero_grad()
        x_b = bin_active(x)
        h1 = conv1(x_b, edge_index)
        h2_b = bin_active(h1)
        out = conv2(h2_b, edge_index)
        loss = criterion(out[tr_m], y_target[tr_m])
        loss.backward()
        opt.step()

    conv1.eval()
    with torch.no_grad():
        x_b = bin_active(x)
        h1 = conv1(x_b, edge_index)
        h2_b = bin_active(h1)
    h2_bin = h2_b

    # Map to {0, 1} representation from {-1, 1} for consistent binary storage
    emb = (h2_bin > 0).float()
    return emb.detach().cpu()

# =============================================================================
# 4. External Baseline: node2binary (WWW 2025)
# =============================================================================
def generate_node2binary_embeddings(
    G: nx.Graph,
    pyg_data: Any,
    k: int = 250,
    epochs: int = 1000,
    device: str = "cpu",
    timeout: int = 1800,
    load_cache: Optional[bool] = None
) -> np.ndarray:
    """
    Subprocess caller for official node2binary (WWW 2025) Leiden random-walk binary embeddings.
    """
    repo_path = N2B_DIR
    data_dir = os.path.join(repo_path, "data")
    leiden_dir = os.path.join(data_dir, "Using_Leiden")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(leiden_dir, exist_ok=True)

    n_nodes = pyg_data.num_nodes
    n_edges = pyg_data.edge_index.shape[1]
    base_name = f"graph_N{n_nodes}_E{n_edges}"

    use_cache = LOAD_CACHED_BINARY if load_cache is None else bool(load_cache)
    cache_npy = os.path.join(leiden_dir, f"{base_name}_k{k}_ep{epochs}.npy")
    fallback_n2b = os.path.join(N2B_DIR, "data", "Using_Leiden", f"{base_name}_k{k}_ep{epochs}.npy")
    # Load cached node2binary embeddings if present to avoid re-training
    if use_cache:
        if os.path.exists(cache_npy):
            print(f"[node2binary] Loaded cached {k}-bit embeddings for {base_name}.")
            return np.load(cache_npy)
        elif os.path.exists(fallback_n2b):
            print(f"[node2binary] Loaded cached {k}-bit embeddings for {base_name} from fallback cache.")
            return np.load(fallback_n2b)
    else:
        print(f"[node2binary] Cache loading disabled (load_cache=False). Re-generating {k}-bit embeddings for {base_name} from scratch...")

    edgelist_path = os.path.join(data_dir, f"{base_name}.edgelist")
    if not os.path.exists(edgelist_path):
        edges = pyg_data.edge_index.cpu().numpy()
        with open(edgelist_path, "w", encoding="utf-8", newline="\n") as f:
            for i in range(edges.shape[1]):
                f.write(f"{edges[0, i]} {edges[1, i]}\n")

    # node2binary official script expects a labels file for NodeClassification evaluation
    # We provide dummy unsupervised labels (zeros) so there is strictly ZERO label leakage
    labels_path = os.path.join(data_dir, f"{base_name}_labels.txt")
    if not os.path.exists(labels_path):
        with open(labels_path, "w", encoding="utf-8", newline="\n") as f:
            for i in range(n_nodes):
                f.write(f"{i}\t{i % 3}\n")

    cmd = [
        sys.executable, "node2binary.py",
        f"data/{base_name}.edgelist",
        str(k), "25", "10", "1", "0", "0.008", "0.01", "8", "1",
        "--trees", "1", "--depth", "3", "--iterations", str(epochs),
        "--task", "NodeClassification", "--labels_path", f"data/{base_name}_labels.txt",
        "--testing_ratio", "0.2",
        "--batchsize", "5000", "--stop-width", "9999", "--verbose", "0.05", "--closed"
    ]

    print(f"[node2binary] Executing official script: {' '.join(cmd[:4])} (timeout={timeout}s / {timeout/60:.0f} min) ...", flush=True)
    t_start = time.time()
    proc = subprocess.Popen(cmd, cwd=repo_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
        if time.time() - t_start > timeout:
            proc.kill()
            raise TimeoutError(f"node2binary exceeded timeout threshold of {timeout}s ({timeout/60:.0f} minutes).")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"node2binary failed with returncode {proc.returncode}")

    emb_path = os.path.join(leiden_dir, f"{base_name}_t1_d3.embeddings.txt")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"node2binary output file not found: {emb_path}")

    B = np.zeros((n_nodes, k), dtype=np.float32)
    with open(emb_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                node_id = int(parts[0])
                vals = [1.0 if x.lower() == 'true' else (0.0 if x.lower() == 'false' else float(x)) for x in parts[1:]]
                if node_id < n_nodes:
                    B[node_id] = vals
            except ValueError:
                continue

    np.save(cache_npy, B)
    return B

# =============================================================================
# 5. External Baseline: NodeSig (IEEE/ACM ASONAM 2022)
# =============================================================================
def generate_official_nodesig_embeddings(
    pyg_data: Any,
    train_mask: Any,
    k: int = 250,
    hops: int = 3,
    device: str = "cpu",
    load_cache: Optional[bool] = None
) -> torch.Tensor:
    """
    Subprocess caller for official C++ compiled NodeSig executable.
    """
    ns_dir = NODESIG_DIR
    tmp_dir = os.path.join(ns_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    use_cache = LOAD_CACHED_BINARY if load_cache is None else bool(load_cache)
    k_nodesig = int(((k + 7) // 8) * 8)
    in_file = os.path.join(tmp_dir, f"in_graph_{pyg_data.num_nodes}_{k_nodesig}.txt")
    out_file = os.path.join(tmp_dir, f"out_emb_{pyg_data.num_nodes}_{k_nodesig}.bin")
    fallback_ns = os.path.join(NODESIG_DIR, "tmp", f"out_emb_{pyg_data.num_nodes}_{k_nodesig}.bin")
    if use_cache:
        if not os.path.exists(out_file) and os.path.exists(fallback_ns):
            import shutil
            shutil.copy2(fallback_ns, out_file)
            print(f"[NodeSig] Restored cached binary embeddings from fallback for N={pyg_data.num_nodes}.")
    else:
        if os.path.exists(out_file):
            try:
                os.remove(out_file)
            except OSError:
                pass
        print(f"[NodeSig] Cache loading disabled (load_cache=False). Generating fresh binary embeddings for N={pyg_data.num_nodes}...")

    if not os.path.exists(in_file) or os.path.getsize(in_file) == 0:
        edges = pyg_data.edge_index.cpu().numpy()
        u, v = edges[0], edges[1]
        mask = (u <= v)
        with open(in_file, 'w', newline='\n') as f:
            for src, dst in zip(u[mask], v[mask]):
                f.write(f"{src} {dst}\n")

    if not os.path.exists(out_file) or os.path.getsize(out_file) == 0:
        exe_path = os.path.join(ns_dir, "nodesig.exe")
        cmd = [exe_path, "--edgefile", in_file, "--embfile", out_file, "--walklen", str(hops), "--dim", str(k_nodesig)]
        res = subprocess.run(cmd, cwd=ns_dir, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"NodeSig execution failed: {res.stderr}")

    with open(out_file, 'rb') as f:
        num_nodes = int.from_bytes(f.read(4), byteorder='little')
        dim = int.from_bytes(f.read(4), byteorder='little')
        raw_data = np.frombuffer(f.read(num_nodes * (dim // 8)), dtype=np.uint8)
        unpacked = np.unpackbits(raw_data).reshape(num_nodes, dim)
        embs = torch.from_numpy(unpacked.astype(np.float32))

    if embs.shape[1] > k:
        embs = embs[:, :k]

    return embs.to(device)

# =============================================================================
# =============================================================================
# 6. External Baseline: NodeSketch (ACM SIGKDD 2019)
# =============================================================================
NODESKETCH_DIR = os.path.join(BENCHMARK_DIR, "NodeSketch")

MATLAB_EXE = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"


def generate_official_nodesketch_embeddings(
    pyg_data: Any,
    k: int = 250,
    order: int = 3,
    alpha: float = 0.01,
    seed: int = 42,
    matlab_exe: str = MATLAB_EXE,
    nodesketch_dir: str = NODESKETCH_DIR,
    load_cache: Optional[bool] = None
) -> torch.Tensor:
    """
    Invokes author's official MATLAB + C MEX NodeSketch (KDD 2019) implementation.
    """
    n = pyg_data.num_nodes
    edge_index = pyg_data.edge_index.cpu().numpy()
    row, col = edge_index[0], edge_index[1]
    data = np.ones(len(row), dtype=np.float64)
    A = sp.csr_matrix((data, (row, col)), shape=(n, n))
    A = A.maximum(A.T)

    use_cache = LOAD_CACHED_BINARY if load_cache is None else bool(load_cache)
    cache_dir = os.path.join(nodesketch_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"nodesketch_N{n}_k{k}_ord{order}_s{seed}.npy")
    fallback_nsk = os.path.join(BENCHMARK_DIR, "NodeSketch", "cache", f"nodesketch_N{n}_k{k}_ord{order}_s{seed}.npy")
    if use_cache:
        if not os.path.exists(cache_file) and os.path.exists(fallback_nsk):
            import shutil
            shutil.copy2(fallback_nsk, cache_file)
        if os.path.exists(cache_file):
            print(f"[NodeSketch] Loaded cached {k}-bit embeddings for N={n} (seed={seed}).")
            cached_data = np.load(cache_file)
            # Modulo 2 fix: ensure strictly 1-bit binary embeddings in {0.0, 1.0}
            if cached_data.max() > 1.0 or cached_data.min() < 0.0 or not np.isin(cached_data, [0.0, 1.0]).all():
                cached_data = ((cached_data % 2) == 1).astype(np.float32)
                np.save(cache_file, cached_data)
            else:
                cached_data = cached_data.astype(np.float32)
            return torch.tensor(cached_data, dtype=torch.float32)
    else:
        print(f"[NodeSketch] Cache loading disabled (load_cache=False). Generating fresh binary embeddings via MATLAB for N={n} (seed={seed})...")

    pid = os.getpid()
    ts = int(time.time() * 1000)
    in_file = os.path.join(nodesketch_dir, f"_temp_in_{pid}_{seed}_{ts}.mat")
    out_file = os.path.join(nodesketch_dir, f"_temp_out_{pid}_{seed}_{ts}.mat")

    try:
        sio.savemat(in_file, {'A': A})
        cmd_str = f"cd '{nodesketch_dir}'; rng({seed}); wrapper_nodesketch('{in_file}', '{out_file}', {k}, {order}, {alpha});"
        res = subprocess.run([matlab_exe, "-batch", cmd_str], capture_output=True, text=True, check=True)
        if not os.path.exists(out_file):
            raise RuntimeError(f"NodeSketch output file not created. MATLAB STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        mat_data = sio.loadmat(out_file)
        raw_embs = mat_data['embs']
        # Modulo 2 fix: 1-bit minwise hashing (parity / LSB binarization)
        bin_embs = ((raw_embs % 2) == 1).astype(np.float32)
        np.save(cache_file, bin_embs)
        return torch.tensor(bin_embs, dtype=torch.float32)
    finally:
        for f in [in_file, out_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


# =============================================================================
# Geometry Metrics
# =============================================================================
def compute_geometry_metrics(Z: Any, y: Any, mask: Optional[Any] = None) -> Dict[str, Any]:
    """Computes exact D_intra, D_inter, and Separation Ratio R."""
    if isinstance(Z, torch.Tensor):
        Z = Z.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        Z = Z[mask]
        y = y[mask]

    n, d = Z.shape
    if n <= 1:
        return {"D_intra": 0.0, "D_inter": 0.0, "Separation_Ratio": 1.0, "N_nodes": n, "N_classes": len(np.unique(y)) if n > 0 else 0}

    # If y is 2D multi-label, take argmax for geometric class partitioning
    if y.ndim == 2:
        y_label = y.argmax(axis=1)
    else:
        y_label = y

    classes, counts = np.unique(y_label, return_counts=True)
    global_mean = Z.mean(axis=0)
    total_ss = np.sum((Z - global_mean) ** 2)
    sum_total = 2.0 * n * total_ss
    n_total = n * (n - 1)

    sum_intra = 0.0
    n_intra = 0
    for c, count in zip(classes, counts):
        if count > 1:
            Z_c = Z[y_label == c]
            ss_c = np.sum((Z_c - Z_c.mean(axis=0)) ** 2)
            sum_intra += 2.0 * count * ss_c
            n_intra += count * (count - 1)

    d_intra = (sum_intra / n_intra) if n_intra > 0 else 0.0
    sum_inter = sum_total - sum_intra
    n_inter = n_total - n_intra
    d_inter = (sum_inter / n_inter) if n_inter > 0 else 0.0
    r = d_inter / (d_intra + 1e-12)

    return {
        "D_intra": round(float(d_intra), 4),
        "D_inter": round(float(d_inter), 4),
        "Separation_Ratio": round(float(r), 4),
        "N_nodes": int(n),
        "N_classes": int(len(classes)),
    }


def fuse_alg_v1_svd_ablation(
    stage_name: str,
    G: nx.Graph,
    pyg_data: Any,
    labels: Any,
    label_mask: Any,
    k: int = 250,
    hops: int = 3,
    alpha: float = 0.5,
    device: str = "cpu",
    seed: int = 42
) -> np.ndarray:
    """Ablation logic dynamically re-constructed for missing methods."""
    np.random.seed(seed)
    A_norm, A_torch = _get_A_torch(G, device)
    
    k_struct = k // 2
    n = G.number_of_nodes()
    
    # 1. Structural
    m = min(n, max(k_struct * 2, 500))
    landmark_idx = np.random.choice(n, m, replace=False)
    A_cc = A_norm[landmark_idx, :][:, landmark_idx].toarray()
    U_c, S_c, _ = np.linalg.svd(A_cc)
    U_c = U_c[:, :k_struct]
    S_c = S_c[:k_struct]
    S_inv = np.diag(1.0 / (S_c + 1e-10))
    A_nc = A_norm[:, landmark_idx]
    U_approx = A_nc.dot(U_c).dot(S_inv)
    R_struct = torch.tensor(np.ascontiguousarray(U_approx), dtype=torch.float32, device=device)
    
    # 2. Labels
    k_label = k - k_struct
    
    # Handle both 1D and 2D multi-label tensors safely
    is_multilabel = False
    if hasattr(labels, "ndim") and labels.ndim == 2:
        is_multilabel = True
        n_classes = labels.shape[1]
    elif isinstance(labels, torch.Tensor) and labels.dim() == 2:
        is_multilabel = True
        n_classes = labels.shape[1]
    elif isinstance(labels, torch.Tensor):
        n_classes = int(labels.max().item()) + 1
    else:
        n_classes = int(np.max(labels)) + 1
        
    labels_t = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)
    label_mask_t = label_mask if isinstance(label_mask, torch.Tensor) else torch.tensor(label_mask, dtype=torch.bool, device=device)
    class_signatures = torch.sign(_get_orthogonal_signatures(n_classes, k_label, dev=device))
    S_gt = torch.zeros(n, k_label, dtype=torch.float32, device=device)
    Y_0 = torch.zeros(n, n_classes, dtype=torch.float32, device=device)
    
    if torch.any(label_mask_t):
        if is_multilabel:
            Y_0[label_mask_t] = labels_t[label_mask_t].float()
            S_gt[label_mask_t] = torch.sign(Y_0[label_mask_t] @ class_signatures)
        else:
            import torch.nn.functional as F
            Y_0[label_mask_t] = F.one_hot(labels_t[label_mask_t].long(), num_classes=n_classes).float()
            S_gt[label_mask_t] = class_signatures[labels_t[label_mask_t].long()]
        
    Y_t = Y_0
    for _ in range(3): Y_t = torch.sparse.mm(A_torch, Y_t)
    probs = Y_t / (Y_t.sum(dim=1, keepdim=True) + 1e-10)
    max_probs, pseudo_c = torch.max(probs, dim=1)
    confident = (~label_mask_t) & (max_probs >= 0.85)
    S_v7 = S_gt.clone()
    if torch.any(confident):
        S_v7[confident] = class_signatures[pseudo_c[confident]]
        
    # 3. Diffusion
    def diffuse(W_init):
        W_acc = W_init.clone()
        curr = W_init
        for _ in range(2):
            curr = torch.sparse.mm(A_torch, curr)
            W_acc += curr
        for _ in range(hops):
            W_acc = torch.sparse.mm(A_torch, W_acc)
        return W_acc
    
    # Resolve Stage
    if stage_name == "Abl_Struct_Only":
        emb = R_struct
    elif stage_name == "Abl_Label_Only":
        emb = S_gt
    elif stage_name == "Abl_Struct_Plus_Label":
        emb = torch.cat([R_struct, S_gt], dim=1)
    elif stage_name == "Abl_Struct_Diffused":
        emb = diffuse(R_struct)
    elif stage_name == "Abl_Struct_Label_Diffused":
        emb = diffuse(torch.cat([R_struct, S_gt], dim=1))
    elif stage_name == "Abl_GT_Stream_Only":
        emb = diffuse(torch.cat([torch.zeros_like(R_struct), S_gt], dim=1))
    elif stage_name == "Abl_Pseudo_Stream_Only":
        emb = diffuse(torch.cat([torch.zeros_like(R_struct), S_v7], dim=1))
    elif stage_name == "Abl_GT_Pseudo_Fusion":
        W_svd_acc = diffuse(torch.cat([torch.zeros_like(R_struct), S_gt], dim=1))
        W_v7_acc = diffuse(torch.cat([torch.zeros_like(R_struct), S_v7], dim=1))
        emb = (1.0 - alpha) * W_svd_acc + alpha * W_v7_acc
    elif stage_name == "Abl_Full_Continuous_W_Fused":
        W_svd_acc = diffuse(torch.cat([R_struct, S_gt], dim=1))
        W_v7_acc = diffuse(torch.cat([R_struct, S_v7], dim=1))
        emb = (1.0 - alpha) * W_svd_acc + alpha * W_v7_acc
    else:
        raise ValueError(f"Unknown stage {stage_name}")
        
    if "Continuous" not in stage_name:
        col_medians = torch.median(emb, dim=0).values
        threshold = alpha * col_medians
        emb = (emb > threshold).float()
        
    return emb.cpu().numpy()




def fuse_alg_v1_chebyshev_blend(
    G: nx.Graph,
    pyg_data: Any,
    labels: Any,
    label_mask: Any,
    k: int = 250,
    hops: int = 3,
    alpha: float = 0.5,
    pseudo_threshold: float = 0.85,
    device: str = "cpu",
    seed: int = 42,
    order: int = 4
) -> np.ndarray:
    """Chebyshev Polynomial Structural Signatures + Label Diffusion Blend.
    
    Computes multi-scale Chebyshev polynomial spectral filters on normalized adjacency:
      T_0(A) = Omega
      T_1(A) = A_norm @ Omega
      T_{m+1}(A) = 2 * A_norm @ T_m - T_{m-1}
    Concatenates Chebyshev polynomial orders and orthogonalizes via QR.
    """
    np.random.seed(seed)
    n = G.number_of_nodes()
    k_struct = k // 2

    A_norm, A_torch = _get_A_torch(G, device)

    d_order = max(1, k_struct // order)
    Omega = np.random.randn(n, d_order).astype(np.float32) / np.sqrt(d_order)

    T_list = [Omega]
    if order > 1:
        T_1 = A_norm.dot(Omega)
        T_list.append(T_1)
        T_prev2 = Omega
        T_prev1 = T_1
        for _ in range(2, order):
            T_curr = 2.0 * A_norm.dot(T_prev1) - T_prev2
            T_list.append(T_curr)
            T_prev2 = T_prev1
            T_prev1 = T_curr

    Y = np.concatenate(T_list, axis=1)
    if Y.shape[1] > k_struct:
        Y = Y[:, :k_struct]
    elif Y.shape[1] < k_struct:
        pad = np.zeros((n, k_struct - Y.shape[1]), dtype=np.float32)
        Y = np.concatenate([Y, pad], axis=1)

    Q, _ = np.linalg.qr(Y)
    R_struct = torch.tensor(np.ascontiguousarray(Q[:, :k_struct]), dtype=torch.float32, device=device)

    return get_v1_blend_from_r_struct(
        R_struct, G, pyg_data, labels, label_mask,
        k=k, hops=hops, alpha=alpha, pseudo_threshold=pseudo_threshold,
        device=device, A_torch=A_torch
    )

# Backward-compatible alias
fuse_alg_v1_random_blend = fuse_alg_v1_randomized_blend

# =============================================================================
# CLI Entry Point: Direct Terminal Execution
# =============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='NyS-Binary and Advanced Baselines Runner')
    parser.add_argument('--method', type=str, default='nystrom',
                        choices=['nystrom', 'svd', 'random', 'chebyshev', 'nodesig', 'bigcn', 'node2binary'],
                        help='Hashing algorithm to run (default: nystrom)')
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Benchmark dataset name (e.g. cora, citeseer, pubmed, chameleon, squirrel, actor, wikics)')
    parser.add_argument('--k', type=int, default=250, help='Bitcode dimension (default: 250)')
    parser.add_argument('--hops', type=int, default=3, help='Graph diffusion hops (default: 3)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Convex blend weight (default: 0.5)')
    parser.add_argument('--pseudo_threshold', type=float, default=0.85, help='Pseudo-label confidence threshold (default: 0.85)')
    parser.add_argument('--m_landmarks', type=int, default=500, help='Nystrom landmark count (default: 500)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()

    print('=' * 80)
    print(f'Running {args.method.upper()} on {args.dataset.upper()} (k={args.k}, hops={args.hops}, alpha={args.alpha})')
    print('=' * 80)

    # 1. Load dataset
    try:
        import benchmark_new_algebraic as bna
        G, pyg_data, y, train_mask, test_mask = bna.load_dataset(args.dataset)
    except Exception as e:
        print(f'[Warning] Could not load dataset via benchmark loader ({e}). Falling back to synthetic graph...')
        n = 2708
        G = nx.erdos_renyi_graph(n, 0.005, seed=args.seed)
        pyg_data = None
        y = np.random.randint(0, 7, size=n)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:int(n * 0.1)] = True
        test_mask = ~train_mask

    # 2. Execute selected method
    t0 = time.time()
    if args.method == 'nystrom':
        emb = fuse_alg_v1_nystrom_blend(
            G, pyg_data, y, train_mask, k=args.k, hops=args.hops,
            alpha=args.alpha, pseudo_threshold=args.pseudo_threshold,
            device=args.device, seed=args.seed, m_landmarks=args.m_landmarks
        )
    elif args.method == 'svd':
        emb = fuse_alg_v1_svd_blend(
            G, pyg_data, y, train_mask, k=args.k, hops=args.hops,
            alpha=args.alpha, pseudo_threshold=args.pseudo_threshold,
            device=args.device
        )
    elif args.method == 'random':
        emb = fuse_alg_v1_randomized_blend(
            G, pyg_data, y, train_mask, k=args.k, hops=args.hops,
            alpha=args.alpha, pseudo_threshold=args.pseudo_threshold,
            device=args.device, seed=args.seed
        )
    elif args.method == 'chebyshev':
        emb = fuse_alg_v1_chebyshev_blend(
            G, pyg_data, y, train_mask, k=args.k, hops=args.hops,
            alpha=args.alpha, pseudo_threshold=args.pseudo_threshold,
            device=args.device, seed=args.seed
        )
    elif args.method == 'nodesig':
        emb = generate_official_nodesig_embeddings(G, pyg_data, k=args.k)
    elif args.method == 'bigcn':
        emb = generate_bigcn_embeddings(G, pyg_data, k=args.k, device=args.device)
    elif args.method == 'node2binary':
        emb = generate_node2binary_embeddings(G, pyg_data, k=args.k)
    elapsed = time.time() - t0

    # 3. Print Results
    print(f'\nEmbedding Output Shape : {emb.shape}')
    print(f'Elapsed Runtime        : {elapsed:.4f} seconds')
    unique_vals = np.unique(emb)
    if len(unique_vals) <= 2:
        print(f'Representation Type    : Strictly Binary {unique_vals.tolist()}')
        ones_pct = np.mean(emb == 1.0) * 100
        print(f'Bit Balance            : {ones_pct:.1f}% ones, {100-ones_pct:.1f}% zeros')
    else:
        print(f'Representation Type    : Continuous float (min={emb.min():.3f}, max={emb.max():.3f})')

    # 4. Quick Downstream Linear Probe
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        clf = LogisticRegression(max_iter=200, C=1.0)
        clf.fit(emb[train_mask], y[train_mask])
        acc = accuracy_score(y[test_mask], clf.predict(emb[test_mask]))
        print(f'Downstream Test Acc    : {acc * 100:.2f}%')
    except Exception as e:
        pass
    print('=' * 80)
