"""
graph_sadp.py
=============

A graph message-passing extension of Supervised SADP (see sadp_core.py).

The point of this file is to show the *two* concrete changes needed to turn
the MLP-shaped `SADPNetwork` into a spiking graph network trained by the same
agreement rule -- nothing else about the rule moves:

  (1) FORWARD: the dense input projection `I_h = S_in @ W1` becomes spiking
      message passing. Each node's input current is the neighbourhood-
      aggregated (and then weight-transformed) spikes of its neighbours:
          I_h[v] = ( sum_{u in N(v)} A_hat[v,u] * s_u ) @ W1
      Because the message function is linear we aggregate the (binary) spikes
      first and transform once: `A_hat @ S_in[:,t]) @ W1`. The class read-out
      stays a *local* per-node linear map (no aggregation) -- only the
      message-passing layers use A_hat.

  (2) UPDATE: the SADP outer product is unchanged in form, but the pre-
      synaptic factor must be the quantity W1 actually multiplied, i.e. the
      *aggregated* input `A_hat @ X`, not the raw node features `X`. That one
      substitution is exactly "accumulate the shared-weight outer product over
      every edge that uses the weight": `A_hat @ X` already sums the shared-
      weight contributions over edges per node, and the outer product with
      kappa then sums over nodes. The k-shift, reward modulation, decay/
      renormalisation and the W2 delta rule are all byte-for-byte the same
      logic as sadp_core.py, with the batch axis B reinterpreted as the node
      axis N.

Nice structural consequences (discussed with the design, restated as code):
  * k-shift <-> hop delay: if you let spikes propagate along edges within the
    T window, a k-hop neighbour's influence arrives k steps late, which is
    precisely what compute_shifted_kappa's offset d models.
  * locality  <-> permutation equivariance: per-node kappa + shared weight +
    order-invariant aggregation is equivariant by construction.
  * reward    <-> graph/node-level credit: the per-sample reward becomes a
    per-node reward for node classification (used here) or a per-graph reward
    for graph classification (see the note at the bottom).

This module is deliberately self-contained so the self-test runs with only
numpy. In your tree, delete the "vendored helpers" block and instead do:
    from sadp_core import (poisson_encode_features, compute_shifted_kappa,
                           _compute_reward)
they are identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ===========================================================================
# Vendored helpers -- IDENTICAL to sadp_core.py. Import from there instead.
# ===========================================================================

def poisson_encode_features(batch_feats: np.ndarray, T: int,
                            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """value in [0,1] -> (B, T, Nin) Poisson/Bernoulli spike train."""
    B, Nin = batch_feats.shape
    rnd = (np.random.rand(B, T, Nin) if rng is None
           else rng.random((B, T, Nin))).astype(np.float32)
    return (rnd < batch_feats[:, None, :]).astype(np.float32)


def _cohens_kappa(h_seg: np.ndarray, o_seg: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    agree = np.mean(h_seg == o_seg[:, :, None], axis=1)
    pa = np.mean(h_seg, axis=1)
    pb = np.mean(o_seg, axis=1)[:, None]
    pe = pa * pb + (1.0 - pa) * (1.0 - pb)
    return (agree - pe) / (1.0 - pe + eps)


def _shifted_overlap(spikes_h, target_spikes, d):
    B, T, Nh = spikes_h.shape
    if d == 0:
        return spikes_h, target_spikes
    if d > 0:
        return spikes_h[:, :T - d, :], target_spikes[:, d:]
    dd = -d
    return spikes_h[:, dd:, :], target_spikes[:, :T - dd]


def compute_shifted_kappa(spikes_h, target_spikes, k_shift=0,
                          weight_by_overlap=False, eps=1e-9) -> np.ndarray:
    """Aggregate Cohen's kappa over d in {-k_shift..k_shift}; k_shift=0 == original."""
    B, T, Nh = spikes_h.shape
    K_eff = min(k_shift, T - 1)
    kappas, weights = [], []
    for d in range(-K_eff, K_eff + 1):
        h_seg, o_seg = _shifted_overlap(spikes_h, target_spikes, d)
        if h_seg.shape[1] == 0:
            continue
        kappas.append(_cohens_kappa(h_seg, o_seg, eps))
        weights.append(h_seg.shape[1])
    kappas_arr = np.stack(kappas, axis=0)
    if weight_by_overlap:
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1, 1)
        return np.sum(kappas_arr * w, axis=0) / np.sum(w)
    return np.mean(kappas_arr, axis=0)


def _compute_reward(out_counts, preds, y, T, mode="binary") -> np.ndarray:
    B = out_counts.shape[0]
    if mode == "binary":
        return np.where(preds == y, 1.0, -1.0).astype(np.float32)
    if mode == "margin":
        correct = out_counts[np.arange(B), y]
        masked = out_counts.copy()
        masked[np.arange(B), y] = -np.inf
        best_other = masked.max(axis=1)
        best_other = np.where(np.isfinite(best_other), best_other, 0.0)
        return ((correct - best_other) / max(T, 1)).astype(np.float32)
    raise ValueError(f"Unknown reward_mode '{mode}'.")


# ===========================================================================
# New: adjacency normalisation
# ===========================================================================

def normalize_adjacency(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """GCN-style symmetric normalisation  D^{-1/2} (A + I) D^{-1/2}.

    A: (N, N) {0,1} (or weighted) adjacency. Returns (N, N) float32.
    In production use a sparse matrix; the message passing below is written as
    dense `A_hat @ S` purely for readability of the sketch.
    """
    A = A.astype(np.float32).copy()
    if add_self_loops:
        A[np.diag_indices_from(A)] = 1.0
    deg = A.sum(axis=1)
    dinv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    dinv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    return (dinv_sqrt[:, None] * A * dinv_sqrt[None, :]).astype(np.float32)


# ===========================================================================
# Config + network
# ===========================================================================

@dataclass
class GraphSADPConfig:
    Nin: int
    Nhid: int
    Nout: int
    architecture: str = "1SADP"          # "1SADP" (1 MP layer) or "2SADP" (2)
    T: int = 25
    lam: float = 0.9
    theta_h_base: float = 0.5
    theta_o: float = 0.5
    eta_out: float = 5e-4
    eta_in: float = 2e-4
    decay: float = 0.9995
    norm_eps: float = 1e-6
    clip_w2: float = 5.0
    k_shift: int = 0
    weight_by_overlap: bool = False
    kappa_eps: float = 1e-9
    reward_mode: str = "none"             # "none" | "binary" | "margin"
    reward_scale: float = 1.0
    seed: int = 42

    def __post_init__(self):
        if self.architecture not in ("1SADP", "2SADP"):
            raise ValueError("architecture must be '1SADP' or '2SADP'")
        if self.reward_mode not in ("none", "binary", "margin"):
            raise ValueError("reward_mode must be 'none', 'binary', or 'margin'")


class GraphSADPNetwork:
    """Spiking graph net trained by Supervised SADP (node-classification form)."""

    def __init__(self, cfg: GraphSADPConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        Nin, Nhid, Nout = cfg.Nin, cfg.Nhid, cfg.Nout

        self.W1 = rng.normal(0, 0.1, (Nin, Nhid)).astype(np.float32)     # message: in->hid
        self.W2 = rng.normal(0, 0.1, (Nhid, Nout)).astype(np.float32)    # LOCAL read-out
        self.theta_h = (cfg.theta_h_base + 0.05 * rng.standard_normal(Nhid)).astype(np.float32)

        if cfg.architecture == "2SADP":
            self.W1_2 = rng.normal(0, 0.1, (Nhid, Nhid)).astype(np.float32)  # message: hid->hid
            self.theta_h2 = (cfg.theta_h_base + 0.05 * rng.standard_normal(Nhid)).astype(np.float32)
        else:
            self.W1_2 = None
            self.theta_h2 = None

    # ----------------------------------------------------------------- forward
    def forward(self, X: np.ndarray, A_hat: np.ndarray,
                rng: Optional[np.random.Generator] = None):
        """LIF forward pass over T steps on a graph.

        X     : (N, Nin) node features in [0,1]
        A_hat : (N, N)  normalised adjacency (from normalize_adjacency)
        returns spikes_h (N,T,Nhid), spikes_h2 or None, spikes_o (N,T,Nout)
        """
        cfg = self.cfg
        N = X.shape[0]
        T = cfg.T

        Vh = np.zeros((N, cfg.Nhid), np.float32)
        Vh2 = np.zeros((N, cfg.Nhid), np.float32) if cfg.architecture == "2SADP" else None
        Vo = np.zeros((N, cfg.Nout), np.float32)

        spikes_h = np.zeros((N, T, cfg.Nhid), np.float32)
        spikes_h2 = np.zeros((N, T, cfg.Nhid), np.float32) if cfg.architecture == "2SADP" else None
        spikes_o = np.zeros((N, T, cfg.Nout), np.float32)

        S_in = poisson_encode_features(X, T, rng=rng)   # (N, T, Nin)

        for t in range(T):
            # (1) MESSAGE PASSING: aggregate neighbour spikes, then transform.
            #     A_hat @ (S_in @ W1) == (A_hat @ S_in) @ W1 ; aggregate first
            #     to keep the pre-matmul operand sparse/binary.
            agg_in = A_hat @ S_in[:, t, :]              # (N, Nin)
            I_h = agg_in @ self.W1                      # (N, Nhid)
            Vh = cfg.lam * Vh + I_h
            spk_h = (Vh > self.theta_h).astype(np.float32)
            Vh[spk_h == 1] = 0.0
            spikes_h[:, t, :] = spk_h

            if cfg.architecture == "2SADP":
                agg_h = A_hat @ spk_h                   # (N, Nhid)  second MP round
                I_h2 = agg_h @ self.W1_2
                Vh2 = cfg.lam * Vh2 + I_h2
                spk_h2 = (Vh2 > self.theta_h2).astype(np.float32)
                Vh2[spk_h2 == 1] = 0.0
                spikes_h2[:, t, :] = spk_h2
                spk_to_out = spk_h2
            else:
                spk_to_out = spk_h

            # (1b) READ-OUT is LOCAL: no aggregation, node classifies itself.
            I_o = spk_to_out @ self.W2                  # (N, Nout)
            Vo = cfg.lam * Vo + I_o
            spk_o = (Vo > cfg.theta_o).astype(np.float32)
            Vo[spk_o == 1] = 0.0
            spikes_o[:, t, :] = spk_o

        return spikes_h, spikes_h2, spikes_o

    # ------------------------------------------------------------------ update
    def update(self, X, y, A_hat, spikes_h, spikes_h2, spikes_o,
               train_mask: Optional[np.ndarray] = None):
        """One weight update. Node axis N replaces the batch axis B.

        train_mask: optional bool (N,) selecting labelled nodes to learn from
                    (transductive semi-supervised setting). None == all nodes.
        returns (preds_all, mean_kappa, mean_reward)
        """
        cfg = self.cfg
        N, T, _ = spikes_o.shape
        if train_mask is None:
            train_mask = np.ones(N, dtype=bool)
        n_lab = int(train_mask.sum())

        targets = np.zeros((N, cfg.Nout), np.float32)
        targets[np.arange(N), y] = 1.0

        out_counts = spikes_o.sum(axis=1)               # (N, Nout)
        preds = np.argmax(out_counts, axis=1)

        # ---- W2: supervised delta rule (pre = node's OWN hidden spikes) ----
        errors = targets[:, None, :] - spikes_o         # (N, T, Nout)
        pre_out = spikes_h2 if cfg.architecture == "2SADP" else spikes_h
        dW2 = np.einsum("nti,ntj->ij", pre_out[train_mask], errors[train_mask]) / n_lab
        self.W2 += cfg.eta_out * dW2

        # correct-class output train per node = the node's teacher signal
        target_spikes = spikes_o[np.arange(N), :, y].astype(np.float32)   # (N, T)

        # ---- optional global (here per-node) reward factor ----
        r = None
        mean_reward = None
        if cfg.reward_mode != "none":
            r = _compute_reward(out_counts, preds, y, T, mode=cfg.reward_mode)
            r = (cfg.reward_scale * r).astype(np.float32)
            mean_reward = float(np.mean(r[train_mask]))

        # ---- W1: k-shifted, reward-modulated SADP ----
        kappa1 = compute_shifted_kappa(spikes_h, target_spikes,
                                       k_shift=cfg.k_shift,
                                       weight_by_overlap=cfg.weight_by_overlap,
                                       eps=cfg.kappa_eps)                  # (N, Nhid)
        kappa1_eff = kappa1 if r is None else kappa1 * r[:, None]
        # *** THE KEY LINE ***  pre-synaptic factor is the AGGREGATED input,
        # because that is exactly what W1 multiplied in forward().
        pre_agg1 = A_hat @ X                                              # (N, Nin)
        dW1 = np.einsum("ni,nj->ij", pre_agg1[train_mask], kappa1_eff[train_mask]) / n_lab
        self.W1 += cfg.eta_in * dW1

        # ---- W1_2: second message-passing layer, 2SADP only ----
        if cfg.architecture == "2SADP":
            kappa2 = compute_shifted_kappa(spikes_h2, target_spikes,
                                           k_shift=cfg.k_shift,
                                           weight_by_overlap=cfg.weight_by_overlap,
                                           eps=cfg.kappa_eps)
            kappa2_eff = kappa2 if r is None else kappa2 * r[:, None]
            # pre for W1_2 = aggregated hidden-1 activity (again: what W1_2 saw)
            pre_agg2 = A_hat @ spikes_h.mean(axis=1)                      # (N, Nhid)
            dW1_2 = np.einsum("ni,nj->ij", pre_agg2[train_mask], kappa2_eff[train_mask]) / n_lab
            self.W1_2 += cfg.eta_in * dW1_2
            self.W1_2 *= cfg.decay
            self.W1_2 /= (np.linalg.norm(self.W1_2, axis=0, keepdims=True) + cfg.norm_eps)

        # ---- decay / renormalise (identical policy to sadp_core) ----
        self.W1 *= cfg.decay
        self.W2 *= cfg.decay
        self.W1 /= (np.linalg.norm(self.W1, axis=0, keepdims=True) + cfg.norm_eps)
        np.clip(self.W2, -cfg.clip_w2, cfg.clip_w2, out=self.W2)

        return preds, float(np.mean(kappa1[train_mask])), mean_reward


# ===========================================================================
# Self-test: a homophilous planted-partition graph the rule can actually fit
# ===========================================================================

def _make_planted_graph(n_per_class=30, n_classes=3, Nin=16,
                        p_in=0.25, p_out=0.02, noise=0.35, seed=0):
    rng = np.random.default_rng(seed)
    N = n_per_class * n_classes
    y = np.repeat(np.arange(n_classes), n_per_class)
    protos = rng.random((n_classes, Nin)).astype(np.float32)
    X = np.clip(protos[y] + noise * rng.standard_normal((N, Nin)), 0, 1).astype(np.float32)

    P = np.where(y[:, None] == y[None, :], p_in, p_out)
    A = (rng.random((N, N)) < P).astype(np.float32)
    A = np.triu(A, 1)
    A = A + A.T                                  # symmetric, no self loops yet
    return X, y, A


def _run_selftest():
    rng = np.random.default_rng(0)
    X, y, A = _make_planted_graph(seed=0)
    A_hat = normalize_adjacency(A, add_self_loops=True)
    N = X.shape[0]
    n_classes = len(np.unique(y))

    # transductive split
    perm = rng.permutation(N)
    train_mask = np.zeros(N, bool); train_mask[perm[:N // 2]] = True
    test_mask = ~train_mask

    for arch in ("1SADP", "2SADP"):
        for k_shift, reward in [(0, "none"), (5, "none"), (0, "margin")]:
            cfg = GraphSADPConfig(Nin=X.shape[1], Nhid=64, Nout=n_classes,
                                  architecture=arch, T=25, k_shift=k_shift,
                                  reward_mode=reward, eta_in=1.5e-3, eta_out=3e-3,
                                  seed=1)
            net = GraphSADPNetwork(cfg)
            first_tr = last_tr = last_te = 0.0
            for epoch in range(40):
                sh, sh2, so = net.forward(X, A_hat, rng=rng)
                preds, kap, rew = net.update(X, y, A_hat, sh, sh2, so,
                                             train_mask=train_mask)
                tr = float(np.mean(preds[train_mask] == y[train_mask]))
                te = float(np.mean(preds[test_mask] == y[test_mask]))
                if epoch == 0:
                    first_tr = tr
                last_tr, last_te = tr, te
            print(f"  arch={arch} k={k_shift} reward={reward:6s} | "
                  f"train {first_tr:.2f}->{last_tr:.2f} | test {last_te:.2f} "
                  f"(chance={1/n_classes:.2f})")

    # k_shift=0 sanity: shifted kappa must equal the plain single-lag kappa
    sh, sh2, so = net.forward(X, A_hat, rng=rng)
    tgt = so[np.arange(N), :, y].astype(np.float32)
    k0 = compute_shifted_kappa(sh, tgt, k_shift=0)
    k0_direct = _cohens_kappa(sh, tgt)
    assert np.allclose(k0, k0_direct), "k_shift=0 must reduce to the original kappa"
    print("  PASS: k_shift=0 reduces to the original single-lag kappa")


if __name__ == "__main__":
    print("graph_sadp.py self-test (planted-partition, homophilous graph):")
    _run_selftest()
    print("done.")

# ---------------------------------------------------------------------------
# Graph-level classification (one label per graph) -- the only differences:
#   * pool node hidden spikes into a graph vector before the read-out, e.g.
#         g = spikes_h.mean(axis=0)            # (T, Nhid)  mean-pool over nodes
#     and run ONE output LIF chain per graph;
#   * the teacher train + kappa are then per-GRAPH, and the SADP outer product
#     accumulates over graphs in the batch instead of over nodes (pre factor is
#     still the aggregated input, A_hat @ X, summed per graph);
#   * reward becomes per-graph (its original meaning).
# Everything else -- the rule, k-shift, decay -- is unchanged.
# ---------------------------------------------------------------------------
