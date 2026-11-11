"""
graph_sadp_surrogate.py
=======================

Surrogate-gradient (BPTT) baseline for the spiking graph network in
graph_sadp.py.

Why this file exists
--------------------
graph_sadp.py trains the spiking GCN with the *local* Supervised-SADP
agreement rule (per-node Cohen's kappa vs. the teacher spike train, no
backprop). This file trains the SAME network with the *standard* method the
SNN community would use as a baseline: backpropagation-through-time with a
surrogate derivative substituted for the non-differentiable spike threshold.

The FORWARD PASS here is deliberately identical to GraphSADPNetwork.forward:
  * Poisson-encoded node features,
  * spiking message passing  I_h = (A_hat @ S_in) @ W1  (aggregate then map),
  * a LOCAL per-node spiking read-out  I_o = spk_to_out @ W2,
  * class scores = summed output spike counts.
The ONLY thing that differs between the two files is how the weights are
updated. So an accuracy gap in a head-to-head is attributable to the learning
rule, not to architecture or framework differences.

Surrogate-gradient specifics
----------------------------
Forward uses the hard Heaviside spike H(u), u = V - theta. Backward replaces
dH/du (a Dirac spike, useless for GD) with the smooth pseudo-derivative of a
sigmoid:
        sigma'(u) = beta * s * (1 - s),   s = sigmoid(beta * u).
Reset is reset-by-subtraction (V -> V - theta*spike) with the reset spike
DETACHED from the graph -- the standard, stable choice, so the temporal
adjoint carries only the leak term lam.

Loss is softmax cross-entropy over the output logits, optimised with a small
Adam. Two read-out decoders are provided (SurrogateGraphConfig.readout):
  * "integrator" (default): a non-spiking leaky-integrator output whose summed
    membrane potential is the logit. This is what most surrogate-trained SNN
    classifiers use -- it keeps output gradients from vanishing and trains
    robustly. The spiking message-passing CORE is unchanged; only the final
    decoder is non-spiking.
  * "spiking": summed output spike counts, byte-for-byte identical to
    graph_sadp's read-out. Correct, but brittle to train (the count gradient
    vanishes whenever the output goes silent) -- kept for a strictly-identical
    forward comparison, not as the recommended baseline.

Correctness
-----------
Hand-written BPTT is easy to get subtly wrong, so `_gradient_check()` verifies
the analytic gradients against central finite differences. The check runs the
network in a fully SMOOTH mode (spike = sigmoid(beta*u)) with the reset trace
frozen, so the finite-difference model is exactly the function the analytic
backward differentiates, and in float64 so the differences aren't swamped by
precision noise. If that matches, the BPTT wiring (temporal adjoint, the
A_hat^T term in message passing, the weight gradients) is correct; the only
intended difference in the real model is hard-forward / surrogate-backward,
which is the definition of a surrogate gradient.

To get the plain (non-graph) surrogate-SNN baseline instead, pass
A_hat = np.eye(N): aggregation becomes the identity and this is a standard
BPTT-trained spiking MLP on the node features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _surrogate(u, beta):
    """Pseudo-derivative dspike/du = beta * s * (1 - s), s = sigmoid(beta u)."""
    s = _sigmoid(beta * u)
    return beta * s * (1.0 - s)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def normalize_adjacency(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """GCN-style symmetric  D^{-1/2}(A+I)D^{-1/2}. (Shared with graph_sadp.py.)"""
    A = A.astype(np.float32).copy()
    if add_self_loops:
        A[np.diag_indices_from(A)] = 1.0
    deg = A.sum(axis=1)
    dinv = np.zeros_like(deg)
    nz = deg > 0
    dinv[nz] = 1.0 / np.sqrt(deg[nz])
    return (dinv[:, None] * A * dinv[None, :]).astype(np.float32)


def _poisson_encode(X, T, rng):
    N, Nin = X.shape
    rnd = rng.random((N, T, Nin))
    return (rnd < X[:, None, :]).astype(X.dtype)


# ---------------------------------------------------------------------------
# config + network
# ---------------------------------------------------------------------------

@dataclass
class SurrogateGraphConfig:
    Nin: int
    Nhid: int
    Nout: int
    architecture: str = "1SADP"        # "1SADP" (1 MP layer) | "2SADP" (2)
    T: int = 25
    lam: float = 0.9
    theta_h_base: float = 0.5
    theta_o: float = 0.5
    readout: str = "integrator"   # "integrator" (non-spiking LI, robust) | "spiking" (count; identical to graph_sadp)
    beta: float = 5.0                  # surrogate sharpness
    lr: float = 2e-2
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    seed: int = 42

    def __post_init__(self):
        if self.architecture not in ("1SADP", "2SADP"):
            raise ValueError("architecture must be '1SADP' or '2SADP'")


class SurrogateGraphSNN:
    """Spiking graph net trained by surrogate-gradient BPTT.

    Weight shapes and initialisation MATCH GraphSADPNetwork exactly so the two
    can be compared like-for-like.
    """

    def __init__(self, cfg: SurrogateGraphConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        Nin, Nhid, Nout = cfg.Nin, cfg.Nhid, cfg.Nout

        self.W1 = rng.normal(0, 0.1, (Nin, Nhid)).astype(np.float32)
        self.W2 = rng.normal(0, 0.1, (Nhid, Nout)).astype(np.float32)
        self.theta_h = (cfg.theta_h_base + 0.05 * rng.standard_normal(Nhid)).astype(np.float32)

        if cfg.architecture == "2SADP":
            self.W1_2 = rng.normal(0, 0.1, (Nhid, Nhid)).astype(np.float32)
            self.theta_h2 = (cfg.theta_h_base + 0.05 * rng.standard_normal(Nhid)).astype(np.float32)
        else:
            self.W1_2 = None
            self.theta_h2 = None

        self._A_hat: Optional[np.ndarray] = None
        self._init_adam()

    def _param_names(self):
        return ["W1", "W2"] + (["W1_2"] if self.cfg.architecture == "2SADP" else [])

    def _init_adam(self):
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._param_names()}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._param_names()}
        self._t = 0

    def cast(self, dtype):
        """Cast all parameters (and reset Adam state) to `dtype`. Used by the
        float64 gradient check; training runs happily in the default float32."""
        for k in ("W1", "W2", "W1_2", "theta_h", "theta_h2"):
            a = getattr(self, k)
            if a is not None:
                setattr(self, k, a.astype(dtype))
        self._init_adam()
        return self

    # ----------------------------------------------------------------- forward
    def forward(self, X, A_hat, rng=None, S_in=None,
                smooth=False, reset_traces=None):
        """Forward pass. Returns (logits, cache).

        smooth=False : hard Heaviside spikes (training / inference).
        smooth=True  : spike = sigmoid(beta*u)  (for the gradient check only).
        reset_traces : optional dict of FIXED spike traces used for the reset
                       subtraction (materialises the 'detached reset' so the
                       finite-difference check is exact). None => live spikes.
        S_in         : optional precomputed (N,T,Nin) Poisson sample; pass it
                       to keep forward deterministic across FD perturbations.
        """
        cfg = self.cfg
        dt = self.W1.dtype
        N = X.shape[0]
        T = cfg.T
        beta = cfg.beta

        import scipy.sparse as sp
        if not sp.issparse(A_hat):
            A_hat = np.asarray(A_hat, dtype=dt)
        self._A_hat = A_hat

        if S_in is None:
            if rng is None:
                rng = np.random.default_rng()
            S_in = _poisson_encode(X.astype(dt), T, rng)
        else:
            S_in = S_in.astype(dt)

        theta_h = self.theta_h
        theta_o = cfg.theta_o

        def spike(u):
            return _sigmoid(beta * u) if smooth else (u > 0).astype(dt)

        agg_in = A_hat.dot(S_in.reshape(N, -1)).reshape(N, T, cfg.Nin)    # (N, T, Nin)

        two = cfg.architecture == "2SADP"

        u_h = np.zeros((N, T, cfg.Nhid), dt)
        spk_h = np.zeros((N, T, cfg.Nhid), dt)
        agg_h = np.zeros((N, T, cfg.Nhid), dt) if two else None
        u_h2 = np.zeros((N, T, cfg.Nhid), dt) if two else None
        spk_h2 = np.zeros((N, T, cfg.Nhid), dt) if two else None
        u_o = np.zeros((N, T, cfg.Nout), dt)
        spk_o = np.zeros((N, T, cfg.Nout), dt)

        Vh = np.zeros((N, cfg.Nhid), dt)
        Vh2 = np.zeros((N, cfg.Nhid), dt) if two else None
        Vo = np.zeros((N, cfg.Nout), dt)

        rh = reset_traces["h"] if reset_traces else None
        rh2 = reset_traces["h2"] if (reset_traces and two) else None
        ro = reset_traces["o"] if reset_traces else None

        for t in range(T):
            # layer 1 (message passing)
            I_h = agg_in[:, t, :] @ self.W1
            reset_prev = (rh[:, t - 1] if rh is not None else spk_h[:, t - 1]) if t > 0 else 0.0
            Vh = cfg.lam * (Vh - theta_h * reset_prev) + I_h
            uh = Vh - theta_h
            sh = spike(uh)
            u_h[:, t, :] = uh
            spk_h[:, t, :] = sh

            if two:
                agg_h[:, t, :] = A_hat @ sh
                I_h2 = agg_h[:, t, :] @ self.W1_2
                reset_prev2 = (rh2[:, t - 1] if rh2 is not None else spk_h2[:, t - 1]) if t > 0 else 0.0
                Vh2 = cfg.lam * (Vh2 - self.theta_h2 * reset_prev2) + I_h2
                uh2 = Vh2 - self.theta_h2
                sh2 = spike(uh2)
                u_h2[:, t, :] = uh2
                spk_h2[:, t, :] = sh2
                spk_to_out_t = sh2
            else:
                spk_to_out_t = sh

            # read-out (local, no aggregation)
            I_o = spk_to_out_t @ self.W2
            if cfg.readout == "spiking":
                reset_prevo = (ro[:, t - 1] if ro is not None else spk_o[:, t - 1]) if t > 0 else 0.0
                Vo = cfg.lam * (Vo - theta_o * reset_prevo) + I_o
                uo = Vo - theta_o
                u_o[:, t, :] = uo
                spk_o[:, t, :] = spike(uo)
            else:  # "integrator": non-spiking leaky-integrator read-out
                Vo = cfg.lam * Vo + I_o
                spk_o[:, t, :] = Vo   # store membrane trace in spk_o slot

        logits = spk_o.sum(axis=1)                         # (N, Nout) counts or summed-membrane
        cache = dict(agg_in=agg_in, u_h=u_h, spk_h=spk_h,
                     agg_h=agg_h, u_h2=u_h2, spk_h2=spk_h2,
                     u_o=u_o, spk_o=spk_o,
                     spk_to_out=(spk_h2 if two else spk_h))
        return logits, cache

    # ------------------------------------------------------------- loss / grads
    def loss(self, logits, y, train_mask):
        p = _softmax(logits)
        n = int(train_mask.sum())
        ll = -np.log(p[np.arange(len(y)), y] + 1e-12)
        return float(ll[train_mask].sum() / n)

    def backward(self, cache, logits, y, train_mask):
        """Analytic BPTT gradients. Returns (grads dict, loss)."""
        cfg = self.cfg
        N = cache["spk_o"].shape[0]
        T = cfg.T
        lam, beta = cfg.lam, cfg.beta
        two = cfg.architecture == "2SADP"
        n_lab = int(train_mask.sum())

        # dL/dlogits from softmax-CE, masked to labelled nodes
        p = _softmax(logits)
        onehot = np.zeros_like(p)
        onehot[np.arange(len(y)), y] = 1.0
        dlogits = (p - onehot)
        dlogits[~train_mask] = 0.0
        dlogits /= n_lab                                   # (N, Nout)

        # output adjoint (branch on read-out type)
        lam_o = np.zeros((N, cfg.Nout), dtype=logits.dtype)
        dW2 = np.zeros_like(self.W2)
        g_to_out = np.zeros((N, T, cfg.Nhid), dtype=logits.dtype)
        spiking_out = (cfg.readout == "spiking")
        spo = _surrogate(cache["u_o"], beta) if spiking_out else None
        for t in range(T - 1, -1, -1):
            # logits = sum_t Vo[t]; for spiking read-out an extra surrogate factor
            src_t = dlogits * spo[:, t, :] if spiking_out else dlogits
            lam_o = src_t + lam * lam_o                     # dL/dVo[t]
            dW2 += cache["spk_to_out"][:, t, :].T @ lam_o   # dL/dI_o[t]=lam_o
            g_to_out[:, t, :] = lam_o @ self.W2.T

        grads = {"W2": dW2}

        if two:
            sph2 = _surrogate(cache["u_h2"], beta)
            lam_h2 = np.zeros((N, cfg.Nhid), dtype=logits.dtype)
            dW1_2 = np.zeros_like(self.W1_2)
            g_spk_h = np.zeros((N, T, cfg.Nhid), dtype=logits.dtype)
            for t in range(T - 1, -1, -1):
                lam_h2 = g_to_out[:, t, :] * sph2[:, t, :] + lam * lam_h2
                dW1_2 += cache["agg_h"][:, t, :].T @ lam_h2
                g_agg_h = lam_h2 @ self.W1_2.T
                g_spk_h[:, t, :] = self._A_hat.T @ g_agg_h  # message-passing transpose
            grads["W1_2"] = dW1_2
            g_hidden = g_spk_h
        else:
            g_hidden = g_to_out

        sph = _surrogate(cache["u_h"], beta)
        lam_h = np.zeros((N, cfg.Nhid), dtype=logits.dtype)
        dW1 = np.zeros_like(self.W1)
        for t in range(T - 1, -1, -1):
            lam_h = g_hidden[:, t, :] * sph[:, t, :] + lam * lam_h
            dW1 += cache["agg_in"][:, t, :].T @ lam_h
        grads["W1"] = dW1

        return grads, self.loss(logits, y, train_mask)

    # --------------------------------------------------------------- optimiser
    def _adam_step(self, grads):
        cfg = self.cfg
        self._t += 1
        b1, b2, eps = cfg.adam_b1, cfg.adam_b2, cfg.adam_eps
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * (g * g)
            mhat = self._m[k] / (1 - b1 ** self._t)
            vhat = self._v[k] / (1 - b2 ** self._t)
            setattr(self, k, getattr(self, k) - cfg.lr * mhat / (np.sqrt(vhat) + eps))

    def train_step(self, X, A_hat, y, train_mask, rng):
        logits, cache = self.forward(X, A_hat, rng=rng, smooth=False)
        grads, loss = self.backward(cache, logits, y, train_mask)
        self._adam_step(grads)
        return loss, np.argmax(logits, axis=1)

    def predict(self, X, A_hat, rng):
        logits, _ = self.forward(X, A_hat, rng=rng, smooth=False)
        return np.argmax(logits, axis=1)


# ---------------------------------------------------------------------------
# gradient check + self-test
# ---------------------------------------------------------------------------

def _make_planted_graph(n_per_class=30, n_classes=3, Nin=16,
                        p_in=0.25, p_out=0.02, noise=0.35, seed=0):
    """Identical generator to graph_sadp.py so the two are comparable."""
    rng = np.random.default_rng(seed)
    N = n_per_class * n_classes
    y = np.repeat(np.arange(n_classes), n_per_class)
    protos = rng.random((n_classes, Nin)).astype(np.float32)
    X = np.clip(protos[y] + noise * rng.standard_normal((N, Nin)), 0, 1).astype(np.float32)
    P = np.where(y[:, None] == y[None, :], p_in, p_out)
    A = (rng.random((N, N)) < P).astype(np.float32)
    A = np.triu(A, 1)
    A = A + A.T
    return X, y, A


def _gradient_check(arch="2SADP", readout="integrator", eps=1e-5, n_probe=12, tol=1e-4):
    """Finite-difference check of the analytic BPTT (fully-smooth, float64)."""
    rng = np.random.default_rng(3)
    X, y, A = _make_planted_graph(n_per_class=6, n_classes=3, Nin=8, seed=7)
    A_hat = normalize_adjacency(A).astype(np.float64)
    X = X.astype(np.float64)
    N = X.shape[0]
    train_mask = np.ones(N, bool)

    cfg = SurrogateGraphConfig(Nin=X.shape[1], Nhid=10, Nout=3,
                               architecture=arch, T=8, beta=4.0, readout=readout, seed=5)
    net = SurrogateGraphSNN(cfg).cast(np.float64)

    S_in = _poisson_encode(X, cfg.T, rng)                  # fix the Poisson draw (float64)

    _, base = net.forward(X, A_hat, S_in=S_in, smooth=True)
    reset_traces = {"h": base["spk_h"], "o": base["spk_o"]}
    if arch == "2SADP":
        reset_traces["h2"] = base["spk_h2"]

    logits, cache = net.forward(X, A_hat, S_in=S_in, smooth=True, reset_traces=reset_traces)
    grads, _ = net.backward(cache, logits, y, train_mask)

    max_rel = 0.0
    for k in net._param_names():
        W = getattr(net, k)
        for _ in range(n_probe):
            i = int(rng.integers(W.shape[0])); j = int(rng.integers(W.shape[1]))
            orig = W[i, j]
            W[i, j] = orig + eps
            lp, _ = net.forward(X, A_hat, S_in=S_in, smooth=True, reset_traces=reset_traces)
            Lp = net.loss(lp, y, train_mask)
            W[i, j] = orig - eps
            lm, _ = net.forward(X, A_hat, S_in=S_in, smooth=True, reset_traces=reset_traces)
            Lm = net.loss(lm, y, train_mask)
            W[i, j] = orig
            fd = (Lp - Lm) / (2 * eps)
            an = grads[k][i, j]
            rel = abs(fd - an) / (max(abs(fd), abs(an)) + 1e-12)
            max_rel = max(max_rel, rel)
    ok = max_rel < tol
    print(f"  grad-check ({arch}, {readout:10s}): max relative error = {max_rel:.2e}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    assert ok, f"gradient check failed for {arch}/{readout}: {max_rel:.2e}"


def _train_demo():
    rng = np.random.default_rng(0)
    X, y, A = _make_planted_graph(seed=0)
    A_hat = normalize_adjacency(A)
    N = X.shape[0]
    n_classes = len(np.unique(y))
    perm = rng.permutation(N)
    train_mask = np.zeros(N, bool); train_mask[perm[:N // 2]] = True
    test_mask = ~train_mask

    for arch in ("1SADP", "2SADP"):
        cfg = SurrogateGraphConfig(Nin=X.shape[1], Nhid=64, Nout=n_classes,
                                   architecture=arch, T=25, beta=5.0, lr=1e-2,
                                   readout="integrator", seed=1)
        net = SurrogateGraphSNN(cfg)
        first = last_tr = 0.0
        loss = 0.0
        for epoch in range(100):
            loss, preds = net.train_step(X, A_hat, y, train_mask, rng)
            tr = float(np.mean(preds[train_mask] == y[train_mask]))
            if epoch == 0:
                first = tr
            last_tr = tr
        te_preds = net.predict(X, A_hat, rng)
        last_te = float(np.mean(te_preds[test_mask] == y[test_mask]))
        print(f"  arch={arch} | train {first:.2f}->{last_tr:.2f} | "
              f"test {last_te:.2f} | final loss {loss:.3f} (chance={1/n_classes:.2f})")


if __name__ == "__main__":
    print("graph_sadp_surrogate.py self-test:")
    for arch in ("1SADP", "2SADP"):
        for readout in ("integrator", "spiking"):
            _gradient_check(arch, readout)
    print("surrogate-gradient training demo (same planted graph as graph_sadp.py):")
    _train_demo()
    print("done.")
