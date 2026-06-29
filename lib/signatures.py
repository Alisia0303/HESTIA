import torch
import numpy as np
import math

from sklearn.cluster import KMeans
from lib.utils import *

class OnlineDiagonalGMM:
    """A diagonal-covariance Gaussian mixture, fit and updated online.
 
    Each adapter in the bank is paired with one instance of this class,
    acting as its "task key" (Section 3.3): a compact density model over
    the adapter-induced embedding distribution, used both for likelihood-
    based routing at inference and for streaming change-point detection
    during training.
    """
    
    def __init__(
        self,
        n_clusters,
        feature_dim,
        device="cpu",
        eps=1e-4,
        min_cluster_weight=0.02,   # Dirichlet prior strength
        split_threshold=0.2,       # Std-dev threshold for split
        cfg=None
    ):
        self.K = self.n_clusters = n_clusters
        self.D = feature_dim
        self.device = device
        self.eps = eps
        self.cfg = cfg

        self.min_cluster_weight = min_cluster_weight
        self.split_threshold = split_threshold

        # Stats
        self.counts = torch.zeros(self.K, device=device)
        self.means = torch.zeros(self.K, self.D, device=device)
        self.vars = torch.ones(self.K, self.D, device=device) * eps

        self.is_initialized = False

    # ---------------------------------------------------------
    # Initialization (sklearn only ONCE)
    # ---------------------------------------------------------
    @torch.no_grad()
    def initialize(self, features):
        feats_np = features.cpu().numpy()
        km = KMeans(n_clusters=self.K, n_init=20)
        labels = km.fit_predict(feats_np)

        self.means = torch.tensor(km.cluster_centers_, device=self.device)
        for k in range(self.K):
            mask = labels == k
            if mask.sum() > 1:
                diff = features[mask] - self.means[k]
                self.vars[k] = diff.var(dim=0) + self.eps
                self.counts[k] = mask.sum()
            else:
                self.vars[k].fill_(1.0)
                self.counts[k] = 1.0

        self.is_initialized = True

    # ---------------------------------------------------------
    # Scoring (Mahalanobis + Dirichlet prior, balanced / anti-collapse)
    # ---------------------------------------------------------

    @torch.no_grad()
    def _cluster_scores(self, features):
        """Per-sample, per-cluster score: diagonal Mahalanobis distance plus
        a Dirichlet-style prior term that discourages collapsing onto a
        single dominant cluster. Shared by `assign` and `score` (previously
        duplicated in both this file and in main.py).
        """
        diff = features[:, None, :] - self.means[None, :, :]
        mahal = (diff ** 2 / self.vars[None, :, :]).sum(dim=2)
        weights = self.counts / (self.counts.sum() + 1e-6)
        prior = -torch.log(weights + self.min_cluster_weight)
        return mahal + prior
 
    @torch.no_grad()
    def assign(self, features):
        """Hard-assign each sample to its closest (lowest-score) cluster."""
        return torch.argmin(self._cluster_scores(features), dim=1)
 
    @torch.no_grad()
    def score(self, features):
        """Per-sample routing / change-point score: the minimum cluster score
        (Eq. for S_t(z) in the paper). Lower means the sample is better
        explained by this task key.
        """
        return torch.min(self._cluster_scores(features), dim=1).values

    # ---------------------------------------------------------
    # Online Update
    # ---------------------------------------------------------
    @torch.no_grad()
    def update(self, features, batch_id):
        """Online (Welford-style) update of cluster means/variances/counts."""
        if not self.is_initialized:
            self.initialize(features)
            return
 
        labels = self.assign(features)
        for k in range(self.K):
            mask = labels == k
            n_new = mask.sum().item()
            if n_new < 2:
                continue
 
            x = features[mask]
            mu_old, var_old, n_old = self.means[k], self.vars[k], self.counts[k]
            mu_batch = x.mean(dim=0)
            var_batch = x.var(dim=0, unbiased=False) + self.eps
            n_total = n_old + n_new
            delta = mu_batch - mu_old
 
            mu_new = mu_old + delta * (n_new / n_total)
            var_new = (
                n_old * var_old + n_new * var_batch + (n_old * n_new / n_total) * delta ** 2
            ) / n_total
 
            self.means[k] = mu_new
            self.vars[k] = torch.clamp(var_new, min=self.eps)
            self.counts[k] = n_total
 
        if batch_id > self.cfg.signature.split_warmup:
            self.split_clusters()

    # ---------------------------------------------------------
    # Split Logic
    # ---------------------------------------------------------
    @torch.no_grad()
    def split_clusters(self):
        """Split clusters whose spread exceeds `split_threshold`, donating
        points from the weakest (lowest-count) cluster to make room.
        """
        for k in range(self.K):
            if self.counts[k] < self.cfg.signature.split_minpoints:
                continue
 
            cluster_std = torch.sqrt(self.vars[k].mean())
            if cluster_std > self.split_threshold:
                dead = torch.argmin(self.counts).item()
                offset = torch.randn(self.D, device=self.device) * cluster_std
                self.means[dead] = self.means[k] + offset
                self.vars[dead] = self.vars[k].clone()
                self.counts[dead] = math.floor(self.counts[k] * 0.5)
                self.counts[k] = self.counts[k] - self.counts[dead]
                log(f"SPLIT happened for cluster {k}, donating points to cluster {dead}. "
                    f"Now cluster {k} has {self.counts[k]} points and cluster {dead} has {self.counts[dead]}")
 
    def reset(self):
        """Reset all statistics (called when a new distribution/adapter is instantiated)."""
        self.is_initialized = False
        self.counts.zero_()
        self.means.zero_()
        self.vars = torch.ones(self.K, self.D, device=self.device) * self.eps

def detect_convergence_stationary_streaming(
    values: torch.Tensor,
    window: int = 6,
    k: float = 2.0,
    min_std: float = 1e-6,
    persistence: int = 1,
    cool_down: int = 50,
):
    """
    Detect convergence via stationarity of sliding windows.

    Args:
        values: Tensor of shape [T]
        window: window size W
        k: std multiplier
        min_std: numerical stability
        persistence: number of consecutive windows required

    Returns:
        converged (bool)
        start_idx (int or None)
    """
    x = values.float()
    T = x.shape[0]

    stable_count = 0

    for t in range(window + cool_down, T - window + 1):
        ref = x[t - window:t]
        mu = ref.mean()
        std = ref.std(unbiased=False).clamp(min=min_std)

        next_window = x[t:t + window]
        within = torch.abs(next_window - mu) <= k * std

        if torch.all(within):
            stable_count += 1
            if stable_count >= persistence:
                return True, t
        else:
            stable_count = 0

    return False, None