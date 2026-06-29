"""Helpers for saving and loading per-distribution ("task") signatures.

Each detected distribution stores: the ridge accumulators (``global_At_A``,
``global_AtA_w``), the closed-form IA3 update derived from them, and the
Gaussian-mixture task key (means/vars/counts). These were previously saved
and re-loaded with near-identical, copy-pasted code in main.py and
evaluate.py.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


def task_checkpoint_dir(output_dir: str, task_id: int) -> str:
    """Path to the checkpoint directory for a given detected distribution."""
    return os.path.join(output_dir, f"ia3_task_{task_id}")


def save_task_checkpoint(output_dir, task_id, global_At_A, global_AtA_w, online_gauss,
                          global_updates=None, head_weight=None, head_bias=None,
                          mahal_history=None):
    """Persist the ridge accumulators and the GMM task key for ``task_id``."""
    ckpt_dir = task_checkpoint_dir(output_dir, task_id)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    torch.save(global_At_A, os.path.join(ckpt_dir, "global_At_A.pth"))
    torch.save(global_AtA_w, os.path.join(ckpt_dir, "global_AtA_w.pth"))
    torch.save(online_gauss.means, os.path.join(ckpt_dir, "mean.pth"))
    torch.save(online_gauss.vars, os.path.join(ckpt_dir, "vars.pth"))
    torch.save(online_gauss.counts, os.path.join(ckpt_dir, "counts.pth"))

    if global_updates is not None:
        torch.save(global_updates, os.path.join(ckpt_dir, "global_updates.pth"))
    if head_weight is not None:
        torch.save(head_weight, os.path.join(ckpt_dir, "head_weight.pth"))
    if head_bias is not None:
        torch.save(head_bias, os.path.join(ckpt_dir, "head_bias.pth"))
    if mahal_history is not None:
        torch.save(mahal_history, os.path.join(ckpt_dir, "mahal_history.pth"))

    return ckpt_dir


def mask_significant_clusters(mean, var, counts, min_count: int = 10):
    """Drop GMM clusters that have fewer than ``min_count`` assigned points.

    Mirrors the original (duplicated) masking logic used everywhere the task
    key was consumed for evaluation/retrieval.
    """
    keep = counts >= min_count
    return mean[keep], var[keep]


def load_task_signature(output_dir, task_id, min_cluster_count: int = 10):
    """Load the masked GMM task key and IA3 ridge solution for ``task_id``."""
    ckpt_dir = task_checkpoint_dir(output_dir, task_id)
    means = torch.load(os.path.join(ckpt_dir, "mean.pth"))
    variances = torch.load(os.path.join(ckpt_dir, "vars.pth"))
    counts = torch.load(os.path.join(ckpt_dir, "counts.pth"))
    mean, var = mask_significant_clusters(means, variances, counts, min_cluster_count)
    global_updates = torch.load(os.path.join(ckpt_dir, "global_updates.pth"))
    return {"mean": mean, "var": var, "global_updates": global_updates}