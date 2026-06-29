"""Utilities for applying IA3 (l_k, l_v, l_ff) parameter-efficient updates.

IA3 scaling vectors are initialized at 1 (the identity transform). Throughout
training we keep a running set of delta updates and materialize the actual
parameter as ``1 + delta``. These helpers centralize that logic, which was
previously duplicated (3 near-identical if/elif blocks, repeated in several
places) across main.py, evaluate.py, and lib/train.py.
"""
from __future__ import annotations

import torch

# Substrings identifying IA3 scaling parameters inside a timm ViT model.
IA3_SUBSTRINGS = ("l_ff", "l_k", "l_v")


def is_ia3_param(name: str) -> bool:
    """Return True if a parameter name belongs to an IA3 scaling vector."""
    return any(tag in name for tag in IA3_SUBSTRINGS)


def collect_ia3_params(module: torch.nn.Module) -> dict:
    """Collect all IA3 scaling parameters of ``module`` into a name -> param dict."""
    return {n: p for n, p in module.named_parameters() if is_ia3_param(n)}


@torch.no_grad()
def apply_ia3_updates(model: torch.nn.Module, updates: dict, device, prefix: str = "model."):
    """Write ``1 + updates[prefix + name]`` into every IA3 parameter of ``model``.

    ``updates`` is keyed by the IA3 parameter name as seen on the wrapping
    ``Net`` module (i.e. prefixed with ``prefix``), matching how Jacobians are
    produced via ``torch.func.functional_call`` on ``Net`` in
    ``lib.train.tangent_tuning_a_batch``.
    """
    for name, param in model.named_parameters():
        if not is_ia3_param(name):
            continue
        delta = updates[prefix + name].reshape(param.shape).to(device)
        param.copy_(torch.ones_like(param, device=device) + delta)


@torch.no_grad()
def reset_ia3_to_identity(model: torch.nn.Module, device):
    """Reset every IA3 parameter of ``model`` back to its identity value (1)."""
    for name, param in model.named_parameters():
        if is_ia3_param(name):
            param.copy_(torch.ones_like(param, device=device))