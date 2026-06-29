import math
import sys

import torch

from lib.utils import *
from utils.vision import *
from lib.ia3 import apply_ia3_updates
from timm.utils import accuracy


def tangent_tuning_a_batch(
    input, target, global_At_A, global_AtA_w, AtA_ridge,
    net_g, jacobian_fn, model, tuning_params, backbone,
    max_epochs=200, patience=3, eta=0.03, lambda_=5e-6,
):
    device = cfg.device
    input = input.to(device)
    target = target.to(device)
    batch_size = target.size(0)

    # 1) Base (pretrained) target-class softmax probabilities.
    with torch.no_grad():
        logits = net_g(input, target).detach()
    m = batch_size

    # 2) Jacobian of the target-class probability w.r.t. the IA3 parameters.
    J = jacobian_fn(tuning_params, input, target)
    for k in list(J.keys()):
        J[k] = J[k].detach()

    # 3) Per-parameter-block design matrix and (ridge-regularized) normal
    # equations, reused across the soft-target refinement iterations below.
    A_cache = {}
    for name, param in tuning_params.items():
        A = J[name].reshape(m, -1)
        d = A.shape[1]
        ATA = A.T @ A
        AtA_ridge[name] = ATA + lambda_ * torch.eye(d, device=ATA.device, dtype=ATA.dtype)
        A_cache[name] = {'A': A, 'ATA': ATA, 'd': d}

    # 4) Fixed-point soft-target refinement: iteratively move the current
    # prediction `z` toward the squared-loss target `b = 2 - p` and track
    # the update that yields the best batch accuracy (early-stopped after
    # `patience` non-improving iterations).
    z = logits
    b = (2 - z).to(device)
    previous_acc = 0.0
    num_stop = 0
    early_stop = False
    epoch = 0
    best_updates = {k: torch.zeros(A_cache[k]['d'], device=device) for k in tuning_params.keys()}

    while not early_stop and epoch < max_epochs:
        z = (1 - eta) * z + eta * b
        vec = z.flatten()

        w_updates = {}
        for name in tuning_params.keys():
            A = A_cache[name]['A']
            ATB = A.T @ vec
            w_updates[name] = torch.linalg.solve(AtA_ridge[name], ATB)

        apply_ia3_updates(model, w_updates, device)

        with torch.no_grad():
            out = model(input)["logits"]
            acc1, _ = accuracy(out, target, topk=(1, 5))

        if acc1 > previous_acc:
            previous_acc = acc1
            num_stop = 0
            for k in w_updates:
                best_updates[k] = w_updates[k].clone()
        else:
            num_stop += 1
            if num_stop >= patience:
                early_stop = True
        epoch += 1

    # 5) Fold this batch's statistics into the running global accumulators
    # and resolve the global closed-form update.
    global_updates = {k: torch.zeros(A_cache[k]['d'], device=device) for k in tuning_params.keys()}
    for name in tuning_params.keys():
        ATA = A_cache[name]['ATA']
        global_At_A[name] += ATA
        global_AtA_w[name] += ATA @ best_updates[name]
        I = torch.eye(global_At_A[name].shape[1], device=global_At_A[name].device, dtype=global_At_A[name].dtype)
        global_updates[name] = torch.linalg.solve(global_At_A[name] + lambda_ * I, global_AtA_w[name])

    del J, A_cache
    torch.cuda.empty_cache()
    return global_updates


def train_one_batch(model: torch.nn.Module, criterion, input: torch.Tensor, target: torch.Tensor,
                     optimizer: torch.optim.Optimizer, device: torch.device, max_norm: float = 0,
                     set_training_mode=True, task_id=-1, class_mask=None):
    """Gradient-based warm start of the classification head on one batch.

    `task_id` is accepted for API/logging symmetry with the rest of the
    pipeline but does not alter training behavior here: task identity is
    unknown in the task-free setting, so no task-conditional logic is
    applied.
    """
    model.train(set_training_mode)
    for _ in range(cfg.dtask.epochs):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', SmoothedValue(window_size=1, fmt='{value:.4f}'))

        logits = model(input)['logits']

        # Mask out classes not present in this batch. We cannot mask by task
        # identity (task-free setting), only by the classes actually
        # observed in the current streaming batch.
        if class_mask is not None:
            not_mask = np.setdiff1d(np.arange(cfg.dtask.nb_classes), class_mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target)
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        torch.cuda.synchronize()

        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    metric_logger.synchronize_between_processes()
    log('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                losses=metric_logger.meters['Loss']))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}