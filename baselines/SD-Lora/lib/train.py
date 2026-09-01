from lib.utils import *
from utils.vision import *
from timm.utils import accuracy
import copy
import torch.nn.functional as F

def tangent_tuning_a_batch(
    idx, input, target, global_At_A, global_AtA_w, AtA_ridge, text_tokens,
    net_g, jacobian_fn, clip_model, tuning_params, backbone,
    max_epochs=200, patience=1, eta=0.03, lambda_=5e-6, 
):
    
    device = cfg.device
    # move inputs
    input = input.to(device)
    target = target.to(device)

    batch_size = target.size(0)
    labels = F.one_hot(target, num_classes=cfg.dtask.nb_classes).float().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # 1) compute base logits (no grad)
    with torch.no_grad():
        logits = net_g(input).detach()   # keep on CPU to reduce GPU mem if desired
    num_classes = logits.size(1)
    m = batch_size * num_classes

    # 2) compute jacobian once and detach (assumes jacobian_fn returns dict of tensors)
    # J = jacobian_fn(tuning_params, input)           # expected dict: name -> (batch, num_classes, ...param_shape...)
    block_size = 10
    J_blocks = []
    for start in range(0, num_classes, block_size):
        class_idx = list(range(start, min(start + block_size, num_classes)))
        J_block = jacobian_fn(tuning_params, input, class_idx)  # dict: name -> (batch, len(class_idx), ...param_shape...)
        J_block = {k: v.detach() for k, v in J_block.items()}
        J_blocks.append(J_block)
    # combine blocks
    J = {}
    for k in J_blocks[0].keys():
        J[k] = torch.cat([block[k] for block in J_blocks], dim=1)  # concat on class dim   
    
    
    
    
    # detach and move to device (keep on same device as later computation)
    for k in list(J.keys()):
        J[k] = J[k].detach()

    # 3) Precompute A matrices, AtA, AtA_ridge and store flattened param shapes
    A_cache = {}
    for name, param in tuning_params.items():
        # J[name] shape assumed: (batch, num_classes, param_shape...) -> flatten per row
        A = J[name].reshape(m, -1)               # (m, d)
        d = A.shape[1]
        ATA = A.T @ A                            # (d, d) - cheap: m is small (160)
        # add ridge (store for later)
        AtA_ridge[name] = ATA + lambda_ * torch.eye(d, device=ATA.device, dtype=ATA.dtype)
        A_cache[name] = {'A': A, 'ATA': ATA, 'd': d}

    # 4) initialize variables for tuning loop
    z = logits           # move logits to device for softmax ops
    b = (labels - z).to(device)      # difference
    previous_acc = 0.0
    num_stop = 0
    early_stop = False
    epoch = 0
    best_updates = {k: torch.zeros(A_cache[k]['d'], device=device) for k in tuning_params.keys()}

    epoch = 0
    num_stop = 0
    early_stop = False
    while not early_stop and epoch < max_epochs:
        # simple z update and probability
        z = (1 - eta) * z + eta * b
        z = z.softmax(dim=1)

        # compute ATB and solve for w_updates per parameter block
        w_updates = {}
        for name in tuning_params.keys():
            A = A_cache[name]['A']           # (m, d)
            # flatten probs -> length m
            vec = z.flatten()            # (m,)
            ATB = A.T @ vec                  # (d,)
            # solve (AtA_ridge @ w = ATB)
            # use torch.linalg.solve; AtA_ridge[name] is (d,d)
            w = torch.linalg.solve(AtA_ridge[name], ATB)  # (d,)
            w_updates[name] = w

        with torch.no_grad():
            out = backbone((input, text_tokens))["logits_per_image"]
            loss = criterion(out, target)
            acc1_before, acc5 = accuracy(out, target, topk=(1,5))
            # previous_acc = acc1_before
        
        for n,p in clip_model.named_parameters():
            if 'l_ff' in n:
                with torch.no_grad():
                    n_updated = 'model.' + n
                    p.copy_(torch.ones(p.shape).to(cfg.device) + w_updates[n_updated].reshape(p.shape).to(cfg.device))
            # if 'l_k' in n:
            #     with torch.no_grad():
            #         n_updated = 'model.' + n
            #         p.copy_(torch.ones(p.shape).to(cfg.device) + w_updates[n_updated].reshape(p.shape).to(cfg.device))
            # if 'l_v' in n:
            #     with torch.no_grad():
            #         n_updated = 'model.' + n
            #         p.copy_(torch.ones(p.shape).to(cfg.device) + w_updates[n_updated].reshape(p.shape).to(cfg.device))
        
        with torch.no_grad():
            out = clip_model((input, text_tokens))["logits_per_image"]
            loss = criterion(out, target)
            acc1, acc5 = accuracy(out, target, topk=(1,5))

        # track best update
        if acc1 > previous_acc:
            previous_acc = acc1
            num_stop = 0
            # store clones of best updates
            for k in w_updates:
                best_updates[k] = w_updates[k].clone()
            if (idx % 5 == 0):
                log(f"Batch {idx} epoch {epoch} - acc before {acc1_before:.4f} - acc after {acc1:.4f} - loss {loss:.4f}")
        else:
            num_stop += 1
            if num_stop >= patience:
                early_stop = True
                #if (idx % 10 == 0):

        epoch += 1

    # 5) update global accumulators using precomputed ATA and best_updates
    global_updates = {k: torch.zeros(A_cache[k]['d'], device=device) for k in tuning_params.keys()}
    for name in tuning_params.keys():
        ATA = A_cache[name]['ATA']     # precomputed
        global_At_A[name] += ATA
        # ensure best_updates[name] is shape (d,)
        global_AtA_w[name] += ATA @ best_updates[name]
        # reconstruct w_updates
        lambda_ = 5e-6
        I = torch.eye(global_At_A[name].shape[1], device=global_At_A[name].device, dtype=global_At_A[name].dtype)
        global_ATA_ridge = global_At_A[name] + lambda_*I 
        global_updates[name] = torch.linalg.solve(global_ATA_ridge, global_AtA_w[name]) 
        
    # cleanup
    del J, A_cache, AtA_ridge
    torch.cuda.empty_cache()

    return global_updates
    
@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader, 
            device, text_tokens, task_id=-1):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    # original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, 10, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            output = model((input, text_tokens))
            logits = output["logits_per_image"]
            loss = criterion(logits, target)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}