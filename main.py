import os
import numpy as np
import torch
import random
import argparse
import pprint
import random
import copy
import models.model_register
import sys

from lib.config import cfg, cfg_from_file
from lib.utils import *
from lib.ia3 import is_ia3_param, collect_ia3_params, apply_ia3_updates, reset_ia3_to_identity
from lib.checkpointing import save_task_checkpoint
from data.dataset import *
from utils.vision import *
from torch.func import jacrev, functional_call
from lib.train import tangent_tuning_a_batch, train_one_batch
from pathlib import Path
from lib.signatures import OnlineDiagonalGMM, detect_convergence_stationary_streaming
from evaluate import evaluate_at_task
from timm.models import create_model
from timm.utils import accuracy
from torch import optim

# Hyperparameters for the closed-form linearized adaptation (Eq. "ridge-sol").
TANGENT_ETA = 0.03
TANGENT_RIDGE_LAMBDA = 1.0
# Learning rate for the (gradient-based) classification-head warm start.
HEAD_LR = 1e-3

def set_seed(seed):
    cfg.seed = seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

class Net(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X, target):
        out = self.model(X)
        logits = out["logits"]
        logits = logits.softmax(dim=1)
        target_logits = torch.gather(logits, 1, target.reshape(target.size(0),1).to(cfg.device)).squeeze(1) 
        out["target_logits"] = target_logits
        return target_logits

def log_gmm_summary(online_gauss, prefix=""):
    """Log a concise summary of the current Gaussian-mixture task key."""
    weights = online_gauss.counts / (online_gauss.counts.sum() + 1e-8)
    log(f"\n{prefix} ===== GMM task key summary =====")
    log(f"Clusters: {online_gauss.means.shape[0]} | Feature dim: {online_gauss.means.shape[1]}")
    for k in range(online_gauss.means.shape[0]):
        log(f"  cluster {k}: count={online_gauss.counts[k].item():.1f}, "
            f"weight={weights[k].item():.3f}, "
            f"mean_norm={online_gauss.means[k].norm().item():.3f}, "
            f"var_mean={online_gauss.vars[k].mean().item():.4f}")
    log("==================================\n")

def learn_continually():
    log('\nRunning experiments using TANGENT FINETUNING')
    tasks = range(cfg.continual.n_tasks)
    if cfg.continual.shuffle_task:
        tasks = torch.randperm(cfg.continual.n_tasks).tolist()
 
    # Define ViT model.
    vit_model = create_model(
        cfg.dtask.model,
        pretrained=cfg.dtask.pretrained,
        num_classes=cfg.dtask.nb_classes,
        drop_rate=cfg.dtask.drop,
        drop_path_rate=cfg.dtask.drop_path,
        drop_block_rate=None,
        head_type=cfg.dtask.head_type,
    )
    vit_model.to(cfg.device)
    for n, p in vit_model.named_parameters():
        p.requires_grad = "head" in n
 
    # Load datasets.
    data_loader, class_mask = build_continual_dataloader(batch_size=cfg.dtask.batch_size)
    log('class_mask %s ' % class_mask)
    log('Number of classes %d ' % cfg.dtask.nb_classes)
 
    # `net_g` wraps a frozen copy of the backbone and exposes target-class
    # softmax probabilities; it is only used to compute Jacobians (via
    # `torch.func.jacrev`) for the closed-form linearized update.
    backbone = copy.deepcopy(vit_model)
    net_g = Net(backbone)
    net_g.to(cfg.device)
    net_g.eval()
 
    tuning_params = collect_ia3_params(net_g)
    log("Tuning parameters are")
    for k, v in tuning_params.items():
        log(f"{k} : {v.shape}")
 
    head_params = sum(p.numel() for n, p in vit_model.named_parameters() if p.requires_grad and "head" in n)
    ia3_params = sum(p.numel() for n, p in net_g.named_parameters() if is_ia3_param(n))
    log(f"Head trainable params: {head_params}")
    log(f"IA3 trainable params: {ia3_params}")
    log(f"Total trainable params (IA3 + head): {head_params + ia3_params}")
 
    def wrapped_g(params, X, target):
        """Wraps `net_g` so that `torch.func.jacrev` can differentiate w.r.t. `params`."""
        return functional_call(net_g, params, (X, target))
 
    jacobian_fn = jacrev(wrapped_g, argnums=0)
 
    global_At_A = {layer: torch.zeros((p.size(-1), p.size(-1)), device=cfg.device)
                   for layer, p in tuning_params.items()}
    global_AtA_w = {layer: torch.zeros(p.size(-1), device=cfg.device)
                     for layer, p in tuning_params.items()}
    AtA_ridge = dict()
 
    previous_global_update = None
    count_save, batch_idx = 0, 0
    COOLDOWN_LIMIT = cfg.signature.cooldown_limit
    cooldown = COOLDOWN_LIMIT
    is_converged = False
    task_class_mask = dict()
    mahal_history = {count_save: []}
    within_seen_classes = set()
 
    criterion = torch.nn.CrossEntropyLoss().to(cfg.device)
    optimizer = optim.Adam(vit_model.parameters(), lr=HEAD_LR)
    online_gauss = OnlineDiagonalGMM(n_clusters=cfg.signature.n_clusters, feature_dim=768, device="cpu",
                                      split_threshold=cfg.signature.split_threshold, cfg=cfg)
    log_gmm_summary(online_gauss)
 
    for task_id in tasks:
        log(f"Task {task_id}: {len(data_loader[task_id]['train'])} training batches")
 
    for task_id in tasks:
        for (input, target) in data_loader[task_id]["train"]:
            input, target = input.to(cfg.device), target.to(cfg.device)
            new_classes = torch.unique(target).tolist()
            target_classes = set(new_classes)
            log(f"New class coming: {new_classes}")
 
            vit_model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast():
                temp_features = vit_model(input)['pre_logits'].cpu()
 
            # Routing / change-point score: min Mahalanobis-plus-prior
            # distance to the current task key (Eq. for S_t(z) in the paper).
            mean_ood = online_gauss.score(temp_features).mean()
            mahal_history[count_save].append(mean_ood)
            log(f"Task {count_save}, Batch {batch_idx}, mean_ood: {mean_ood}")
 
            if cooldown > 0:
                cooldown -= 1
                log(f"[Cooldown active] Skipping detection... {cooldown} left")
                within_seen_classes |= target_classes  # update seen classes before converging
            else:
                if not is_converged:  # start detecting
                    within_seen_classes |= target_classes  # still update seen classes before converging
                    values_t = torch.tensor(mahal_history[count_save])
                    converged, converged_idx = detect_convergence_stationary_streaming(
                        values_t, window=cfg.signature.window_t, k=cfg.signature.k, cool_down=COOLDOWN_LIMIT)
                    if converged:
                        is_converged = True
                        converged_value = values_t[converged_idx].item() if converged_idx is not None else None
                        log("Convergence starts at index: %d" % converged_idx)
                        log("Value at convergence: %s " % converged_value)
 
                # After convergence, if new classes come then we mark this as a new incoming distribution.
                elif is_converged and not target_classes.issubset(within_seen_classes):
                    log(f"Distribution {count_save}-th is detected at batch {batch_idx} with mean_ood: {mean_ood}")
                    log(f"New coming classes: {new_classes}")
 
                    save_task_checkpoint(cfg.dtask.output_dir, count_save, global_At_A, global_AtA_w,
                                          online_gauss, global_updates=previous_global_update)
                    task_class_mask[count_save] = list(within_seen_classes)
 
                    # Reset everything for the next distribution.
                    within_seen_classes = set()
                    is_converged = False
                    mahal_history[count_save + 1] = []
                    cooldown = COOLDOWN_LIMIT
                    for layer in tuning_params.keys():
                        global_At_A[layer].zero_()
                        global_AtA_w[layer].zero_()
                    reset_ia3_to_identity(vit_model, cfg.device)
                    online_gauss.reset()
                    optimizer = optim.Adam(vit_model.parameters(), lr=HEAD_LR)
                    count_save += 1
                    batch_idx = 0
 
            # Gradient-based warm start of the classification head for the ViT.
            log(f"Within seen class: {within_seen_classes}")
            train_one_batch(model=vit_model, criterion=criterion, input=input, target=target,
                             optimizer=optimizer, device=cfg.device, max_norm=1.0,
                             set_training_mode=True, task_id=count_save, class_mask=new_classes)
 
            # Sync the (just-updated) head into the Jacobian-computation
            # copies before solving the closed-form update.
            with torch.no_grad():
                net_g.model.head.weight.copy_(vit_model.head.weight)
                net_g.model.head.bias.copy_(vit_model.head.bias)
                backbone.head.weight.copy_(vit_model.head.weight)
                backbone.head.bias.copy_(vit_model.head.bias)
            for p in net_g.parameters():
                p.requires_grad = False
 
            # Closed-form linearized IA3 update (Eq. "ridge-sol" in the paper).
            global_updates = tangent_tuning_a_batch(
                input, target, global_At_A, global_AtA_w, AtA_ridge, net_g, jacobian_fn,
                vit_model, tuning_params, backbone, eta=TANGENT_ETA, lambda_=TANGENT_RIDGE_LAMBDA)
            apply_ia3_updates(vit_model, global_updates, cfg.device)
 
            # Accuracy of the global update on the current batch (classes outside
            # this batch are masked out, since task identity is unknown).
            with torch.no_grad():
                out = vit_model(input)["logits"]
                not_mask = np.setdiff1d(np.arange(cfg.dtask.nb_classes), new_classes)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(cfg.device)
                out = out.index_fill(dim=1, index=not_mask, value=float('-inf'))
                acc1, acc5 = accuracy(out, target, topk=(1, 5))
            log(f"GLOBAL UPDATES ON BATCH {batch_idx} is: {acc1}")
 
            with torch.no_grad():
                features = vit_model(input)['pre_logits'].cpu()
            online_gauss.update(features, batch_idx)
 
            previous_global_update = global_updates
            batch_idx += 1
 
        # Evaluation at the end of each task.
        with torch.no_grad():
            temp_class_mask = task_class_mask
            temp_class_mask[count_save] = list(within_seen_classes)
            task_test_stats = evaluate_at_task(
                vit_model.head.weight, vit_model.head.bias, previous_global_update,
                online_gauss.means, online_gauss.vars, online_gauss.counts,
                data_loader, temp_class_mask, task_id, cfg)
            log(f"Global Acc Matrix after truth (Not predicted) task {task_id} : \n {task_test_stats['acc_matrix']}")
 
    # Final checkpoint after the last batch of the last task.
    save_task_checkpoint(cfg.dtask.output_dir, count_save, global_At_A, global_AtA_w, online_gauss,
                          global_updates=global_updates, head_weight=vit_model.head.weight,
                          head_bias=vit_model.head.bias, mahal_history=mahal_history)
    task_class_mask[count_save] = list(within_seen_classes)
    torch.save(task_class_mask, os.path.join(cfg.dtask.output_dir, 'class_mask.pth'))

def main():
    parser = argparse.ArgumentParser(description='TANGENT FINETUNING in Continual Learning')
    parser.add_argument('--cfg', dest='cfg_file', default='./configs/cifar-100.yml')

    args = parser.parse_args()
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    cfg_name = os.path.splitext(os.path.basename(args.cfg_file))[0]
    # logging.basicConfig(
    #     filename=os.path.join(LOG_DIR, f"{cfg_name}.txt"),
    #     filemode="a",
    #     level=logging.DEBUG,
    #     format="%(asctime)s | %(levelname)s | %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    #     force=True
    # )
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    log_dir = "logs"
    os.makedirs(cfg.dtask.output_dir, exist_ok=True)

    final_filename = os.path.join(
        log_dir,
        f"{cfg_name}_seed{cfg.seed}_nc{cfg.signature.n_clusters}_k{cfg.signature.k}_wt{cfg.signature.window_t}_more_trial.log"
    )
    logging.basicConfig(
        filename=final_filename,
        filemode="a",
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    log(pprint.pformat(cfg))

    cfg.output_dir = os.path.join(cfg.dtask.output_dir, f"{cfg_name}_seed_{cfg.seed}")

    gpu_list = cfg.gpu_ids.split(',')
    gpus = [int(iter) for iter in gpu_list]
    cfg.device = torch.device('cuda:' + str(gpus[0]))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    set_seed(cfg.seed)

    if cfg.continual.method.run_merlin:
        learn_continually()

if __name__ == '__main__':
    main()