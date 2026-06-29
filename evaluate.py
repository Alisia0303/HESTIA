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
from lib.ia3 import apply_ia3_updates
from lib.checkpointing import load_task_signature, mask_significant_clusters, task_checkpoint_dir
from data.dataset import *
from utils.vision import *
from torch.func import jacrev, functional_call
from lib.train import tangent_tuning_a_batch, train_one_batch
from pathlib import Path
# from lib.signatures import OnlineDiagonalGMM, detect_convergence_stationary_streaming
from timm.models import create_model
from timm.utils import accuracy
from torch import optim

def set_seed(seed):
    cfg.seed = seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

def _build_vit_model():
    model = create_model(
        cfg.dtask.model,
        pretrained=cfg.dtask.pretrained,
        num_classes=cfg.dtask.nb_classes,
        drop_rate=cfg.dtask.drop,
        drop_path_rate=cfg.dtask.drop_path,
        drop_block_rate=None,
        head_type=cfg.dtask.head_type,
    )
    model.to(cfg.device)
    return model

def evaluate_at_task(current_head, current_bias, current_global_updates, current_mean, current_var,
                      current_counts, data_loader, class_mask, current_real_task, cfg):
    """Evaluate every detected distribution's adapter on the streams seen so far.
 
    `current_*` describe the in-progress (not yet checkpointed) distribution;
    all earlier distributions are reloaded from their saved checkpoints.
    """
    vit_model = _build_vit_model()
    with torch.no_grad():
        vit_model.head.weight.copy_(current_head)
        vit_model.head.bias.copy_(current_bias)
 
    log('class_mask %s ' % class_mask)
 
    task_updates = dict()
    cls_stat = dict()
    current_count_save = sum(
        1 for name in os.listdir(cfg.dtask.output_dir)
        if name.startswith("ia3") and os.path.isdir(os.path.join(cfg.dtask.output_dir, name))
    )
 
    for task_id in range(current_count_save + 1):
        if task_id == current_count_save:
            # In-progress distribution: use the live (in-memory) statistics.
            mean, var = mask_significant_clusters(current_mean, current_var, current_counts)
            apply_ia3_updates(vit_model, current_global_updates, cfg.device)
        else:
            # Previously detected distribution: reload its saved checkpoint.
            sig = load_task_signature(cfg.dtask.output_dir, task_id)
            mean, var = sig["mean"], sig["var"]
            apply_ia3_updates(vit_model, sig["global_updates"], cfg.device)
 
        cls_stat[task_id] = {"mean": mean, "var": var}
        task_updates[task_id] = copy.deepcopy(vit_model)
 
    if current_real_task == 0:
        acc_matrix = np.zeros((cfg.continual.n_tasks, cfg.continual.n_tasks))
    else:
        acc_matrix = torch.load(os.path.join(cfg.dtask.output_dir, 'global_acc_matrix.pth'), weights_only=False)
 
    test_stats = evaluate_till_now(data_loader, cfg.device, current_real_task, acc_matrix=acc_matrix,
                                    model_lst=task_updates, cls_stat=cls_stat, class_mask=class_mask)
    torch.save(test_stats['acc_matrix'], os.path.join(cfg.dtask.output_dir, 'global_acc_matrix.pth'))
    return test_stats

def run_evaluation():
    log('\nRunning evaluation using TANGENT FINETUNING')
    data_loader, _ = build_continual_dataloader(batch_size=cfg.dtask.batch_size)
    class_mask = torch.load(os.path.join(cfg.dtask.output_dir, 'class_mask.pth'))
    log('class_mask %s ' % class_mask)
    log('Number of classes %d ' % cfg.dtask.nb_classes)
 
    count_save = len(class_mask)
    vit_model = _build_vit_model()
 
    last_ckpt = task_checkpoint_dir(cfg.dtask.output_dir, count_save - 1)
    head_weight = torch.load(os.path.join(last_ckpt, 'head_weight.pth'))
    head_bias = torch.load(os.path.join(last_ckpt, 'head_bias.pth'))
    with torch.no_grad():
        vit_model.head.weight.copy_(head_weight)
        vit_model.head.bias.copy_(head_bias)
 
    task_updates = dict()
    cls_stat = dict()
    for task_id in range(count_save):
        sig = load_task_signature(cfg.dtask.output_dir, task_id)
        cls_stat[task_id] = {"mean": sig["mean"], "var": sig["var"]}
        apply_ia3_updates(vit_model, sig["global_updates"], cfg.device)
        task_updates[task_id] = copy.deepcopy(vit_model)
 
    acc_matrix = np.zeros((cfg.continual.n_tasks, cfg.continual.n_tasks))
    evaluate_till_now(data_loader, cfg.device, cfg.continual.n_tasks - 1, acc_matrix,
                       model_lst=task_updates, cls_stat=cls_stat, class_mask=class_mask)
    
    
def main():
    parser = argparse.ArgumentParser(description='TANGENT FINETUNING in Continual Learning')
    parser.add_argument('--cfg', dest='cfg_file', default='./config/cifar-100.yml')
    parser.add_argument('--portion', '-p',  type=int, default=1)

    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    log(pprint.pformat(cfg))

    gpu_list = cfg.gpu_ids.split(',')
    gpus = [int(iter) for iter in gpu_list]
    cfg.device = torch.device('cuda:' + str(gpus[0]))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    set_seed(cfg.seed)

    if cfg.continual.method.run_merlin:
        run_evaluation()

if __name__ == '__main__':
    main()
