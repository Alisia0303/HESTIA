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
from data.dataset import *
from utils.vision import *
from torch.func import jacrev, functional_call
from lib.train import tangent_tuning_a_batch, train_one_batch
from pathlib import Path
from lib.signatures import OnlineDiagonalGMM, detect_convergence_stationary_streaming
from timm.models import create_model
from timm.utils import accuracy
from torch import optim

def set_seed(seed):
    cfg.seed = seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

def run_evaluation():
    log('\nRunning evaluation using TANGENT FINETUNING')

    # Load datasets.
    data_loader, _ = build_continual_dataloader(batch_size=cfg.dtask.batch_size)
    class_mask = torch.load(cfg.dtask.output_dir + '/class_mask.pth')
    log('class_mask %s ' % class_mask)
    log('Number of classes %d ' % cfg.dtask.nb_classes)
    count_save = len(class_mask)

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
    head_weight = torch.load(cfg.dtask.output_dir + f'/ia3_task_{count_save - 1}/head_weight.pth')
    head_bias = torch.load(cfg.dtask.output_dir + f'/ia3_task_{count_save - 1}/head_bias.pth')

    with torch.no_grad():
        vit_model.head.weight.copy_(head_weight)
        vit_model.head.bias.copy_(head_bias) 

    # Load signatures (keys in DB).
    task_updates = dict()
    cls_stat = dict()

    for task_id in range(count_save):
        cls_stat[task_id] = dict()
        remove_rows = list()
        
        checkpoint_path = os.path.join(cfg.dtask.output_dir, 'ia3_task_{}'.format(task_id))
        cls_mean = torch.load(os.path.join(checkpoint_path, 'mean.pth') )
        cls_cov = torch.load(os.path.join(checkpoint_path, 'vars.pth') )
        cls_count = torch.load(os.path.join(checkpoint_path, 'counts.pth'))
        
        for cluster_id in range(len(cls_mean)):
            if cls_count[cluster_id] < 10: # not consider insignificant clusters
                remove_rows.append(cluster_id)
                continue

        mask = torch.ones(cls_mean.shape[0], dtype=torch.bool, device=cls_mean.device)
        mask[remove_rows] = False
        
        cls_stat[task_id]["mean"] = cls_mean[mask]
        cls_stat[task_id]["var"] = cls_cov[mask]
    
        global_updates = torch.load(os.path.join(checkpoint_path, 'global_updates.pth'))

        for n,p in vit_model.named_parameters():
            if 'l_ff' in n:
                with torch.no_grad():
                    n_updated = 'model.' + n
                    p.copy_(torch.ones(p.shape).to(cfg.device) + global_updates[n_updated].reshape(p.shape).to(cfg.device))
            if 'l_k' in n:
                with torch.no_grad():
                    n_updated = 'model.' + n
                    p.copy_(torch.ones(p.shape).to(cfg.device) + global_updates[n_updated].reshape(p.shape).to(cfg.device))
            if 'l_v' in n:
                with torch.no_grad():
                    
                    n_updated = 'model.' + n
                    p.copy_(torch.ones(p.shape).to(cfg.device) + global_updates[n_updated].reshape(p.shape).to(cfg.device))

        task_updates[task_id] = copy.deepcopy(vit_model)
    
    #Time to run evaluation.
    acc_matrix = np.zeros((cfg.continual.n_tasks, cfg.continual.n_tasks))
    test_stats = evaluate_till_now(data_loader, cfg.device, cfg.continual.n_tasks - 1, acc_matrix, model_lst=task_updates,
                                   cls_stat=cls_stat, class_mask=class_mask)
    
    
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
