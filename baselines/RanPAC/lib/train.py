from lib.config import cfg
from torch import optim
from utils.vision import *
from typing import Iterable
from pathlib import Path
from data.dataset import *
from timm.utils import accuracy
from torch.distributions import MultivariateNormal
from lib.random_projection import setup_RP, replace_fc, optimise_ridge_parameter
from lib.utils import *

import torch
import math
import numpy as np
import os
import copy


cls_mean = dict()
cls_cov = dict()
gaussian_dist = dict()
task_lora_universe_head = []

def train_one_batch(model: torch.nn.Module, criterion, input: torch.Tensor, 
                    target: torch.Tensor, optimizer: torch.optim.Optimizer,
                    device: torch.device, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None,  ):
    
    model.train(set_training_mode)
    # original_model.eval()

    for epoch in range(cfg.dtask.epochs):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', SmoothedValue(window_size=1, fmt='{value:.4f}'))
        
        output = model(input)
        logits = output["logits"]

        # Cannot mask because we dont know task_id
        # here is the trick to mask out classes of non-current tasks
        if True and class_mask is not None:
            mask = class_mask#[task_id]
            not_mask = np.setdiff1d(np.arange(cfg.dtask.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        
        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        import sys
        if not math.isfinite(loss.item()):
            print('reg loss ', output["reg_W_Q"], output["reg_W_V"])
            print('ortho loss ', output["ortho_loss_Q"] + output["ortho_loss_V"])
            print("Loss is {}, stopping training".format(loss.item()))
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
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader, 
            device, task_id=-1,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)
    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, 10, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(input)
            logits = output['logits']
            loss = criterion(logits, target)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, acc_matrix=None):
    stat_matrix = np.zeros((3, cfg.continual.n_tasks))  # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id + 1):
        test_stats = evaluate(model=model, data_loader=data_loader[i]['val'], device=device, task_id=i,)
        
        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1,
        avg_stat[0],
        avg_stat[1],
        avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats

def train_and_evaluate(tasks, model, criterion, data_loader, device, class_mask, acc_matrix, 
                       added_units=4, network=None, M=0, Q=None, G=None,
                       mapping_classes=None, classes_seen_so_far=None):
    # Create new optimizer for each task to clear optimizer status
    for task_id in (tasks):    
        #if task_id > 0:
            
        #     model.add_new_units(added_units)
        #     model.to(device)

        
        optimizer = optim.Adam(model.parameters(), lr=cfg.dtask.lr)

        for idx, (input, target) in enumerate(data_loader[task_id]['train']):
            print(f"Task {task_id} - Batch {idx}")
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            new_classes = torch.unique(target).tolist()
            train_stats = train_one_batch(model=model, criterion=criterion,
                                          input=input, target=target, optimizer=optimizer,
                                          device=device, max_norm=1.0,
                                          set_training_mode=True, task_id=task_id, class_mask=new_classes)
            
            for n, p in model.named_parameters():
                if ('_lora_' in n):
                    p.requires_grad = False
                    # print(n)
        
            log('BEGIN TO LEARN LDA CLASSIFIER OF TASK {}'.format(task_id))
            # Update RP model
            network.update_backbone(copy.deepcopy(model))
            del network.fc
            network.fc=None
            #new_heads = sum([len(class_mask[tid]) for tid in range(task_id + 1)]) # need updates
            classes_seen_so_far = classes_seen_so_far | set(new_classes)
            network.update_fc(len(classes_seen_so_far))
            #freeze RP backbone
            for n, p in network.named_parameters():
                if 'convnet' in n:
                    p.requires_grad = False

            n_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
            log('number of params: %d' % n_parameters)
            if task_id == 0:
                W_rand = setup_RP(network, M)
            
            # need updates
            Y, Features_h, Q, G = replace_fc(network, input, target, W_rand, Q, G, mapping_classes)
        
        ridge=optimise_ridge_parameter(Features_h,Y)
        Wo=torch.linalg.solve(G+ridge*torch.eye(G.size(dim=0)),Q).T
        network.fc.weight.data=Wo[0:network.fc.weight.shape[0],:].to(cfg.device)
        # Store task-id network
        #task_lora_universe_head.append(copy.deepcopy(network))

        log('BEGIN TO EVALUATE ALL TASKS UNTIL NOW - TASK {}'.format(task_id))
        # head_weight = copy.deepcopy(network.fc.weight.data) # Get latest classification weight
        # for tid in range(task_id + 1):
        #     task_lora_universe_head[tid].fc.weight = torch.nn.Parameter(head_weight)

        test_stats = evaluate_till_now(model=network, data_loader=data_loader, device=device, 
                                       task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix)

        
        # if task_id > 0:
            # test_stats = evaluate_till_now(data_loader, device, task_id, acc_matrix=acc_matrix, 
            #                             alpha=alpha, models=task_lora_universe_head, 
            #                             gaussian_dist=gaussian_dist, mapping_classes=mapping_classes)
        # model.merge_units(optimizer)
            
    


    