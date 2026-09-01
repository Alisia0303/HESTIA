from lib.config import cfg, cfg_from_file
from lib.utils import *
from data.dataset import *
from lib.train import prepare_model, evaluate_till_now_with_pred_tid
from lib.random_projection import SimpleVitNet, setup_RP, replace_fc, optimise_ridge_parameter, predict_task_id, evaluate_till_now_wo_matching
from models.model_register import get_model

import numpy as np
import torch
import random
import argparse
import pprint
import random
import copy

def learn_continually():
    log('\nRunning experiments using Random Projection layer')

    # data_loader, class_mask = build_continual_dataloader(batch_size=cfg.dtask.batch_size)
    if cfg.run_label == "VTAB5T-Sim50":
        data_loader = torch.load(cfg.dtask.data_path + "/dataloader_output.pt")
        class_mask = np.load(cfg.dtask.data_path + "/class_mask.npy", allow_pickle=True).tolist()
    elif cfg.run_label == "VTAB5T-Sim75":
        data_loader = torch.load(cfg.dtask.data_path + "/dataloader_output_sim75.pt")
        class_mask = np.load(cfg.dtask.data_path + "/class_mask_sim75.npy", allow_pickle=True).tolist()
    else:
        data_loader, class_mask = build_continual_dataloader(batch_size=cfg.dtask.batch_size)
    log('class_mask %s ' % class_mask)
    log('Number of classes %d ' % cfg.dtask.nb_classes)

    mapping_classes = dict()
    for i in range(len(class_mask)):
        for j in range(len(class_mask[i])):
            if i == 0:
                mapping_classes[class_mask[i][j]] = j
            else:
                mapping_classes[class_mask[i][j]] = (mapping_classes[class_mask[i-1][-1]] + 1) + j


    network = SimpleVitNet()
    M = 10000
    Q=torch.zeros(M, cfg.dtask.nb_classes)
    G=torch.zeros(M,M)
    acc_matrix = np.zeros((cfg.continual.n_tasks, cfg.continual.n_tasks))

    cls_mean = torch.load(cfg.dtask.output_dir + '/cls_mean.pth', weights_only=False)
    cls_cov = torch.load(cfg.dtask.output_dir + '/cls_cov.pth', weights_only=False)
    task_specific_backbones = []
    model = get_model()

    for task_id in range(cfg.continual.n_tasks):
    # for task_id in [cfg.continual.n_tasks - 1]:
        if task_id > 0:
            model.add_new_units(cfg.dtask.added_units)
            model.to(cfg.device)
        
        model = prepare_model(model, model_id=task_id)
        task_specific_backbones.append(copy.deepcopy(model))
        network.update_backbone(model)
        del network.fc
        network.fc=None
        new_heads = sum([len(class_mask[tid]) for tid in range(task_id + 1)])
        network.update_fc(new_heads)
        #freeze backbone
        for n, p in network.named_parameters():
            if 'convnet' in n:
                p.requires_grad = False
                
        n_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
        log('number of params: %d' % n_parameters)
        if task_id == 0:
            W_rand = setup_RP(network, M)
        
        Y, Features_h, Q, G = replace_fc(network, data_loader[task_id]["train"], W_rand, Q, G, mapping_classes)
        ridge=optimise_ridge_parameter(Features_h,Y)
        Wo=torch.linalg.solve(G+ridge*torch.eye(G.size(dim=0)),Q).T
        network.fc.weight.data=Wo[0:network.fc.weight.shape[0],:].to(cfg.device)

        #Eval
        if not cfg.dtask.wo_matching:
            cls_mean_until_task, cls_cov_until_task = {}, {}
            for tid in range(task_id + 1):
                cls_mean_until_task[tid] = cls_mean.get(tid)
                cls_cov_until_task[tid] = cls_cov.get(tid)

            pred_task_id = predict_task_id(task_id, data_loader, cls_mean_until_task, cls_cov_until_task, 
                                    cfg.device, added_units=cfg.dtask.added_units, alpha=cfg.dtask.alpha)
            task_specific_network = []

            for i in range(task_id + 1):
                network_i = copy.deepcopy(network)
                network_i.update_backbone(task_specific_backbones[i])
                network_i.eval()
                task_specific_network.append(copy.deepcopy(network_i))

            test_stats = evaluate_till_now_with_pred_tid(pred_task_id, task_specific_network, data_loader,
                                cfg.device, task_id=task_id, acc_matrix=acc_matrix, alpha=cfg.dtask.alpha, mapping_classes=mapping_classes)
        else:
            test_stats = evaluate_till_now_wo_matching(network, data_loader, cfg.device, task_id, acc_matrix=acc_matrix, 
                                                       alpha=cfg.dtask.alpha, mapping_classes=mapping_classes)

        if task_id > 0:
            model.merge_units()

def set_seed(seed):
    cfg.seed = seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

def main():
    parser = argparse.ArgumentParser(description='Adaptive Lora for Continual Learning')
    parser.add_argument('--cfg', dest='cfg_file', default='./config/MRPC.yml')

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
        learn_continually()


if __name__ == '__main__':
    main()
