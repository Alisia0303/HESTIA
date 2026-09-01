from lib.config import cfg, cfg_from_file
from lib.utils import *
from data.dataset import *
from models.model_register import get_model
from lib.random_projection import SimpleVitNet
from torch.distributions import MultivariateNormal
from lib.random_projection import setup_RP, replace_fc, optimise_ridge_parameter
from lib.utils import *
from lib.train import prepare_model, evaluate_till_now

import numpy as np
import torch
import random
import argparse
import pprint
import random
import copy

def set_seed(seed):
    cfg.seed = seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

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

    # Define a RP model
    network = SimpleVitNet()
    acc_matrix = np.zeros((cfg.continual.n_tasks, cfg.continual.n_tasks))
    M = 10000
    Q=torch.zeros(M, cfg.dtask.nb_classes)
    G=torch.zeros(M,M)
    cls_mean = torch.load(cfg.dtask.output_dir + '/cls_mean.pth', weights_only=False)
    cls_cov = torch.load(cfg.dtask.output_dir + '/cls_cov.pth', weights_only=False)
    task_lora_universe_head = []
    model = get_model()

    # Load Gaussians
    gaussian_dist = dict()
    for task_id in range(10):
        gaussian_dist[task_id] = []
        for cluster_id in range(len(cls_mean[task_id])):
            mean = cls_mean[task_id][cluster_id]
            var = cls_cov[task_id][cluster_id]
            if var.mean() == 0:
                print("var.mean is empty")
                continue
            m = MultivariateNormal(torch.tensor(mean).to(cfg.device), (torch.tensor(var).to(cfg.device) + 1e-4 * torch.eye(mean.shape[0]).to(cfg.device)))
            gaussian_dist[task_id].append(copy.deepcopy(m))

    for task_id in range(10):
        model = prepare_model(model, task_id).to(cfg.device)
        network.update_backbone(copy.deepcopy(model))
        del network.fc
        network.fc=None
        new_heads = sum([len(class_mask[tid]) for tid in range(task_id + 1)])
        network.update_fc(new_heads)
        #freeze RP backbone
        for n, p in network.named_parameters():
            if 'convnet' in n:
                p.requires_grad = False

        if task_id == 0:
                W_rand = setup_RP(network, M)
            
        Y, Features_h, Q, G = replace_fc(network, data_loader[task_id]["train"], W_rand, Q, G)
        ridge=optimise_ridge_parameter(Features_h,Y)
        Wo=torch.linalg.solve(G+ridge*torch.eye(G.size(dim=0)),Q).T
        network.fc.weight.data=Wo[0:network.fc.weight.shape[0],:].to(cfg.device)
        # Store task-id network
        task_lora_universe_head.append(copy.deepcopy(network))
        head_weight = copy.deepcopy(network.fc.weight.data)
        for tid in range(len(task_lora_universe_head)):
            task_lora_universe_head[tid].fc.weight = torch.nn.Parameter(head_weight)

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        test_stats = evaluate_till_now(data_loader, cfg.device, task_id, acc_matrix=acc_matrix, 
                                                alpha=0.3, models=task_lora_universe_head, 
                                                gaussian_dist=gaussian_dist, mapping_classes=None)

        end_event.record()

        torch.cuda.synchronize()  # Ensure all operations finish
        total_time = start_event.elapsed_time(end_event)  # in milliseconds
        # avg_time = total_time / num_runs

        print(f"Total inference time runs: {total_time:.4f} ms")

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
