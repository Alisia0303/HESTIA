from lib.config import cfg, cfg_from_file
from lib.utils import *
from data.dataset import *
from models.model_register import get_model

import numpy as np
import torch
import random
import argparse
import pprint
import random
import math

def set_seed(seed):
    cfg.seed = seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

def prepare_model(model: torch.nn.Module, model_id=None):
    checkpoint_path = os.path.join(cfg.dtask.output_dir, 'lora_task_{}'.format(model_id), 'checkpoint/')
    
    model.Q_lora_pool_B = torch.load(checkpoint_path + '/Q_lora_pool_B.pth')
    model.Q_lora_pool_A = torch.load(checkpoint_path + '/Q_lora_pool_A.pth')
    model.V_lora_pool_B = torch.load(checkpoint_path + '/V_lora_pool_B.pth')
    model.V_lora_pool_A = torch.load(checkpoint_path + '/V_lora_pool_A.pth')

    if model_id > 0:
        model.W_Q = torch.load(checkpoint_path + "/W_Q.pth")
        model.W_V = torch.load(checkpoint_path + "/W_V.pth")
        model.new_Q_lora_B = torch.load(checkpoint_path + '/new_Q_lora_B.pth')
        model.new_Q_lora_A = torch.load(checkpoint_path + '/new_Q_lora_A.pth')
        model.new_V_lora_B = torch.load(checkpoint_path + '/new_V_lora_B.pth')
        model.new_V_lora_A = torch.load(checkpoint_path + '/new_V_lora_A.pth')

    return model

def mahalanobis_distance(x, mu, Sigma):
    diff = x - mu                             # (d,)
    inv_Sigma = torch.linalg.inv(Sigma)       # (d, d)
    dist_squared = diff @ inv_Sigma @ diff.T    # scalar
    return (dist_squared)           # Mahalanobis distance

def check_theories(data_loader, current_tid = 0, current_cluster = 0, model = None, num_tasks=10, tau=None, d=768,
                   cluster_means = None, cluster_cov = None, epsilon = 0.01):
    # Compute delta
    i_tasks = list(set(range(num_tasks)) - {current_tid})
    with torch.no_grad():
        deltas = []
        for task_id in i_tasks:
            model = prepare_model(model, task_id).to(cfg.device) # prepare function: h_i
            for s in range(tau[task_id]):
                mean, var = cluster_means[task_id][s], cluster_cov[task_id][s]
                for x in data_loader: # D_k^t
                    x = x.to(cfg.device)
                    dists = torch.mean(mahalanobis_distance(model(x, alpha=cfg.dtask.alpha)['pre_logits'], \
                                            torch.tensor(mean).to(cfg.device), torch.tensor(var).to(cfg.device)))
                delta = dists/d - 1
                deltas.append(delta)
    delta = min(deltas)
    # Compute kappa
    log_ratios = []
    Lambda_tk = torch.tensor(cluster_cov[current_tid][current_cluster]).to(cfg.device)
    for task_id in i_tasks:
        for s in range(tau[task_id]):
            Lambda_si = torch.tensor(cluster_cov[task_id][s]).to(cfg.device)
            log_det00 = torch.logdet(Lambda_tk)
            log_detsi = torch.logdet(Lambda_si)
            # log_ratio = log(Lambda_si/Lambda_00)
            log_ratio = log_detsi - log_det00
            log_ratios.append(log_ratio)
            
    kappa = min(log_ratios)
    # Check Eq. (28)
    N = (num_tasks - 1)*sum(tau) + 1
    sec_term = math.log((N/epsilon)*(math.sqrt(1 + 4*d/(math.log(N/epsilon))) + 1))
    sigma_squared = []
    with torch.no_grad():
        for task_id in i_tasks:
            model = prepare_model(model, task_id).to(cfg.device) 
            for x in data_loader: # D_k^t
                x = x.to(cfg.device)
                H = model(x, alpha=cfg.dtask.alpha)["pre_logits"]
                if H.size(0) == 1:
                    s_squared, _ = torch.max(H, dim=1)
                    s_squared = s_squared[0]
                else:
                    s_squared = torch.max(torch.var(H, dim=0, unbiased=True))#.mean()
                sigma_squared.append(s_squared)
    sigma_squared = max(sigma_squared)
    M = 3*d*sigma_squared - kappa
    first_term = 0.25*(M + torch.sqrt(torch.square(M) - 24*d*sigma_squared*kappa)) - kappa
    max_val = max(first_term, sec_term)
    threshold = max_val*2/d
    return delta, threshold

def main():
    parser = argparse.ArgumentParser(description='Theory in Continual Learning')
    parser.add_argument('--cfg', dest='cfg_file', default='./config/cifar-100.yml')

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

    tasks = range(cfg.continual.n_tasks)
    if cfg.continual.shuffle_task:
        tasks = torch.randperm(cfg.continual.n_tasks).tolist()

    if cfg.run_label == "VTAB5T-Sim50":
        data_loader = torch.load(cfg.dtask.data_path + "/dataloader_output.pt", weights_only=False)
        class_mask = np.load(cfg.dtask.data_path + "/class_mask.npy", allow_pickle=True).tolist()
        cfg.dtask.n_components = [70 , 100, 100, 20]
    else:
        data_loader, class_mask = build_continual_dataloader(batch_size=cfg.dtask.batch_size)
        cfg.dtask.n_components = [cfg.dtask.n_components]*cfg.continual.n_tasks

    log('class_mask %s ' % class_mask)
    log('Number of classes %d ' % cfg.dtask.nb_classes)

    model = get_model()

    for n, p in model.named_parameters():
        p.requires_grad = False
    
    cluster_labels = torch.load(cfg.dtask.output_dir + '/cls_labels.pth')
    cluster_means = torch.load(cfg.dtask.output_dir + '/cls_mean.pth')
    cluster_cov = torch.load(cfg.dtask.output_dir + '/cls_cov.pth')

    d, epsilon = 768, 0.01
    correct, results = 0, []
    for task_id in tasks:
        log('Task ID %d '% task_id)
        for s in range(cfg.dtask.n_components[task_id]):
            log('Cluster ID %d '% s)
            data_idx = [i for i, x in enumerate(cluster_labels[task_id]) if x == s]
            if len(data_idx) == 0:
                log("Empty cluster")
                results.append(("Empty cluster"))
                continue
            try:
                # Use train datasets because we need clusters from cluster_labels => cannot use val datasets
                datasets = [data_loader[task_id]["train"].dataset[idx_x][0] for idx_x in data_idx]
            except:
                log("data idx %s " % data_idx)
            sub_loader = torch.stack(datasets)
            sub_loader = sub_loader.unsqueeze(0)
            delta, threshold = check_theories(sub_loader, task_id, s, model, cfg.continual.n_tasks, cfg.dtask.n_components,
                                              d, cluster_means, cluster_cov, epsilon)
            if delta > threshold:
                log(f"Δ: {delta:.4f}, Threshold: {threshold:.4f}, Status: Pass")
                correct += 1
                results.append((delta, threshold, "Pass"))
            else:
                log(f"Δ: {delta:.4f}, Threshold: {threshold:.4f}, Status: Fail")
                results.append((delta, threshold, "Fail"))
    
    log("Correct: %d "% correct)
    # Write to file
    with open(cfg.dtask.output_dir + '/' + cfg.run_label + '_theories.txt') as f:
        for item in results:
            f.writelines(item + '\n')
        f.close()

if __name__ == '__main__':
    main()