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
        if "head" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False 

    # Load datasets.
    data_loader, class_mask = build_continual_dataloader(batch_size=cfg.dtask.batch_size)
    log('class_mask %s ' % class_mask)
    log('Number of classes %d ' % cfg.dtask.nb_classes)

    # Define net_g
    backbone = copy.deepcopy(vit_model)
    net_g = Net(backbone)
    net_g.to(cfg.device)
    net_g.eval()

    tuning_params = dict()
    for n,p in net_g.named_parameters():
        if 'l_ff' in n:
            tuning_params[n] = p
        if 'l_k' in n:
            tuning_params[n] = p
        if 'l_v' in n:
            tuning_params[n] = p
    
    log("Tuning parameters are")
    # log(tuning_params)
    for k in tuning_params.keys():
        log(f"{k} : {tuning_params[k].shape}")

    #Define the wrapped function that works with `torch.func.jacrev`
    def wrapped_g(params, X, target):
        """Wraps the neural network g while keeping its structure intact using functional_call."""
        out = functional_call(net_g, params, (X,target))
        return out
    
    jacobian_fn = jacrev(wrapped_g, argnums = 0) 
    global_At_A = dict()
    global_AtA_w = dict()
    AtA_ridge = dict()
    for layer in tuning_params.keys():
        size = tuning_params.get(layer).size(-1)
        global_At_A[layer] = torch.zeros((size, size)).to(cfg.device)
        global_AtA_w[layer] = torch.zeros((size)).to(cfg.device)

    previous_global_update = None
    count_save, batch_idx = 0, 0
    COOLDOWN_LIMIT = cfg.signature.cooldown_limit
    cooldown = COOLDOWN_LIMIT
    is_converged = False 
    class_mask = dict()
    mahal_history = dict()
    mahal_history[count_save] = list()
    within_seen_classes = set()
    criterion = torch.nn.CrossEntropyLoss().to(cfg.device)
    optimizer = optim.Adam(vit_model.parameters(), lr=0.001)
    online_gauss = OnlineDiagonalGMM(n_clusters=cfg.signature.n_clusters, feature_dim=768, device="cpu", 
                                     split_threshold=cfg.signature.split_threshold, cfg=cfg)

    for task_id in tasks:
        for (input, target) in (data_loader[task_id]["train"]):       
            input, target = input.to(cfg.device), target.to(cfg.device)     
            new_classes = torch.unique(target).tolist()
            target_classes = set(new_classes)
            vit_model.eval()

            with torch.no_grad(), torch.cuda.amp.autocast():
                temp_features = vit_model(
                        (input)
                )['pre_logits'].cpu()
                
            # Compute mahalanobis distance. Same as learning Online GMM.
            diff = temp_features[:, None, :] - online_gauss.means[None, :, :]
            mahal = (diff ** 2 / online_gauss.vars[None, :, :]).sum(dim=2)
            weights = online_gauss.counts / (online_gauss.counts.sum() + 1e-6)
            prior = -torch.log(weights + online_gauss.min_cluster_weight)
            scores = mahal + prior
            min_score = torch.min(scores, dim=1).values
            mean_ood = min_score.mean()
            mahal_history[count_save].append(mean_ood)
            
            log(f"Task {count_save}, Batch {batch_idx}, mean_ood: {mean_ood}")

            if cooldown > 0:
                cooldown -= 1
                log(f"[Cooldown active] Skipping detection... {cooldown} left")
                within_seen_classes = within_seen_classes | target_classes # update seen classes before converging
            else:
                if (not is_converged): # start detecting
                    within_seen_classes = within_seen_classes | target_classes # still update seen classes before converging
                    values_t = torch.tensor(mahal_history[count_save])
                    converged, converged_idx = detect_convergence_stationary_streaming(values_t, window=cfg.signature.window_t, 
                                                                                       k=cfg.signature.k, cool_down=COOLDOWN_LIMIT)
                    if converged:
                        is_converged = True
                        converged_value = values_t[converged_idx].item() if converged_idx is not None else None
                        log("Convergence starts at index: %d" % converged_idx)
                        log("Value at convergence: %d " % converged_value )
            
                # after convergence, if new classes come then we mark it as a new coming dist.
                elif is_converged and (not target_classes.issubset(within_seen_classes)): 
                    log(f"Distribution {count_save}-th is detected at batch {batch_idx} with mean_ood: {mean_ood}")
                    log(f"New coming classes: %s " % new_classes)
                    checkpoint_path = os.path.join(cfg.dtask.output_dir, f'ia3_task_{count_save}')
                    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
                    torch.save(global_At_A, checkpoint_path + '/global_At_A.pth')
                    torch.save(global_AtA_w, checkpoint_path + '/global_AtA_w.pth')
                    torch.save(online_gauss.means, checkpoint_path + '/mean.pth')
                    torch.save(online_gauss.vars, checkpoint_path + '/vars.pth')
                    torch.save(online_gauss.counts, checkpoint_path + '/counts.pth')
                    torch.save(previous_global_update, checkpoint_path + '/global_updates.pth')
                    class_mask[count_save] = list(within_seen_classes)
                    # reset everything
                    within_seen_classes = set()
                    is_converged = False
                    mahal_history[count_save + 1] = list()
                    cooldown = COOLDOWN_LIMIT

                    for layer in tuning_params.keys():
                        global_At_A[layer].zero_()
                        global_AtA_w[layer].zero_()
                    
                    for n,p in vit_model.named_parameters():
                        if 'l_ff' in n:
                            with torch.no_grad():
                                n_updated = 'model.' + n
                                p.copy_(torch.ones(p.shape).to(cfg.device))
                        if 'l_k' in n:
                            with torch.no_grad():
                                n_updated = 'model.' + n
                                p.copy_(torch.ones(p.shape).to(cfg.device))
                        if 'l_v' in n:
                            with torch.no_grad():                    
                                n_updated = 'model.' + n
                                p.copy_(torch.ones(p.shape).to(cfg.device) )

                    online_gauss.reset()
                    optimizer = optim.Adam(vit_model.parameters(), lr=0.001)
                    count_save += 1
                    batch_idx = 0

            # Tuning classification head for ViT.
            train_stats = train_one_batch(model=vit_model, criterion=criterion,
                                          input=input, target=target, optimizer=optimizer,
                                          device=cfg.device, max_norm=1.0,
                                          set_training_mode=True, task_id=count_save, class_mask=new_classes )
            
            # Prepare head to compute closed-form solution.
            with torch.no_grad():
                net_g.model.head.weight.copy_(vit_model.head.weight)
                net_g.model.head.bias.copy_(vit_model.head.bias)
                backbone.head.weight.copy_(vit_model.head.weight)
                backbone.head.bias.copy_(vit_model.head.bias)

            for n, p in net_g.named_parameters():
                p.requires_grad = False

            # Compute closed-form solution.
            global_updates = tangent_tuning_a_batch(batch_idx, input, target,
                                                        global_At_A, global_AtA_w, AtA_ridge,
                                                        net_g, jacobian_fn,
                                                        vit_model, tuning_params, backbone,
                                                        eta = 0.03, lambda_=1)
            
            # Compute global update accuracies on the current batch.
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

            with torch.no_grad():
                out = vit_model((input))["logits"]
                if new_classes is not None:
                    mask = new_classes
                    not_mask = np.setdiff1d(np.arange(cfg.dtask.nb_classes), mask)
                    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(cfg.device)
                    out = out.index_fill(dim=1, index=not_mask, value=float('-inf'))
                    
                acc1, acc5 = accuracy(out, target, topk=(1,5))
            log(f"GLOBAL UPDATES ON BATCH {batch_idx} is: {acc1}")

            if (cfg.run_label == "VTAB5T-large") and (batch_idx > 2000):
                pass
            else:
                with torch.no_grad():
                    features = vit_model(input)['pre_logits'].cpu()
                
                online_gauss.update(features, batch_idx)
                
            previous_global_update = global_updates
            batch_idx += 1

    # Last save after final batch of last task
    checkpoint_path = os.path.join(cfg.dtask.output_dir, f'ia3_task_{count_save}')
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    torch.save(global_At_A, checkpoint_path + '/global_At_A.pth')
    torch.save(global_AtA_w, checkpoint_path + '/global_AtA_w.pth')
    torch.save(online_gauss.means, checkpoint_path + '/mean.pth')
    torch.save(online_gauss.vars, checkpoint_path + '/vars.pth')
    torch.save(online_gauss.counts, checkpoint_path + '/counts.pth')
    torch.save(global_updates, checkpoint_path + '/global_updates.pth')
    torch.save(vit_model.head.weight, checkpoint_path + '/head_weight.pth')
    torch.save(vit_model.head.bias, checkpoint_path + '/head_bias.pth')
    torch.save(mahal_history, checkpoint_path + '/mahal_history.pth')
    class_mask[count_save] = list(within_seen_classes)
    torch.save(class_mask, cfg.dtask.output_dir + '/class_mask.pth')

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