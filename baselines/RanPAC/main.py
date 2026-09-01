from lib.config import cfg, cfg_from_file
from lib.utils import *
from data.dataset import *
from lib.train import train_and_evaluate
from models.model_register import get_model
from lib.random_projection import SimpleVitNet


import numpy as np
import torch
import random
import argparse
import pprint
import random

def learn_continually():
    log('\nRunning experiments using CLEOPATRA')
    tasks = range(cfg.continual.n_tasks)
    if cfg.continual.shuffle_task:
        tasks = torch.randperm(cfg.continual.n_tasks).tolist()

    if cfg.run_label == "VTAB5T-Sim50":
        data_loader = torch.load(cfg.dtask.data_path + "/dataloader_output.pt", weights_only=False)
        class_mask = np.load(cfg.dtask.data_path + "/class_mask.npy", allow_pickle=True).tolist()
        cfg.dtask.n_components = [70 , 100, 100, 20]
        # In case that our labels aren't sorted 
        mapping_classes = dict()
        for i in range(len(class_mask)):
            for j in range(len(class_mask[i])):
                if i == 0:
                    mapping_classes[class_mask[i][j]] = j
                else:
                    mapping_classes[class_mask[i][j]] = (mapping_classes[class_mask[i-1][-1]] + 1) + j

    else:
        data_loader, class_mask = build_continual_dataloader(batch_size=cfg.dtask.batch_size)
        cfg.dtask.n_components = [cfg.dtask.n_components]*cfg.continual.n_tasks
        if cfg.run_label == 'VTAB5T-custom':
            cfg.dtask.n_components[0] = 100
        mapping_classes = None

        train, test = 0,0
        for i in range(cfg.continual.n_tasks):
            log("Task ID %d " %i)
            log("Length data: train %d, test %d " % (len(data_loader[i]["train"]), len(data_loader[i]["val"])))
            train += len(data_loader[i]["train"])
            test += len(data_loader[i]["val"])
        log("Total length data: train %d, test %d " % (train, test))

    log('class_mask %s ' % class_mask)
    log('Number of classes %d ' % cfg.dtask.nb_classes)

    # Define a ViT model
    model = get_model()
    if cfg.dtask.freeze:
        for n, p in model.named_parameters():
            if ('_lora_' in n) or ('head' in n):
                p.requires_grad = True
                print(n)
            else:
                p.requires_grad = False

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log('number of params: %d' % n_parameters)

    # Define a RP model
    network = SimpleVitNet()
    criterion = torch.nn.CrossEntropyLoss().to(cfg.device)
    acc_matrix = np.zeros((cfg.continual.n_tasks, cfg.continual.n_tasks))
    M = 10000
    Q=torch.zeros(M, cfg.dtask.nb_classes)
    G=torch.zeros(M,M)
    classes_seen_so_far = set()

    train_and_evaluate(tasks, model, criterion, data_loader, cfg.device, class_mask, acc_matrix, 
                       added_units=cfg.dtask.added_units, network=network, M=M, Q=Q, G=G, 
                       mapping_classes=mapping_classes, classes_seen_so_far=classes_seen_so_far)
    
def set_seed(seed):
    cfg.seed = seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

def main():
    parser = argparse.ArgumentParser(description='CLEOPATRA in Continual Learning')
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

    if cfg.continual.method.run_merlin:
        learn_continually()


if __name__ == '__main__':
    main()