import logging
import torch
from lib.config import cfg
import numpy as np

import matplotlib.pyplot as plt
# import seaborn as sns
import os
from utils.vision import *
from timm.utils import accuracy




def log(message, print_to_console=True, log_level=logging.DEBUG):
    if log_level == logging.INFO:
        logging.info(message)
    elif log_level == logging.DEBUG:
        logging.debug(message)
    elif log_level == logging.WARNING:
        logging.warning(message)
    elif log_level == logging.ERROR:
        logging.error(message)
    elif log_level == logging.CRITICAL:
        logging.critical(message)
    else:
        logging.debug(message)

    if print_to_console:
        print(message)


def compute_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))
    return result


class Metrics:
    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


# one-hot the layer index
def one_hot(data, max_value):
    ones = torch.sparse.torch.eye(max_value)
    return ones.index_select(0, data)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lr_linear(epoch):
    lr = cfg.kernels.learning_rate * np.minimum((-epoch) * 1. / (cfg.kernels.epochs) + 1, 1.)
    return max(0, lr)


def tonp(x):
    return x.cpu().detach().numpy()


def plot(samples, filename='samples'):
    try:
        d = 7 if '7x7' in cfg.model else 3
        f, axs = plt.subplots(nrows=5, ncols=5, figsize=(12, 10))
        for x, ax in zip(samples.reshape((-1, d, d)), axs.flat):
            # sns.heatmap(tonp(x), ax=ax, cmap='Greens')
            ax.axis('off')
        f.savefig(os.path.join(cfg.output_dir, filename), dpi=200)
        plt.close(f)
    except Exception as error:
        log('Exception occurred while plotting. Ignoring.', log_level=logging.ERROR)
        log(error, log_level=logging.ERROR)


def plot_reconstructions(data, samples, filename='reconstructions'):
    try:
        d = 7 if '7x7' in cfg.model else 3
        f, axs = plt.subplots(nrows=5, ncols=5, figsize=(15, 7))
        for x, x_rec, ax in zip(data.reshape((-1, d, d)), samples.reshape((-1, d, d)), axs.flat):
            # sns.heatmap(np.concatenate((tonp(x), tonp(x_rec)), 1), ax=ax, cmap='Greens')
            ax.axis('off')
        f.savefig(os.path.join(cfg.output_dir, filename), dpi=200)
        plt.close(f)
    except Exception as error:
        log('Exception occurred while plotting. Ignoring.', log_level=logging.ERROR)
        log(error, log_level=logging.ERROR)


def get_glove_embedding(data):
    vectors = torch.load('./glove_embeddings.pkl')
    return torch.FloatTensor(vectors[str(data)])


def compute_offset(task):
    offset1 = task * cfg.continual.n_class_per_task
    offset2 = (task + 1) * cfg.continual.n_class_per_task
    return int(offset1), int(offset2)


def compute_forgetting(accuracies):
    acc_matrix = []
    num_tasks = len(accuracies)
    for index, acc in enumerate(accuracies):
        acc_matrix.append(np.pad(acc, [(0, num_tasks - (index + 1))], mode='constant', constant_values=101))

    acc_matrix = np.array(acc_matrix)
    forgetness = 0
    for t in range(num_tasks-1):
        forgetness += np.max(acc_matrix[t:num_tasks-1,t] - acc_matrix[-1,t])
    avg_forgetness = forgetness / float(num_tasks-1)
    return avg_forgetness


def evaluate_till_now(data_loader, device, task_id=-1, acc_matrix=None, model_lst=None, gaussian_dicts=None, text_tokens=None, num_models = None):
    with torch.no_grad():
        pred_task_id = []
        for tid in range(task_id + 1):
            knn_likelihoods = []
            for mid in range(num_models):
                likelihood_batch = []
                for inputs, targets in data_loader[tid]["val"]:
                    inputs = inputs.to(cfg.device, non_blocking=True)
                    feature = model_lst[mid]((inputs.to(cfg.device), text_tokens))['image_features']
                    max_likelihood = None
                    for m in gaussian_dicts[mid]:
                        log_likelihood = torch.mean(m.log_prob(feature))

                        if max_likelihood is None:
                            max_likelihood = log_likelihood
                        else:
                            max_likelihood = torch.max(max_likelihood, log_likelihood)

                    likelihood_batch.append(max_likelihood)
                likelihood_batch = torch.stack(likelihood_batch)
                knn_likelihoods.append(likelihood_batch)

            knn_likelihoods = torch.stack(knn_likelihoods)
            pred_task_id.append(torch.argmax(knn_likelihoods, dim=0))
            log(f"Task: {tid}")
            log(pred_task_id[tid])

    # Compute accuracy of matching
    # pred_tid_stats = [(pred_task_id[i]==i).sum().item() for i in range(task_id + 1)]
    # true_tid_stats = [len(pred_task_id[i]) for i in range(task_id + 1)]
    # acc_matching = sum(pred_tid_stats)/sum(true_tid_stats)
    # acc_tid_stats = [pred/true for (pred,true) in zip(pred_tid_stats, true_tid_stats)]
    # log(f"Accuracy of matching: {acc_matching}" )
    # log(f"Accuracy of matching according to each task: {acc_tid_stats}")

    test_stats = evaluate_till_now_with_pred_tid(pred_task_id, model_lst, data_loader, device,task_id=task_id, 
                                                 acc_matrix=acc_matrix, text_tokens=text_tokens)
    
    return test_stats


def evaluate_till_now_with_pred_tid(pred_task_id,  model_lst, data_loader, device, task_id=-1, 
                                    acc_matrix=None, text_tokens=None):
    stat_matrix = np.zeros((3, cfg.continual.n_tasks))
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for tid in range(task_id + 1):
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Test: [Task {}]'.format(tid + 1)

        with torch.no_grad():
            b_id = 0
            for input, target in metric_logger.log_every(data_loader[tid]["val"], 10, header):
                input = input.to(cfg.device, non_blocking=True)
                target = target.to(cfg.device, non_blocking=True)
                pred_tid = int(pred_task_id[tid][b_id].cpu())
                output = model_lst[pred_tid]((input, text_tokens))
                logits = output['logits_per_image']
                loss = criterion(logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))

                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

                b_id += 1
        
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        log('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

        test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        stat_matrix[0, tid] = test_stats['Acc@1']
        stat_matrix[1, tid] = test_stats['Acc@5']
        stat_matrix[2, tid] = test_stats['Loss']

        acc_matrix[tid, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id + 1, avg_stat[0], avg_stat[1], avg_stat[2])

    forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
    backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

    result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    log(result_str)

    return test_stats