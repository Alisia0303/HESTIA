from torch import nn
from utils.vision import *
from lib.config import cfg
from timm.utils import accuracy

import torch
import math
import torch.nn.functional as F
import copy
import numpy as np

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        print(self.out_features, in_features)
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()
        self.use_RP=False

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        if not self.use_RP:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            if self.W_rand is not None:
                inn = torch.nn.functional.relu(input @ self.W_rand)
            else:
                inn=input
                #inn=torch.bmm(input[:,0:100].unsqueeze(-1), input[:,0:100].unsqueeze(-2)).flatten(start_dim=1) #interaction terms instead of RP
            out = F.linear(inn,self.weight)

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}
    
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(None)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

class SimpleVitNet(BaseNet):
    def __init__(self):
        super().__init__()

    def update_backbone(self, model):
        self.convnet = get_convnet(model)

    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def forward(self, x):
        x = self.convnet(x)["pre_logits"]
        out = self.fc(x)
        out["pre_logits"] = x
        return out
    
def get_convnet(model=None):
    if model is None:
        return None
        
    model.out_dim=768
    return model.eval()

def setup_RP(network, M):
    network.fc.weight = nn.Parameter(torch.Tensor(network.fc.out_features, M).to(device='cuda'))
    network.fc.reset_parameters()
    network.fc.W_rand=torch.randn(network.fc.in_features, M).to(device='cuda')
    W_rand=copy.deepcopy(network.fc.W_rand)
    return W_rand

def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot

def replace_fc(network, data, label, W_rand, Q, G, mapping_classes=None):
    network = network.eval()
    network.fc.use_RP=True
    network.fc.W_rand=W_rand
    Features_f = []
    label_list = []
    with torch.no_grad():
        #for i, batch in enumerate(loader):
        #    (data,label)=batch
        data=data.cuda()
        label=label.cuda()
        if mapping_classes is not None:
            label = torch.tensor(list(map(lambda x: mapping_classes[int(x.cpu())], label))).to(cfg.device, non_blocking=True)
        
        embedding = network.convnet(data)["pre_logits"]
        Features_f.append(embedding.cpu())
        label_list.append(label.cpu())
    Features_f = torch.cat(Features_f, dim=0)
    label_list = torch.cat(label_list, dim=0)
    
    Y=target2onehot(label_list, cfg.dtask.nb_classes)
    Features_h=torch.nn.functional.relu(Features_f @ network.fc.W_rand.cpu())
    Q=Q+Features_h.T @ Y 
    G=G+Features_h.T @ Features_h
    return Y, Features_h, Q, G

def optimise_ridge_parameter(Features,Y):
    ridges=10.0**np.arange(-8,9)
    num_val_samples=int(Features.shape[0]*0.8)
    losses=[]
    Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
    G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
    for ridge in ridges:
        Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T #better nmerical stability than .inv
        Y_train_pred=Features[num_val_samples::,:]@Wo.T
        losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:]))
    ridge=ridges[np.argmin(np.array(losses))]
    print("Optimal lambda: ", ridge)
    return ridge
