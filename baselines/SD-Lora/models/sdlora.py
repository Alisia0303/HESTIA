# import logging
# import numpy as np
# import torch
# from torch import nn, optim
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from tqdm import trange

# import timm
# from backbone.lora import LoRA_ViT_timm

# from utils.inc_net import IncrementalNet
# from models.base import BaseLearner
# from utils.toolkit import tensor2numpy
# from models_clip import load_clip_IA3
# from models_clip.clip_IA3 import build_model
# from lib.signatures import OnlineGaussianKMeans
# num_workers = 8
# CLIP_THRESHOLD = 100.0
# GAUSS_COOLDOWN = 60
# MAX_GRAD_NORM = 1.0
# INNER_STEPS = 2
# class cfgc(object):
#     backbonename = 'ViT-B/16'
#     NCTX = 16
#     CTXINIT = ''
#     CSC = False
#     CLASS_TOKEN_POSITION = 'end'

# class Net(torch.nn.Module):
#     def __init__(self, model, text_tokens):
#         super().__init__()
#         self.model = model
#         self.text_tokens = text_tokens

#     def forward(self, X):
#         out = self.model((X, self.text_tokens))
#         logits = out["logits_per_image"]
#         logits = logits.softmax(dim=1)
#         #target_logits = torch.gather(logits, 1, target.reshape(target.size(0),1).to(cfg.device)).squeeze(1) 
#         return logits


# # Define clip model.

# class Learner(BaseLearner):
#     """
#     Task-Free Continual Learner
#     - Each task = one batch
#     - Fixed CIFAR-100 classifier (100 outputs)
#     """

#     def __init__(self, args):
#         super().__init__(args)

#         # Build model ONCE
#         self._cur_batch = 0
#         self._network = IncrementalNet(args, pretrained=True)
#         self._network.update_fc(100)  
#         self.online_gauss = OnlineGaussianKMeans(n_clusters=5, feature_dim=512, device='cuda:0')
#         self.recent_batch_energy = 0
#         self.current_real_task = 0
#         self.truth_real_task = 0
#         self.create_orthorgonal = False
#         self.optimizer = None
#         self.scheduler = None
#         self.last_real_task = 0
#         clip_cfg = cfgc()
#         backbone_name = clip_cfg.backbonename
#         url = load_clip_IA3._MODELS[backbone_name]
#         model_path = load_clip_IA3._download(url)
#         model = torch.jit.load(model_path, map_location="cpu").eval()
#         self.clip_model = build_model(model.state_dict()).to(self._device)
#         self.cool_down = 60
#         # class_names = [
#         #     "apple", "aquarium_fish", "baby", "bear", "beaver",
#         #     "bed", "bee", "beetle", "bicycle", "bottle",
#         #     "bowl", "boy", "bridge", "bus", "butterfly",
#         #     "camel", "can", "castle", "caterpillar", "cattle",
#         #     "chair", "chimpanzee", "clock", "cloud", "cockroach",
#         #     "couch", "crab", "crocodile", "cup", "dinosaur",
#         #     "dolphin", "elephant", "flatfish", "forest", "fox",
#         #     "girl", "hamster", "house", "kangaroo", "keyboard",
#         #     "lamp", "lawn_mower", "leopard", "lion", "lizard",
#         #     "lobster", "man", "maple_tree", "motorcycle", "mountain",
#         #     "mouse", "mushroom", "oak_tree", "orange", "orchid",
#         #     "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
#         #     "plain", "plate", "poppy", "porcupine", "possum",
#         #     "rabbit", "raccoon", "ray", "road", "rocket",
#         #     "rose", "sea", "seal", "shark", "shrew",
#         #     "skunk", "skyscraper", "snail", "snake", "spider",
#         #     "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
#         #     "tank", "telephone", "television", "tiger", "tractor",
#         #     "train", "trout", "tulip", "turtle", "wardrobe",
#         #     "whale", "willow_tree", "wolf", "woman", "worm"
#         # ]
#         class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#         mapping_class_idx_name = dict()
#         for idx in range(len(class_names)):
#             mapping_class_idx_name[idx] = class_names[idx]
#         all_classes_text = []
#         for i in range(len(class_names)):
#             text_prompts = f"A photo of class {mapping_class_idx_name[i]}" 
#             all_classes_text.append(text_prompts)

#         self.text_tokens = load_clip_IA3.tokenize(all_classes_text).to(self._device)
       


#     def after_task(self):
#         pass
    
#     # ============================================================
#     # Task-Free Incremental Training
#     # ============================================================
#     def incremental_train(self, data_manager):
#         if self._cur_batch == 0:
#             test_dataset = data_manager.get_dataset(None, source="test")
            
#             self.global_test_loader = DataLoader(
#                 test_dataset,
#                 batch_size=256,
#                 shuffle=False,
#                 num_workers=num_workers,
#             )
#             self.task_test_datasets = data_manager.task_test_datasets
#             test_truth_task_dataset = self.task_test_datasets[0]
#             self.test_loader = DataLoader(
#                 test_truth_task_dataset,
#                 batch_size=256,
#                 shuffle=False,
#                 num_workers=num_workers,
#             )


            
        
#         self._cur_batch += 1
#         logging.info(f"[Task-Free] Training task (batch) {self._cur_batch}")
#         train_dataset = data_manager.get_dataset(
#             self._cur_batch - 1,
#             source="train",
#         )

#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=self.args["batch_size"],
#             shuffle=False,
#             num_workers=num_workers,
#         )

#         if len(self._multiple_gpus) > 1:
#             self._network = nn.DataParallel(self._network, self._multiple_gpus)

#         self._train(train_loader, self.test_loader)

#         if len(self._multiple_gpus) > 1:
#             self._network = self._network.module

#     # ============================================================
#     # Backbone Update (LoRA)
#     # ============================================================
#     def update_network(self, index=True):
#         vit = timm.create_model(
#             "vit_base_patch16_224",
#             pretrained=True,
#             num_classes=0,
#         )
#         rank = 10
#         lora_model = LoRA_ViT_timm(
#             vit_model=vit.eval(),
#             r=rank,
#             num_classes=10,
#             index=index,
#             increment=self.args["increment"],
#             filepath=self.args["filepath"],
#             cur_task_index=self._cur_task,
#         )

#         lora_model.out_dim = 768
#         return lora_model

#     # ============================================================
#     # Training Dispatcher
#     # ============================================================
#     def _train(self, train_loader, test_loader):
#         self._network.to(self._device)
#         self._network.train()

#         for inputs, targets, task in train_loader:
#             if inputs.shape[-1] != 224:
#                 inputs = F.interpolate(
#                     inputs,
#                     size=(224, 224),
#                     mode="bilinear",
#                     align_corners=False
#                 )



            
#             inputs, targets = inputs.to(self._device), targets.to(self._device)

#             print("\n" + "=" * 80)
#             print(
#                 f"[BATCH START] "
#                 f"BatchID={self._cur_batch} | "
#                 f"RealTask={int(task[0])} | "
#                 f"PredictedTask={self.current_real_task} | "
#                 f"BatchSize={targets.size(0)}"
#             )
#             print(f"Labels in batch: {targets.tolist()}")

#             # ==================================================
#             # 1. OOD CHECK
#             # ==================================================
#             current_task = int(task[0])
#             is_new_task = self.check_ood(current_task)
#             if current_task != self.truth_real_task:
#                 self.truth_real_task = int(task[0])
#                 test_truth_task_dataset = self.task_test_datasets[int(task[0])]
#                 self.test_loader = DataLoader(
#                     test_truth_task_dataset,
#                     batch_size=256,
#                     shuffle=True,
#                     num_workers=num_workers,
#                 )   
#             # ==================================================
#             # 2. OPTIMIZER / SCHEDULER SETUP
#             # ==================================================
#             if self.optimizer is None:
#                 print("[OPTIMIZER] Initializing optimizer (Task 0)")
#                 self.optimizer = optim.SGD(
#                     self._network.parameters(),
#                     lr=self.args["init_lr"],
#                     momentum=0.9,
#                 )
#                 # self.scheduler = optim.lr_scheduler.MultiStepLR(
#                 #     self.optimizer,
#                 #     milestones=self.args["init_milestones"],
#                 #     gamma=self.args["init_lr_decay"],
#                 # )

#             elif is_new_task:
#                 print(
#                     f"[OPTIMIZER] New real task detected → "
#                     f"Switching LoRA (Task {self.current_real_task})"
#                 )


#                 self._network.backbone = self.update_network(index=False)
#                 self._network.to(self._device)

#                 self.optimizer = optim.SGD(
#                     filter(lambda p: p.requires_grad, self._network.parameters()),
#                     lr=self.optimizer.param_groups[0]["lr"],
#                     momentum=0.9,
#                 )
#                 # self.scheduler = optim.lr_scheduler.MultiStepLR(
#                 #     self.optimizer,
#                 #     milestones=self.args["milestones"],
#                 #     gamma=self.args["lrate_decay"],
#                 # )

#             # ==================================================
#             # 3. BATCH ACCURACY BEFORE TRAIN
#             # ==================================================
#             with torch.no_grad():
#                 if self.current_real_task == 0:
#                     logits = self._network(inputs)["logits"]
#                 else:
#                     logits, _ = self._network(inputs, ortho_loss=True)
#                     logits = logits["logits"]

#                 preds = torch.argmax(logits, dim=1)
#                 acc_before = 100.0 * (preds == targets).sum().item() / targets.size(0)

#             print(
#                 f"[BEFORE TRAIN] "
#                 f"RealTask={self.current_real_task} | "
#                 f"Acc={acc_before:.2f}%"
#             )

#             # ==================================================
#             # 4. TRAINING LOOP
#             # ==================================================
#             for step in trange(
#                 5,
#                 desc=f"TaskFree | RealTask={self.current_real_task}",
#                 leave=False,
#             ):
#                 if self.current_real_task == 0:
#                     logits = self._network(inputs)["logits"]
#                 else:
#                     logits, _ = self._network(inputs, ortho_loss=True)
#                     logits = logits["logits"]
    
#                 loss = F.cross_entropy(logits, targets)


#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#                 tqdm.write(
#                     f"[STEP {step+1}/2] "
#                     f"Loss={loss.item():.4f}"
#                 )

#             # ==================================================
#             # 5. BATCH ACCURACY AFTER TRAIN
#             # ==================================================
#             with torch.no_grad():
#                 if self.current_real_task == 0:
#                     logits = self._network(inputs)["logits"]
#                 else:
#                     logits, _ = self._network(inputs, ortho_loss=True)
#                     logits = logits["logits"]

#                 preds = torch.argmax(logits, dim=1)
#                 acc_after = 100.0 * (preds == targets).sum().item() / targets.size(0)
            
#             current_lr = self.optimizer.param_groups[0]["lr"]
#             print(
#                 f"[AFTER TRAIN] "
#                 f"RealTask={self.current_real_task} | "
#                 f"Acc={acc_after:.2f}% | "
#                 f"ΔAcc={acc_after - acc_before:+.2f}% | "
#                 f"LR={current_lr:.6f}"
#             )

#             # ==================================================
#             # 6. UPDATE GAUSSIAN AFTER TRAIN
#             # ==================================================
#             with torch.no_grad():
#                 feats = self.clip_model((inputs, self.text_tokens))["image_features"]
#                 self.online_gauss.update(feats)
#                 print("[GAUSSIAN] Updated with trained features")

#             # ==================================================
#             # 7. LR STEP IF TASK SWITCH
#             # ==================================================
#             if is_new_task and self.scheduler is not None:
#                 self.scheduler.step()
#                 print("[SCHEDULER] LR stepped due to task change")

#             print("=" * 80 + "\n")
#             return








#     # ============================================================
#     # Initial Training
#     # ============================================================
#     def check_ood(self, task):
#         # Resize images to 224
#         # with torch.no_grad():
#         #     feats = self.clip_model((inputs, self.text_tokens))["image_features"]

#         #     dists = []
#         #     for k in range(self.online_gauss.n_clusters):
#         #         mu = self.online_gauss.means[k]
#         #         cov = self.online_gauss.covs[k]
#         #         cov = cov + 1e-4 * torch.eye(cov.shape[-1], device=cov.device)
#         #         diff = feats - mu
#         #         d = torch.einsum("bi,ij,bj->b", diff, torch.linalg.inv(cov), diff)
#         #         dists.append(d)

#         #     energy = torch.min(torch.stack(dists), dim=0).values.mean().item()
#         #     diff_energy = energy - self.recent_batch_energy
#         #     self.recent_batch_energy = energy

#         #     print(
#         #         f"[OOD] Task={self.current_real_task} "
#         #         f"Energy={energy:.2f} Δ={diff_energy:.2f}"
#         #     )

#         #     if self.cool_down > 0:
#         #         self.cool_down -= 1
#         #         return False

#         if task != self.current_real_task:
#             self.cool_down = GAUSS_COOLDOWN
#             self.current_real_task += 1
#             self.last_real_task = self.current_real_task - 1

#             self.online_gauss = OnlineGaussianKMeans(
#                 n_clusters=5, feature_dim=512, device=self._device
#             )
#             save_lora_name = self.args['filepath']
#             if len(self._multiple_gpus) > 1:
#                 self._network.module.backbone.save_lora_parameters(save_lora_name, self.current_real_task)
#                 self._network.module.save_fc(save_lora_name, self._cur_task)
#             else:
#                 self._network.backbone.save_lora_parameters(save_lora_name, self.current_real_task)
#                 self._network.save_fc(save_lora_name, self._cur_task)

#             print(f"[NEW TASK] RealTask={self.current_real_task}")
#             return True

#         return False


#     def _init_train(self, train_loader, test_loader, optimizer, scheduler):
#         losses, correct, total = 0.0, 0, 0
#         self._network.train()        
#         for _, inputs, targets in train_loader:
#             inputs, targets = inputs.to(self._device), targets.to(self._device)


#             for _ in range(5):
#                 logits = self._network(inputs)["logits"]
#                 loss = F.cross_entropy(logits, targets)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#             scheduler.step()
#         # train_acc = 100.0 * correct / total
        
#         # if epoch % 5 == 0:
#         #     test_acc = self._compute_accuracy(self._network, test_loader)
#         #     info = f"[Task {self._cur_task}] Epoch {epoch+1} | Loss {losses/len(train_loader):.3f} | Train {train_acc:.2f} | Test {test_acc:.2f}"
#         # else:
#         #     info = f"[Task {self._cur_task}] Epoch {epoch+1} | Loss {losses/len(train_loader):.3f} | Train {train_acc:.2f}"

#         # prog_bar.set_description(info)

                


#     # ============================================================
#     # Task-Free Update (LoRA)
#     # ============================================================
#     def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
#         self._network.train()

#         for _, inputs, targets in train_loader:
#             inputs, targets = inputs.to(self._device), targets.to(self._device)

#             for _ in range(2):
#                 logits, ortho_loss = self._network(inputs, ortho_loss=True)
#                 logits = logits["logits"]

#                 # GLOBAL classification loss
#                 loss = F.cross_entropy(logits, targets)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()



#             scheduler.step()

import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

import timm
from backbone.lora import LoRA_ViT_timm
import torch.distributed as dist
import os

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        print(self._known_classes)
        print(self._total_classes)
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        print(self._known_classes)
        print(self._total_classes)
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            # model = nn.parallel.DistributedDataParallel(model, device_ids=[self._device], output_device=self._device, find_unused_parameters=True)
        # if len(self._multiple_gpus) > 1:
        #     self._network = self._network.module

        self._train(self.train_loader, self.test_loader)

        # to test
        # self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def incremental_train_vtab(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum_vtab()
        names = ['eurosat', 'oxford_flowers102', 'oxford_iiit_pet', 'patch_camelyon', 'resisc45']
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset_vtab(np.arange(self._known_classes, self._total_classes), names, self._cur_task, source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True,
                                       num_workers=num_workers)
        test_dataset = data_manager.get_dataset_vtab(np.arange(0, self._total_classes), names, self._cur_task, source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False,
                                      num_workers=num_workers)
        
        self._train(self.train_loader, self.test_loader)


    def update_network(self, index=True):
        # if use VIT-B-16
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)

        # if use DINO
        # model = timm.create_model('vit_base_patch16_224_dino', pretrained=True, num_classes=0)

        # SD-LoRA-RR
        '''
        if self._cur_task >=4 and self._cur_task <8:
            rank = 8 #8
        elif self._cur_task >=8:
            rank = 6 #6
        # elif self._cur_task >=8:
        #     rank = 4
        else:
            rank = 10
        '''
        rank=10
        model = LoRA_ViT_timm(vit_model=model.eval(), r=rank, num_classes=10, index=index, increment= self.args['increment'], filepath=self.args['filepath'], 
        cur_task_index= self._cur_task)
        model.out_dim = 768
        return model

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
                # weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)

        else:
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
            self._network.backbone = self.update_network(index=False)
            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)       
            self._network.to(self._device) 

            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        save_lora_name = self.args['filepath']

        if len(self._multiple_gpus) > 1:
            self._network.module.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            self._network.module.save_fc(save_lora_name, self._cur_task)
        else:
            self._network.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            self._network.save_fc(save_lora_name, self._cur_task)

    def get_optimizer(self):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                momentum=0.9, 
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                # lr=self.init_lr, 
                self.args["lrate"],
                # weight_decay=self.weight_decay
                betas=(0.9, 0.999)
            )
            
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )

        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.args['min_lr'])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler



    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        self._network.train()

        
        for i, (_, inputs, targets) in enumerate(train_loader):

            if inputs.shape[-2:] != (224, 224):
                inputs = F.interpolate(
                    inputs,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False
                )



            correct, total = 0, 0
            for epoch in range(2):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                print(targets)

                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)

                optimizer.step()
                
                lr = optimizer.param_groups[0]["lr"]
                
                print(f"[TRAINING] Task {self._cur_task}, Batch {i},Epoch: {epoch}, Loss: {loss}, lr: {lr}")
            
            with torch.no_grad():
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                batch_train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

                if i == len(train_loader) / 2:
                    test_acc = self._compute_accuracy(self._network, test_loader)
                    print(f"[MID TRAIN] Task {self._cur_task}, Batch {i}, Loss: {loss} ,Batch_accuracy: {batch_train_acc}, Test_accuracy: {test_acc}")
                else:
                    print(f"[AFTER TRAIN] Task {self._cur_task}, Batch {i}, Loss: {loss} ,Batch_accuracy: {batch_train_acc}")
        


    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        for i, (_, inputs, targets) in enumerate(train_loader):
            if inputs.shape[-2:] != (224, 224):
                inputs = F.interpolate(
                    inputs,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False
                )
            correct, total = 0, 0
            for epoch in range(2):
                self._network.train()
                inputs, targets = inputs.to(self._device), targets.to(self._device)


                print(targets)
                # logits = self._network(inputs)["logits"]
                logits, ortho_loss = self._network(inputs, ortho_loss=True)
                logits = logits['logits'] 
                
                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes :], fake_targets
                )
                # print('@@@@@@@@@@@@@@loss2', loss_clf, torch.mean(ortho_loss))

                # loss = loss_clf + 10* torch.mean(ortho_loss)
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)

                optimizer.step()
                print(f"[TRAINING] Task {self._cur_task}, Batch {i},Epoch: {epoch}, Loss: {loss}")

            

            with torch.no_grad():
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits, ortho_loss = self._network(inputs, ortho_loss=True)
                logits = logits['logits'] 
                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes :], fake_targets
                )
                print(loss_clf)
                loss = loss_clf
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                batch_train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
                if i == len(train_loader) / 2:
                    test_acc = self._compute_accuracy(self._network, test_loader)
                    print(f"[MID TRAIN] Task {self._cur_task}, Batch {i}, Loss: {loss} ,Batch_accuracy: {batch_train_acc}, Test_accuracy: {test_acc}")
                else:
                    print(f"[AFTER TRAIN] Task {self._cur_task}, Batch {i}, Loss: {loss} ,Batch_accuracy: {batch_train_acc}")
            
            