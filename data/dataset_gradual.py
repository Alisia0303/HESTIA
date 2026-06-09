import random
import torch
import math
import collections
import os
import os.path
import numpy as np
import glob

from torchvision import datasets, transforms
from lib.config import cfg
from torch.utils.data.dataset import Subset, ConcatDataset
from typing import Any, Tuple
from shutil import move, rmtree
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
from data.dataset_utils import read_image_file, read_label_file
from PIL import Image
from collections import defaultdict

class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes


def build_transform(is_train):
    resize_im = True #args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * 224)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(224))
    t.append(transforms.ToTensor())

    return transforms.Compose(t)

def get_dataset(dataset, transform_train, transform_val):
    if dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(cfg.dtask.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100(cfg.dtask.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(cfg.dtask.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(cfg.dtask.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'MNIST':
        dataset_train = MNIST_RGB(cfg.dtask.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(cfg.dtask.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'FashionMNIST':
        dataset_train = FashionMNIST(cfg.dtask.data_path, train=True, download=True, transform=transform_train)
        dataset_val = FashionMNIST(cfg.dtask.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(cfg.dtask.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(cfg.dtask.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'NotMNIST':
        dataset_train = NotMNIST(cfg.dtask.data_path, train=True, download=True, transform=transform_train)
        dataset_val = NotMNIST(cfg.dtask.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'Imagenet-R':
        dataset_train = Imagenet_R(cfg.dtask.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Imagenet_R(cfg.dtask.data_path, train=False, download=True, transform=transform_val).data
    
    elif dataset == 'TinyImagenet':
        dataset_train = TinyImagenet(cfg.dtask.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = TinyImagenet(cfg.dtask.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'VTAB5T':
        dataset_train = VTAB5T(cfg.dtask.data_path, train=True, download=False, transform=transform_train).data
        dataset_val = VTAB5T(cfg.dtask.data_path, train=False, download=False, transform=transform_val).data

    # VTAB 5T large
    elif dataset == 'eurosat':
        dataset_train = EuroSAT(cfg.dtask.data_path, train=True, download=False, transform=transform_train).data
        dataset_val = EuroSAT(cfg.dtask.data_path, train=False, download=False, transform=transform_val).data

    elif dataset == 'oxford_flowers102':
        dataset_train = Flowers(cfg.dtask.data_path, train=True, download=False, transform=transform_train).data
        dataset_val = Flowers(cfg.dtask.data_path, train=False, download=False, transform=transform_val).data

    elif dataset == 'oxford_iiit_pet':
        dataset_train = Pets(cfg.dtask.data_path, train=True, download=False, transform=transform_train).data
        dataset_val = Pets(cfg.dtask.data_path, train=False, download=False, transform=transform_val).data

    elif dataset == 'patch_camelyon':
        dataset_train = PatchCamelyon(cfg.dtask.data_path, train=True, download=False, transform=transform_train).data
        dataset_val = PatchCamelyon(cfg.dtask.data_path, train=False, download=False, transform=transform_val).data

    elif dataset == 'resisc45':
        dataset_train = Resisc(cfg.dtask.data_path, train=True, download=False, transform=transform_train).data
        dataset_val = Resisc(cfg.dtask.data_path, train=False, download=False, transform=transform_val).data
    
    return dataset_train, dataset_val

def split_single_dataset_gradual(dataset_train, dataset_val, batch_size=16, batches_per_phase=15):
    cfg.dtask.nb_classes = len(dataset_val.classes)
    assert cfg.dtask.nb_classes % cfg.continual.n_tasks == 0
    classes_per_task = cfg.dtask.nb_classes // cfg.continual.n_tasks
    batch_size = cfg.dtask.batch_size

    labels = list(range(cfg.dtask.nb_classes))
    if cfg.dtask.shuffle:
        random.shuffle(labels)

    split_datasets = []
    mask = []
    samples_per_phase = batch_size * batches_per_phase

    for t in range(cfg.continual.n_tasks):
        scope = labels[t * classes_per_task:(t + 1) * classes_per_task]
        mask.append(scope)

        # Group train indices by class within this task's scope
        class_to_indices = defaultdict(list)
        for k, target in enumerate(dataset_train.targets):
            if int(target) in scope:
                class_to_indices[int(target)].append(k)
        for cls in scope:
            random.shuffle(class_to_indices[cls])

        class_ptr = {cls: 0 for cls in scope}
        ordered_train_indices = []

        # Phase p: active = scope[0:p+2]  →  p=0: {0,1}, p=1: {0,1,2}, ...
        num_phases = max(1, len(scope) - 1)
        for phase in range(num_phases):
            active = scope[:phase + 2] if len(scope) > 1 else scope
            phase_indices = []

            # Round-robin across active classes until we fill the phase
            while len(phase_indices) < samples_per_phase:
                any_added = False
                for cls in active:
                    if class_ptr[cls] < len(class_to_indices[cls]):
                        phase_indices.append(class_to_indices[cls][class_ptr[cls]])
                        class_ptr[cls] += 1
                        any_added = True
                        if len(phase_indices) >= samples_per_phase:
                            break
                if not any_added:
                    break  # all active classes exhausted

            random.shuffle(phase_indices)
            ordered_train_indices.extend(phase_indices)

        # Drain any remaining samples from all classes (all labels already visible)
        remaining = []
        for cls in scope:
            remaining.extend(class_to_indices[cls][class_ptr[cls]:])
        random.shuffle(remaining)
        ordered_train_indices.extend(remaining)

        test_indices = [h for h, tgt in enumerate(dataset_val.targets) if int(tgt) in scope]
        split_datasets.append([Subset(dataset_train, ordered_train_indices), Subset(dataset_val, test_indices)])

    return split_datasets, mask


def split_single_dataset_task_agnostic(dataset_train, dataset_val):
    split_datasets = list()
    mask = list()
    ex_per_stream_batch_train = math.floor(len(dataset_train) / cfg.continual.n_tasks)
    ex_per_stream_batch_val = math.floor(len(dataset_val) / cfg.continual.n_tasks)

    if (len(dataset_train) % cfg.continual.n_tasks) > 0:
        cfg.continual.n_tasks += 1

    dataset_train_idx = [idx for _, idx in enumerate(range(len(dataset_train)))]
    dataset_val_idx = [idx for _, idx in enumerate(range(len(dataset_val)))]

    for _ in range(cfg.continual.n_tasks):
        train_split_indices = dataset_train_idx[:ex_per_stream_batch_train]
        val_split_indices = dataset_val_idx[:ex_per_stream_batch_val]
        dataset_train_idx = dataset_train_idx[ex_per_stream_batch_train:]
        dataset_val_idx = dataset_val_idx[ex_per_stream_batch_val:]
        subset_train, subset_val = Subset(dataset_train, train_split_indices), Subset(dataset_val, val_split_indices)
        split_datasets.append([subset_train, subset_val])
        mask.append(Subset(dataset_train.targets, train_split_indices))

    return split_datasets, mask

def split_single_dataset_task_agnostic_sort(dataset_train, dataset_val):
    split_datasets = list()
    mask = list()
    ex_per_stream_batch_train = math.floor(len(dataset_train) / cfg.continual.n_tasks)
    ex_per_stream_batch_val = math.floor(len(dataset_val) / cfg.continual.n_tasks)

    if (len(dataset_train) % cfg.continual.n_tasks) > 0:
        cfg.continual.n_tasks += 1

    dataset_train_stat = dict()
    dataset_val_stat = dict()
    dataset_train_idx = list()
    dataset_val_idx = list()

    for idx, val in enumerate(dataset_train.targets):
        if val in dataset_train_stat.keys():
            dataset_train_stat[val].append(idx)
        else:
            dataset_train_stat[val] = [idx]

    for idx, val in enumerate(dataset_val.targets):
        if val in dataset_val_stat.keys():
            dataset_val_stat[val].append(idx)
        else:
            dataset_val_stat[val] = [idx]

    dataset_train_stat = collections.OrderedDict(sorted(dataset_train_stat.items()))
    dataset_val_stat = collections.OrderedDict(sorted(dataset_val_stat.items()))

    for key in dataset_train_stat.keys():
        dataset_train_idx.extend(dataset_train_stat.get(key))

    for key in dataset_val_stat.keys():
        dataset_val_idx.extend(dataset_val_stat.get(key))

    for _ in range(cfg.continual.n_tasks):
        train_split_indices = dataset_train_idx[:ex_per_stream_batch_train]
        val_split_indices = dataset_val_idx[:ex_per_stream_batch_val]
        dataset_train_idx = dataset_train_idx[ex_per_stream_batch_train:]
        dataset_val_idx = dataset_val_idx[ex_per_stream_batch_val:]
        subset_train, subset_val = Subset(dataset_train, train_split_indices), Subset(dataset_val, val_split_indices)
        split_datasets.append([subset_train, subset_val])
        mask.append(Subset(dataset_train.targets, train_split_indices))

    return split_datasets, mask

def split_custom_dataset(datasets):
    all_data_train, all_data_val = list(), list()
    label_offset = 0

    for i in range(len(datasets)):
        dataset_train, dataset_val = datasets[i]['data_train'], datasets[i]['data_test']
        # datasets += [{'data_id':i ,'data_train':dataset_train, 'data_test': dataset_val}]
        num_classes = len(dataset_val.classes)

        def make_target_transform(offset):
            return lambda x: x + offset
        
        offset_transform = make_target_transform(label_offset)
        dataset_train.target_transform = offset_transform
        dataset_val.target_transform = offset_transform

        all_data_train.append(dataset_train)
        all_data_val.append(dataset_val)
        label_offset += num_classes

    all_targets_train, all_targets_val = [], []
    n_classes = 10

    for dataset in all_data_train:
        targets = getattr(dataset, 'targets', None)
        if targets is None:
            raise ValueError("Dataset doesn't expose .targets - need custom extraction")
        
        transformed_targets = [dataset.target_transform(t) for t in targets]
        all_targets_train.extend(transformed_targets)

    for dataset in all_data_val:
        targets = getattr(dataset, 'targets', None)
        if targets is None:
            raise ValueError("Dataset doesn't expose .targets - need custom extraction")
        
        transformed_targets = [dataset.target_transform(t) for t in targets]
        all_targets_val.extend(transformed_targets)

    full_dataset_train, full_dataset_val = ConcatDataset(all_data_train), ConcatDataset(all_data_val)
    class_to_indices_train, class_to_indices_val = defaultdict(list), defaultdict(list)

    for idx, label in enumerate(all_targets_train):
        class_to_indices_train[label].append(idx)
    for idx, label in enumerate(all_targets_val):
        class_to_indices_val[label].append(idx)

    all_classes_train = list(class_to_indices_train.keys())[:cfg.continual.n_tasks*n_classes]
    all_classes_val = list(class_to_indices_val.keys())[:cfg.continual.n_tasks*n_classes]
    all_classes_train.sort()
    all_classes_val.sort()
    tasks_train = [[] for _ in range(cfg.continual.n_tasks)]
    tasks_val = [[] for _ in range(cfg.continual.n_tasks)]

    for i, cls in enumerate(all_classes_train):
        task_id = i // n_classes
        tasks_train[task_id].append(cls)
    for i, cls in enumerate(all_classes_val):
        task_id = i // n_classes
        tasks_val[task_id].append(cls)

    task_datasets = [[] for i in range(len(tasks_train))]
    for task_id, class_ids in enumerate(tasks_train):
        task_indices = []
        for cls in class_ids:
            task_indices.extend(class_to_indices_train[cls])

        task_dataset = Subset(full_dataset_train, task_indices)
        task_datasets[task_id] = [task_dataset]
    for task_id, class_ids in enumerate(tasks_val):
        task_indices = []
        for cls in class_ids:
            task_indices.extend(class_to_indices_val[cls])

        task_dataset = Subset(full_dataset_val, task_indices)
        task_datasets[task_id].extend([task_dataset])
    
    return task_datasets

def build_continual_dataloader(batch_size=1, preprocess=None):
    dataloader = list()
    class_mask = list()
    transform_train = preprocess if preprocess is not None else build_transform(True) 
    transform_val = preprocess if preprocess is not None else build_transform(False)

    if cfg.run_label in ['CIFAR-10', 'CIFAR-100', 'IMAGENET-R', 'TINY-IMAGENET', 'VTAB5T']:
        if cfg.run_label == 'CIFAR-100':
            dataset_train, dataset_val = get_dataset('CIFAR100', transform_train, transform_val)
        elif cfg.run_label == 'CIFAR-10':
            dataset_train, dataset_val = get_dataset('CIFAR10', transform_train, transform_val)
        elif cfg.run_label == 'IMAGENET-R':
            dataset_train, dataset_val = get_dataset('Imagenet-R', transform_train, transform_val)
        elif cfg.run_label == 'TINY-IMAGENET':
            dataset_train, dataset_val = get_dataset('TinyImagenet', transform_train, transform_val)
        elif cfg.run_label == 'VTAB5T':
            dataset_train, dataset_val = get_dataset('VTAB5T', transform_train, transform_val)

        splited_dataset, class_mask = split_single_dataset_gradual(dataset_train, dataset_val)
    
    elif cfg.run_label == '5-dataset':
        dataset_list = ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST']
        nb_classes = 0

    elif cfg.run_label == 'VTAB5T-large':
        dataset_list = ['eurosat', 'resisc45', 'oxford_flowers102', 'oxford_iiit_pet', 'patch_camelyon']
        nb_classes = 0

    elif cfg.run_label == 'VTAB5T-custom':
        dataset_list = ['eurosat', 'oxford_flowers102', 'oxford_iiit_pet', 'resisc45']
        datasets = []
        nb_classes = 0

        for i in range(len(dataset_list)):
            dataset_train, dataset_val = get_dataset(dataset_list[i], transform_train, transform_val)
            datasets += [{'data_id':i ,'data_train':dataset_train, 'data_test': dataset_val}]
            splited_dataset = split_custom_dataset(datasets)

    for i in range(cfg.continual.n_tasks):
        if cfg.run_label in ['CIFAR-10', 'CIFAR-100', 'IMAGENET-R', 'TINY-IMAGENET', "VTAB5T"]:
            dataset_train, dataset_val = splited_dataset[i]

        elif cfg.run_label in ['5-dataset', 'VTAB5T-custom', 'VTAB5T-large']:
            if cfg.run_label == 'VTAB5T-custom':
                dataset_train, dataset_val = splited_dataset[i][0], splited_dataset[i][1]
                if class_mask is not None:
                    class_mask.append([i + nb_classes for i in range(10)])
                    nb_classes += 10

            else:
                dataset_train, dataset_val = get_dataset(dataset_list[i], transform_train, transform_val)
                transform_target = Lambda(target_transform, nb_classes)
    
                if class_mask is not None:
                    class_mask.append([i + nb_classes for i in range(len(dataset_val.classes))])
                    nb_classes += len(dataset_val.classes)
    
                if not cfg.dtask.task_inc:
                    dataset_train.target_transform = transform_target
                    dataset_val.target_transform = transform_target

        # sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_train = torch.utils.data.SequentialSampler(dataset_train)  # was RandomSampler
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=batch_size, #args.batch_size,
            num_workers=4 #args.num_workers,
            # pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=batch_size, #args.batch_size,
                num_workers=4, #args.num_workers,
                # pin_memory=args.pin_mem,
            )
        dataloader.append({'train': data_loader_train, 'val': data_loader_val})

    return dataloader, class_mask

class MNIST_RGB(datasets.MNIST):
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST_RGB, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.train = train  # training set or test set

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class FashionMNIST(MNIST_RGB):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
            and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

class NotMNIST(MNIST_RGB):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/main/data/notMNIST.zip'
        self.filename = 'notMNIST.zip'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()

        if self.train:
            fpath = os.path.join(root, 'notMNIST', 'Train')

        else:
            fpath = os.path.join(root, 'notMNIST', 'Test')


        X, Y = [], []
        folders = os.listdir(fpath)

        for folder in folders:
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    X.append(np.array(Image.open(img_path).convert('RGB')))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    print("File {}/{} is broken".format(folder, ims))
        self.data = np.array(X)
        self.targets = Y

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img)
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class SVHN(datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(SVHN, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.classes = np.unique(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)
    
class Imagenet_R(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
        self.filename = 'imagenet-r.tar'

        self.fpath = os.path.join(root, 'imagenet-r')
        if not os.path.isfile(self.fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'imagenet-r')):
            import tarfile
            tar_ref = tarfile.open(os.path.join(root, self.filename), 'r')
            tar_ref.extractall(root)
            tar_ref.close()
        
        if not os.path.exists(self.fpath + '/train') and not os.path.exists(self.fpath + '/test'):
            self.dataset = datasets.ImageFolder(self.fpath, transform=transform)
            
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            
            train, val = torch.utils.data.random_split(self.dataset, [train_size, val_size])
            train_idx, val_idx = train.indices, val.indices
    
            self.train_file_list = [self.dataset.imgs[i][0] for i in train_idx]
            self.test_file_list = [self.dataset.imgs[i][0] for i in val_idx]

            self.split()
        
        if self.train:
            fpath = self.fpath + '/train'

        else:
            fpath = self.fpath + '/test'

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        train_folder = self.fpath + '/train'
        test_folder = self.fpath + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for c in self.dataset.classes:
            if not os.path.exists(os.path.join(train_folder, c)):
                os.mkdir(os.path.join(os.path.join(train_folder, c)))
            if not os.path.exists(os.path.join(test_folder, c)):
                os.mkdir(os.path.join(os.path.join(test_folder, c)))
        
        for path in self.train_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(train_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)

        for path in self.test_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(test_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)
        
        for c in self.dataset.classes:
            path = os.path.join(self.fpath, c)
            rmtree(path)

class TinyImagenet(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        self.filename = 'tiny-imagenet-200.zip'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)
        
        if not os.path.exists(os.path.join(root, 'tiny-imagenet-200')):
            import zipfile
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(os.path.join(root))
            zip_ref.close()

            self.split()

        if self.train:
            fpath = root + 'tiny-imagenet-200/train'

        else:
            fpath = root + 'tiny-imagenet-200/test'
        
        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        test_folder = self.root + 'tiny-imagenet-200/test'

        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(test_folder)

        val_dict = {}
        with open(self.root + 'tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]
                
        paths = glob.glob(self.root + 'tiny-imagenet-200/val/images/*')
        for path in paths:
            if '\\' in path:
                path = path.replace('\\', '/')
            file = path.split('/')[-1]
            folder = val_dict[file]
            if not os.path.exists(test_folder + '/' + folder):
                os.mkdir(test_folder + '/' + folder)
                os.mkdir(test_folder + '/' + folder + '/images')
            
            
        for path in paths:
            if '\\' in path:
                path = path.replace('\\', '/')
            file = path.split('/')[-1]
            folder = val_dict[file]
            src = path
            dst = test_folder + '/' + folder + '/images/' + file
            move(src, dst)
        
        rmtree(self.root + 'tiny-imagenet-200/val')

class VTAB5T(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.fpath = os.path.join(root, 'vtab')
        
        if not os.path.exists(self.fpath + '/train') and not os.path.exists(self.fpath + '/test'):
            self.dataset = datasets.ImageFolder(self.fpath, transform=transform)
            
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            
            train, val = torch.utils.data.random_split(self.dataset, [train_size, val_size])
            train_idx, val_idx = train.indices, val.indices
    
            self.train_file_list = [self.dataset.imgs[i][0] for i in train_idx]
            self.test_file_list = [self.dataset.imgs[i][0] for i in val_idx]

            self.split()
        
        if self.train:
            fpath = self.fpath + '/train'

        else:
            fpath = self.fpath + '/test'

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        train_folder = self.fpath + '/train'
        test_folder = self.fpath + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for c in self.dataset.classes:
            if not os.path.exists(os.path.join(train_folder, c)):
                os.mkdir(os.path.join(os.path.join(train_folder, c)))
            if not os.path.exists(os.path.join(test_folder, c)):
                os.mkdir(os.path.join(os.path.join(test_folder, c)))
        
        for path in self.train_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(train_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)

        for path in self.test_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(test_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)
        
        for c in self.dataset.classes:
            path = os.path.join(self.fpath, c)
            rmtree(path)

class EuroSAT(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train


        if self.train:
            fpath = os.path.join(root, 'eurosat', 'train')

        else:
            fpath = os.path.join(root, 'eurosat', 'test')

        self.data = datasets.ImageFolder(fpath, transform=transform)

class Flowers(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train


        if self.train:
            fpath = os.path.join(root, 'oxford_flowers102', 'train')

        else:
            fpath = os.path.join(root, 'oxford_flowers102', 'test')

        self.data = datasets.ImageFolder(fpath, transform=transform)


class Pets(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train


        if self.train:
            fpath = os.path.join(root, 'oxford_iiit_pet', 'train')

        else:
            fpath = os.path.join(root, 'oxford_iiit_pet', 'test')
        self.data = datasets.ImageFolder(fpath, transform=transform)


class PatchCamelyon(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train


        if self.train:
            fpath = os.path.join(root, 'patch_camelyon', 'train')

        else:
            fpath = os.path.join(root, 'patch_camelyon', 'test')
        self.data = datasets.ImageFolder(fpath, transform=transform)



class Resisc(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train


        if self.train:
            fpath = os.path.join(root, 'resisc45', 'train')

        else:
            fpath = os.path.join(root, 'resisc45', 'test')
        self.data = datasets.ImageFolder(fpath, transform=transform)