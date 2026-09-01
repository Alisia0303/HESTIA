import logging
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFAR224, iImageNetR,iImageNetA,CUB, objectnet, omnibenchmark, vtab
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
            
    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))






class BatchTaskDataManager:
    """
    Task-free stream DataManager
    - Class-ordered (task by task)
    - Shuffle INSIDE each task
    - Slice into fixed-size batches
    - get_dataset(i) returns EXACTLY one batch
    """

    def __init__(self, base_data_manager, batch_size, shuffle=True):
        self.base_dm = base_data_manager
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.nb_classes = base_data_manager.nb_classes
        self.classes_per_task = 10

        # --------------------------------------------------
        # 1. Use RAW training data (NO transforms)
        # --------------------------------------------------
        self.train_data = base_data_manager._train_data
        self.train_targets = base_data_manager._train_targets

        # Test dataset stays normal
        self.test_dataset = base_data_manager.get_dataset(
            np.arange(self.nb_classes),
            source="test",
            mode="test",
        )

        # --------------------------------------------------
        # 2. Build ordered index stream (task by task)
        # --------------------------------------------------
        ordered_indices = []

        nb_tasks = int(np.ceil(self.nb_classes / self.classes_per_task))

        for task_id in range(nb_tasks):
            cls_start = task_id * self.classes_per_task
            cls_end = min(cls_start + self.classes_per_task, self.nb_classes)

            task_classes = np.arange(cls_start, cls_end)

            task_indices = np.where(
                np.isin(self.train_targets, task_classes)
            )[0]

            if self.shuffle:
                np.random.shuffle(task_indices)

            ordered_indices.append(task_indices)

        # Concatenate → global stream
        self.stream_indices = np.concatenate(ordered_indices)

        # --------------------------------------------------
        # 3. Slice stream into fixed-size batches
        # --------------------------------------------------
        self.batches = [
            self.stream_indices[i : i + batch_size]
            for i in range(0, len(self.stream_indices), batch_size)
        ]

        self._nb_tasks = len(self.batches)

    # --------------------------------------------------
    # API expected by your training loop
    # --------------------------------------------------
    @property
    def nb_tasks(self):
        return self._nb_tasks

    def get_task_size(self, task_id):
        return len(self.batches[task_id])

    def get_dataset(self, task_id=None, source="train", mode="train"):
        if source == "test":
            return self.test_dataset

        assert task_id is not None, "Task id must be provided"

        batch_indices = self.batches[task_id]

        # Create dataset ON DEMAND
        images = self.train_data[batch_indices]
        labels = self.train_targets[batch_indices]

        trsf = transforms.Compose([
            *self.base_dm._train_trsf,
            *self.base_dm._common_trsf,
        ])

        return DummyDataset(images, labels, trsf, self.base_dm.use_path)

class TaskFreeDataset(Dataset):
    def __init__(self, images, labels, task_ids, trsf, use_path):
        self.images = images
        self.labels = labels
        self.task_ids = task_ids
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]
        t = self.task_ids[idx]

        if self.use_path:
            x = Image.open(x).convert("RGB")
        else:
            x = Image.fromarray(x)

        if self.trsf is not None:
            x = self.trsf(x)

        return x, y, t


class TaskFreeDataManager(object):
    """
    Task-Free version of DataManager
    - SAME init signature
    - Batch = learning unit
    - No task mixing
    - Real task_id returned per sample
    """

    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = args
        self.dataset_name = dataset_name
        self.shuffle = shuffle
        self.seed = seed

        self.batch_size = args["batch_size"]
        self.classes_per_task = 5

        self._setup_data(dataset_name, shuffle, seed)
        self._build_stream()
        self._build_test_sets()
    
    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order

        self._class_order = order
        logging.info(self._class_order)

        self._train_targets = _map_new_class_index(self._train_targets, order)
        self._test_targets = _map_new_class_index(self._test_targets, order)

        self.nb_classes = len(order)
        self.nb_real_tasks = int(np.ceil(self.nb_classes / self.classes_per_task))
        print(self.nb_classes, self.nb_real_tasks)

        
        
    def _build_stream(self):
        self.batches = []
        self.batch_task_ids = []

        for task_id in range(self.nb_real_tasks):
            cls_start = task_id * self.classes_per_task
            cls_end = min(cls_start + self.classes_per_task, self.nb_classes)

            # --------------------------------------------------
            # 1. Select ALL samples of this task (class-based)
            # --------------------------------------------------
            task_classes = self._class_order[cls_start:cls_end]
            
            task_indices = np.where(
                np.isin(self._train_targets, task_classes)
            )[0]
            # --------------------------------------------------
            # 2. FULL SHUFFLE → label-mixed batches guaranteed
            # --------------------------------------------------
            if self.shuffle:
                np.random.shuffle(task_indices)

            # --------------------------------------------------
            # 3. Arbitrary slicing → task-free batches
            # --------------------------------------------------
            for start in range(0, len(task_indices), self.batch_size):
                batch_indices = task_indices[start : start + self.batch_size]
                self.batches.append(batch_indices)
                self.batch_task_ids.append(task_id)
        self._nb_tasks = len(self.batches)


    def _build_test_sets(self):
        trsf = transforms.Compose([
            *self._test_trsf,
            *self._common_trsf,
        ])

        self.full_test_dataset = TaskFreeDataset(
            self._test_data,
            self._test_targets,
            task_ids=np.full(len(self._test_targets), -1),
            trsf=trsf,
            use_path=self.use_path,
        )

        self.task_test_datasets = {}

        for task_id in range(self.nb_real_tasks):
            cls_start = task_id * self.classes_per_task
            cls_end = min(cls_start + self.classes_per_task, self.nb_classes)

            task_classes = self._class_order[cls_start:cls_end]
            task_indices = np.where(
                np.isin(self._test_targets, task_classes)
            )[0]
           

            self.task_test_datasets[task_id] = TaskFreeDataset(
                self._test_data[task_indices],
                self._test_targets[task_indices],
                task_ids=np.full(len(task_indices), task_id),
                trsf=trsf,
                use_path=self.use_path,
            )
    
    @property
    def nb_tasks(self):
        return self._nb_tasks

    def get_task_size(self, task):
        return len(self.batches[task])

    def get_dataset(self, task, source="train", mode="train"):
        if source == "test":
            return self.full_test_dataset

        indices = self.batches[task]
        task_id = self.batch_task_ids[task]

        trsf = transforms.Compose([
            *self._train_trsf,
            *self._common_trsf,
        ])

        return TaskFreeDataset(
            self._train_data[indices],
            self._train_targets[indices],
            task_ids=np.full(len(indices), task_id),
            trsf=trsf,
            use_path=self.use_path,
        )

    def get_test_dataset(self, task_id):
        return self.task_test_datasets[task_id]





class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "cifar224":
        return iCIFAR224(args)
    elif name == "imagenetr":
        return iImageNetR(args)
    elif name == "imageneta":
        return iImageNetA()
    elif name == "cub":
        return CUB()
    elif name == "objectnet":
        return objectnet()
    elif name == "omnibenchmark":
        return omnibenchmark()
    elif name == "vtab":
        return vtab()

    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
