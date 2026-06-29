import numpy as np
from easydict import EasyDict as edict



root = edict()
cfg = root

root.prompt_tuning_path = 'config/prompt_tuning.json'

root.run_label = 'merlin'
root.gpu_ids = '0'
root.verbose = False
root.device = None

root.seed = 180

root.model = 'ResNet32'

# Training Dynamics
root.epochs = 2
root.within_batch_update_count = 5
root.n_models = 2
root.learning_rate = 0.001
root.momentum = 0.9
root.weight_decay = 0.0001
root.batch_size_train = 512
root.batch_size_test = 100
root.input_size = 0
root.is_cifar_10 = False
root.is_cifar_100 = False
root.is_mini_imagenet = False

# Placeholders
root.timestamp = 'placeholder'
root.output_dir = 'placeholder'

# For CL
root.continual = edict()
root.continual.task = 'permuted_mnist'
root.continual.shuffle_task = False
root.continual.shuffle_datapoints = False
root.continual.rebuild_dataset = False
root.continual.n_tasks = 5
root.continual.n_class_per_task = 0
root.continual.samples_per_task = 1000
root.continual.validation_samples_per_task = 1000
root.continual.epochs = 10
root.continual.n_finetune_epochs = 10
root.continual.finetune_learning_rate = 0.001
root.continual.learning_rate = 0.001
root.continual.batch_size_train = 128
root.continual.batch_size_test = 128

root.continual.method = edict()
root.continual.method.run_merlin = True
root.continual.method.run_ewc = False
root.continual.method.run_gem = False
root.continual.method.run_icarl = False
root.continual.method.run_single_model = False
root.continual.method.run_gss = False
root.continual.method.run_gss_greedy = False

#For downstream task
root.dtask = edict()
root.dtask.type = 'class_increment'
root.dtask.nb_classes = 100
root.dtask.batch_size = 16
root.dtask.epochs = 1
root.dtask.model = 'vit_base_patch16_224'
root.dtask.input_size = 224
root.dtask.pretrained = True
root.dtask.drop = 0.0
root.dtask.drop_path = 0.0
root.dtask.opt = 'adam'
root.dtask.opt_eps = 1e-8
root.dtask.opt_betas = (0.9, 0.999)
root.dtask.clip_grad = 1.0
root.dtask.momentum = 0.9
root.dtask.weight_decay = 0.0
root.dtask.reinit_optimizer = True
root.dtask.sched = 'constant'
root.dtask.lr = 0.03
root.dtask.lr_noise = None
root.dtask.lr_noise_pct = 0.67
root.dtask.lr_noise_std = 1.0
root.dtask.warmup_lr = 1e-6
root.dtask.min_lr = 1e-5
root.dtask.decay_epochs = 30
root.dtask.warmup_epochs = 5
root.dtask.cooldown_epochs = 10
root.dtask.patience_epochs = 10
root.dtask.decay_rate = 0.1
root.dtask.unscale_lr = True
root.dtask.color_jitter = None
root.dtask.aa = None
root.dtask.smoothing = 0.1
root.dtask.train_interpolation = 'bicubic'
root.dtask.reprob = 0.0
root.dtask.remode = 'pixel'
root.dtask.recount = 1
root.dtask.data_path = './local_datasets/'
root.dtask.dataset = 'Split-CIFAR100'
root.dtask.shuffle = False
root.dtask.output_dir = './output_cifar/'
root.dtask.seed = 42
root.dtask.eval = 'store_true'
root.dtask.num_workers = 4
root.dtask.pin_mem = 'store_true'
root.dtask.no_pin_mem = 'store_false'
root.dtask.train_mask = True
root.dtask.task_inc = False
root.dtask.initializer = 'uniform'
root.dtask.use_prompt_mask = False
root.dtask.batchwise_prompt = True
root.dtask.predefined_key = ''
root.dtask.pull_constraint = True
root.dtask.pull_constraint_coeff = 0.1
root.dtask.global_pool = 'token'
root.dtask.head_type = 'prompt'
root.dtask.freeze = ['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed']
root.dtask.print_freq = 10

#For signature learning
root.signature = edict()
root.signature.cooldown_limit = 50
root.signature.n_clusters = 8
root.signature.split_threshold = 0.2
root.signature.window_t = 20
root.signature.k = 1.2
root.signature.split_warmup = 30
root.signature.split_minpoints = 10

def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not b.__contains__(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.
    """
    import yaml
    with open(filename, 'r') as f:
        # yaml_cfg = edict(yaml.load(f))
        yaml_cfg = edict(yaml.full_load(f))

    _merge_a_into_b(yaml_cfg, root)
