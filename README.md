# HESTIA: Task-Free Continual Learning via Order-Invariant Linearized Adaptation and Density-Guided Adapter Routing

Welcome to the official code repository for **"Task-Free Continual Learning via Order-Invariant Linearized Adaptation and Density-Guided Adapter Routing"**, accepted at the **Forty-Second Annual Conference on Uncertainty in Artificial Intelligence (UAI 2026)**.

**Authors:** Hang Thi-Thuy Le<sup>1</sup>, Nam-Quan Nguyen<sup>2</sup>, Lam-Huy Nguyen<sup>2</sup>, Dien Dinh<sup>2</sup>, Minh Hoang<sup>3</sup>, Trong Nghia Hoang<sup>1</sup>

<sup>1</sup>Washington State University, Pullman, Washington, USA
<sup>2</sup>University of Science, VNU-HCM, Ho Chi Minh City, Vietnam
<sup>3</sup>Princeton University, Princeton, New Jersey, USA

[Paper](https://proceedings.mlr.press/v337/le26a.html) | [OpenReview](https://openreview.net/forum?id=Gu542jC262)

## 📜 Abstract

Task-free continual learning (TFCL) requires adapting to non-stationary streams without task boundaries or identifiers, where catastrophic forgetting is amplified by both distribution shifts and order-sensitive batch-streaming optimization. We propose HESTIA, which couples order-invariant linearized adaptation with density-guided adapter routing. Linearizing around a pretrained model yields additive sufficient statistics across batches, producing updates that are invariant to batch order so reducing order-induced forgetting without replay. To handle distribution shifts, HESTIA dynamically expands adapters and associates each with a density-based task key modeling its induced embedding distribution. The same density signals enable lightweight change-point detection during training and principled retrieval at inference by selecting the most compatible adapter via likelihood-based scoring. We further provide a theoretical characterization of retrieval error in terms of density estimation quality and cross-adapter embedding separability. Across benchmarks, HESTIA consistently improves accuracy and achieves lower forgetting than strong TFCL baselines under both standard and realistic streaming scenarios.

## 👀 Overview

HESTIA addresses two complementary sources of catastrophic forgetting in task-free continual learning:

1. **Order-induced forgetting**, arising from batch-order-dependent gradient updates within a locally stationary stream. HESTIA mitigates this via an **order-invariant linearized adaptation** framework built around a linearization of the pretrained model, which yields additive sufficient statistics that accumulate across batches without requiring a replay buffer.
2. **Distribution-shift-induced forgetting**, arising when the underlying data distribution changes over time. HESTIA addresses this via **density-guided adapter routing**: it dynamically expands lightweight adapters and pairs each with a Gaussian-mixture-based task key summarizing its induced embedding distribution. The same density signals drive lightweight change-point detection during training and likelihood-based adapter retrieval at inference.

We also provide a theoretical characterization of retrieval error in terms of density estimation quality and cross-adapter embedding separability, and validate HESTIA on Split CIFAR-10/100, Split ImageNet-R, and VTAB5T against strong TFCL baselines.

## 🛠️ Environment Setup

We recommend using a Python 3.12 virtual environment.

```bash
python3.12 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## 🚀 Running Experiments

To reproduce our main results, run the following commands, one for each benchmark:

```bash
# Split CIFAR-10
python main.py --cfg configs/cifar-10-seed-15.yml

# Split CIFAR-100
python main.py --cfg configs/cifar-100-seed-15.yml

# Split ImageNet-R
python main.py --cfg configs/imgn-r-seed-15.yml

# VTAB5T
python main.py --cfg configs/vtab-large-seed-25.yml
```

> **Note on VTAB5T:** this dataset must be downloaded manually from https://github.com/google-research/task_adaptation and placed into `./local_datasets/vtab_data/` before running the corresponding config.

Each config file specifies the dataset, streaming protocol, backbone, and HESTIA-specific hyperparameters (ridge coefficient, change-point window/tolerance, number of Gaussian mixture components, etc.) used in our experiments.

## 📈 Running Baselines

We provide the codebase to reproduce the baseline methods evaluated in our paper inside the `baselines/` directory. 

### 1. L2P and DualPrompt
The L2P and DualPrompt baselines are implemented via the HiDe-Prompt codebase.

```bash
cd baselines/HiDe-Prompt/

# Split CIFAR-10
python -m main cifar10_l2p --seed 15
python -m main cifar10_dualprompt --seed 15

# Split CIFAR-100
python -m main cifar100_l2p --seed 15
python -m main cifar100_dualprompt --seed 15

# Split ImageNet-R
python -m main imr_l2p --seed 15
python -m main imr_dualprompt --seed 15

# VTAB5T
python -m main vtab5t_large_l2p --seed 15
python -m main vtab5t_large_dualprompt --seed 15
```

### 2. RanPAC

```bash
cd baselines/RanPAC/

# Split CIFAR-10
python -m main --cfg './config/cifar-10-seed-15-ranpac.yml'

# Split CIFAR-100
python -m main --cfg './config/cifar-100-seed-15-ranpac.yml'

# Split ImageNet-R
python -m main --cfg './config/imgn-r-seed-15-ranpac.yml'

# VTAB5T
python -m main --cfg './config/vtab-large-seed-15-ranpac.yml'
```

## 📊 Results

HESTIA consistently achieves state-of-the-art accuracy and the lowest forgetting across all evaluated benchmarks (Split CIFAR-10/100, Split ImageNet-R, VTAB5T) while using fewer trainable parameters than competing Transformer-based TFCL baselines. Please refer to the paper for detailed comparisons, ablations, and theoretical analysis.

## 📝 Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@InProceedings{pmlr-v337-le26a,
  title = 	 {Task-Free Continual Learning via Order-Invariant Linearized Adaptation and Density-Guided Adapter Routing},
  author =       {Le, Hang Thi-Thuy and Nguyen, Nam-Quan and Nguyen, Lam-Huy and Dinh, Dien and Hoang, Minh and Hoang, Trong Nghia},
  booktitle = 	 {Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence},
  pages = 	 {3274--3297},
  year = 	 {2026},
  editor = 	 {Perković, Emilija and Malinsky, Daniel},
  volume = 	 {337},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--21 Aug},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v337/main/assets/le26a/le26a.pdf},
  url = 	 {https://proceedings.mlr.press/v337/le26a.html},
  abstract = 	 {Task-free continual learning (TFCL) aims to adapt models to non-stationary data streams without knowing task boundaries. In TFCL, catastrophic forgetting arises from both the evolving data distributions and the order-sensitive nature of batch-streaming optimization. To address these challenges, we propose a new TFCL framework that mitigates optimization-induced forgetting via order-invariant linearized adaptation during learning and accommodates evolving data distributions via density-guided adapter routing for more accurate and effective adapter retrieval during inference. We also provide a theoretical characterization of retrieval error in terms of density estimation quality and cross-adapter embedding separability. Experiments across multiple benchmarks demonstrate that our proposed method consistently achieves higher accuracy and lower forgetting than strong TFCL baselines under both standard and realistic streaming settings. For reproducibility, our experimental code is available at: https://github.com/Alisia0303/HESTIA.git .}
}

```

## 🙏 Acknowledgement

We thank the authors of prior TFCL and PEFT-based continual learning works whose baselines and evaluation protocols informed our experimental design, including L2P, DualPrompt, RanPAC, and SD-LoRA.
