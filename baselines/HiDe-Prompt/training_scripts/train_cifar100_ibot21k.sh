#!/bin/bash

for seed in 42 40 44
do
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        cifar100_hideprompt_5e \
        --model vit_base_patch16_224_21k_ibot \
        --original_model vit_base_patch16_224_21k_ibot \
        --batch-size 24 \
        --epochs 20 \
        --data-path ./datasets \
        --lr 0.0005 \
        --ca_lr 0.005 \
        --crct_epochs 30 \
        --seed 42 \
        --train_inference_task_only \
        --output_dir ./output/cifar100_ibot21k_multi_centroid_mlp_2_seed &> output_cifar_100_hide_tii_t1.txt
done

#[Average accuracy till task10] Acc@1: 84.3100 Acc@5: 97.7900 Loss: 0.5217 Forgetting: 6.6222 Backward: -6.6222

for seed in 42 40 44
do
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        cifar100_hideprompt_5e \
        --model vit_base_patch16_224_21k_ibot \
        --original_model vit_base_patch16_224_21k_ibot \
        --batch-size 24 \
        --epochs 20 \
        --data-path ./datasets \
        --ca_lr 0.005 \
        --crct_epochs 30 \
        --seed 42 \
	--prompt_momentum 0.1 \
	--reg 0.1 \
	--length 5 \
        --trained_original_model ./output/cifar100_ibot21k_multi_centroid_mlp_2_seed \
        --larger_prompt_lr \
	--output_dir ./output/cifar100_ibot21k_pe_seed_t3 &> output_cifar_100_hideprompt_5e_t3.txt
done


