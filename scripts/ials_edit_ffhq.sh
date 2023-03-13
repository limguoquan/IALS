#!/bin/sh

##Job Script for FYP

#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --job-name=IALS-edit-ffhq
#SBATCH --output=./output/output_%x_%j.out
#SBATCH --error=./output/error_%x_%j.err

CUDA_VISIBLE_DEVICES=0 python ials_edit_ffhq.py \
--seed 0 \
--step 0.1 \
--n_steps 10 \
--dataset ffhq \
--base interfacegan \
--attr1 smiling \
--attr2 young \
--lambda1 0.75 \
--lambda2 0 \
--real_image 1