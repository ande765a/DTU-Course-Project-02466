#!/bin/sh
#BSUB -q gpuv100
#BSUB -J asr-model
#BSUB -n 16
#BSUB -R "span[block=1]"
#BSUB -gpu "num=4:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -u andersbthuesen@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o logs/exp1-100-%J.out
#BSUB -e logs/exp1-100-%J.err

nvidia-smi
module load cuda/10.2

PATH=~/miniconda3/bin:$PATH

./src/train.py \
  --data-path /work3/s183926/data/librispeech \
  --dataset train-clean-360 \
  --batch-size 128 \
  --num-epochs 40 \
  --model ResNet \
  --parallel \
  --log-dir /work3/s183926/runs/resnet-100_real \
  --save models/resnet_100-real_40-epochs.pt
