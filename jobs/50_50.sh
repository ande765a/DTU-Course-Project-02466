#!/bin/sh
#BSUB -q gpuv100
#BSUB -J asr-50_50
#BSUB -n 8
#BSUB -R "span[block=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=32GB]"
#BSUB -u andersbthuesen@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o logs/50_50-%J.out
#BSUB -e logs/50_50-%J.err

module load cuda/10.2 cudnn/v7.6.5.32-prod-cuda-10.2
nvidia-smi

PATH=~/miniconda3/bin:$PATH

./src/train.py \
  --data-path /work3/s183926/data/librispeech \
  --real_dataset train-clean-360 \
  --synth_dataset train-clean-360-synth \
  --split 0.5 \
  --batch-size 32 \
  --num-epochs 100 \
  --model DilatedResNet \
  --num-workers 8 \
  --parallel \
  --log-dir ./runs/50_50_100-epochs \
  --save ./models/50_50_100-epochs.pt \
