#!/bin/sh
#BSUB -q gpuv100
#BSUB -J asr-model
#BSUB -n 8
#BSUB -R "span[block=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
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
  --train-dataset train-clean-360 \
  --batch-size 32 \
  --num-epochs 1 \
  --save models/100-real.pt
