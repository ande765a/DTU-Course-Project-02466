#!/bin/sh
#BSUB -q gpuv100
#BSUB -J asr-model-test
#BSUB -n 8
#BSUB -R "span[block=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -W 6:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u s183926@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o training-%J.out
#BSUB -e training_%J.err

nvidia-smi
module load cuda/10.2

./src/train.py \
  --data-path /work3/s183926/data/librispeech \
  --train-dataset train-clean-360 \
  --batch-size 64 \
  --parallel