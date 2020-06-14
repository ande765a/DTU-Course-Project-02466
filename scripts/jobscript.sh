#!/bin/sh
#BSUB -q gpuv100
#BSUB -J synthesize-dataset
#BSUB -n 8
#BSUB -R "span[block=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=32GB]"
#BSUB -u andersbthuesen@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o logs/exp1-100-%J.out
#BSUB -e logs/exp1-100-%J.err

module load cuda/10.2 cudnn/v7.6.5.32-prod-cuda-10.2
nvidia-smi

PATH=~/miniconda3/bin:$PATH

./synthesize_dataset.py /work3/s183926/data/librispeech/LibriSpeech/train-clean-360 /work3/s183926/data/librispeech/LibriSpeech/train-clean-360-synth