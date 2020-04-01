#!/bin/sh

./src/train.py \
  --data-path /work3/s183926/data/librispeech \
  --train-dataset train-clean-360 \
  --batch-size 32 \
  --parallel True