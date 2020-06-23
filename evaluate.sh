#!/bin/sh

./src/evaluate.py \
  --data-path /work3/s183926/data/librispeech \
  --test-dataset test-clean \
  --parallel \
  --real-data 0.75 \
  --synth \
  --load ./final/75_25_100-epochs.pt
