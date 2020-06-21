#!/bin/sh

./src/evaluate.py \
  --data-path /work3/s183926/data/librispeech \
  --test-dataset test-clean \
  --parallel \
  --load ./models/50_0_100-epochs.pt
