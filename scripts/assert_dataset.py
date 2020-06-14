#!/bin/env python
import sys
sys.path.insert(0, "../src")

from librispeech import LibriSpeech

data_path = "/work3/s183926/data/librispeech"

real_dataset = LibriSpeech(data_path, "train-clean-360", download=False)
synth_dataset = LibriSpeech(data_path, "train-clean-360-synth", download=False)

assert len(real_dataset) == len(
    synth_dataset), "Length of datasets should be the same"
assert real_dataset[0][2] == synth_dataset[0][
    2], "Text should be the same for same elements"

print("Success!")