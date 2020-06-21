#!/usr/bin/env python3
import torch
import argparse
import torch.nn as nn
import numpy as np
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from models import DilatedResNet
from librispeech import LibriSpeech
from torchaudio.transforms import MFCC
from data import dictionary, waveforms_to_padded_mfccs, encode_utterances, tensor_to_text, pad_collate
from evaluation import collapse, remove_blanks, WER, CER
from tqdm import tqdm

import matplotlib.pyplot as plt


def evaluate(data_path, test_dataset, load, batch_size=16, parallel=False):
  mfcc = MFCC(n_mfcc=64)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  original_model = DilatedResNet(n_classes=len(dictionary) + 1)
  original_model = original_model.to(device)
  model = DataParallel(original_model) if parallel else original_model
  model.load_state_dict(torch.load(load))

  test_dataset = LibriSpeech(data_path, test_dataset, download=True)

  test_dataloader = DataLoader(
      test_dataset, batch_size=batch_size, collate_fn=pad_collate)

  all_CERS = []
  all_WERS = []

  for waveforms, utterances in tqdm(test_dataloader):

    X, X_lengths = waveforms_to_padded_mfccs(waveforms, mfcc)
    y, y_lengths = encode_utterances(utterances)

    X = X.to(device)
    X_lengths = X_lengths.to(device)
    y = y.to(device)

    pred_y = original_model(X)
    pred_y = pred_y.permute(2, 0, 1)
    pred_y_lengths = original_model.forward_shape(X_lengths)

    test_texts_real = utterances
    test_texts_pred = [
        remove_blanks(collapse(tensor_to_text(tensor[:l])))
        for tensor, l in zip(
            pred_y.permute(1, 0, 2).argmax(dim=2), pred_y_lengths)
    ]

    CERS = [
        CER(input, target)
        for input, target in zip(test_texts_real, test_texts_pred)
    ]
    WERS = [
        WER(input, target)
        for input, target in zip(test_texts_real, test_texts_pred)
    ]

    all_CERS += CERS
    all_WERS += WERS

  np.save("CERS.npy", all_CERS)
  np.save("WERS.npy", all_WERS)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="ASR Model Evaluator")
  parser.add_argument(
      "--data-path", type=str, help="Path for data", default="../data")
  parser.add_argument(
      "--test-dataset", type=str, help="Test dataset", default="test-clean")
  parser.add_argument("--batch-size", type=int, help="Batch size", default=16)
  parser.add_argument(
      "--parallel",
      type=bool,
      nargs="?",
      const=True,
      help="Train in parallel",
      default=False)

  parser.add_argument(
      "--load", type=str, help="Model parameters", default="model.pt")

  args = parser.parse_args()

  evaluate(
      data_path=args.data_path,
      test_dataset=args.test_dataset,
      load=args.load,
      batch_size=args.batch_size,
      parallel=args.parallel)
