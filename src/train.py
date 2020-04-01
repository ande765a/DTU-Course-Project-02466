#!/usr/bin/env python3
import torch
import argparse
import torch.nn as nn
import numpy as np
import multiprocessing
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torchaudio.datasets import LIBRISPEECH
from models.Basic import Basic
from tqdm import tqdm
from evaluation import WER, CER, collapse, remove_blanks

dictionary = "ABCDEFGHIJKLMNOPQRSTUVWXYZ' "


def text_to_tensor(text, dictionary=dictionary):
  """
  This function will convert a string of text
  to a tensor of character indicies in the given dictionary.
  The indicies will start from 1, as 0 means the blank
  character.
  """
  return torch.tensor([
      dictionary.index(c) + 1 if c in dictionary else 0
      for c in list(text.upper())
  ])


def tensor_to_text(tensor, dictionary=dictionary):
  return "".join(["-" if i == 0 else dictionary[i - 1] for i in tensor])


def pad_collate(datapoints):
  waveforms, _, utterances, *rest = zip(*datapoints)
  batch_size = len(datapoints)
  waveform_lengths = torch.tensor([waveform.shape[1] for waveform in waveforms])
  waveforms = pad_sequence([wave.T for wave in waveforms], batch_first=True)
  utterance_lengths = torch.tensor([len(utterance) for utterance in utterances])

  # We convert our label text to tensor of dictionary indicies
  # and reshape the data to (N, S) where N is batch size
  # and S is max target length. Is needed for CTCLoss:
  # https://pytorch.org/docs/stable/nn.html#torch.nn.CTCLoss
  utterances = torch.cat(
      [text_to_tensor(utterance) for utterance in utterances])

  return batch_size, waveforms, waveform_lengths, utterances, utterance_lengths


def train(data_path="../data",
          train_dataset="dev-clean",
          num_epochs=10,
          batch_size=32,
          parallel=False,
          device_name=None,
          num_workers=multiprocessing.cpu_count()):
  dataset = LIBRISPEECH(data_path, train_dataset, download=True)
  dataloader = DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=True,
      collate_fn=pad_collate,
      num_workers=num_workers)
  device = torch.device(device_name) if device_name else torch.device(
      "cuda" if torch.cuda.is_available() else "cpu")
  model = Basic(n_classes=len(dictionary) + 1).to(device)
  if parallel:
    model = nn.DataParallel(model)

  optimizer = Adam(model.parameters(), lr=1e-3)
  loss_fn = CTCLoss()
  print(f"Using device: {device}")

  train_cer_history = []
  loss_history = []
  for epoch in range(num_epochs):
    try:
      print(f"Training epoch: {epoch+1}")
      tqdm_dataloader = tqdm(dataloader)
      for i, (batch_size, X, X_lengths, y,
              y_lengths) in enumerate(tqdm_dataloader):

        # First we zero our gradients, to make everything work nicely.
        optimizer.zero_grad()

        X = X.permute(0, 2, 1).to(device)
        X_lengths = X_lengths.to(device)
        y = y.to(device)

        # We predict the outputs using our model
        # and reshape the data to size (T, N, C) where
        # T is target length, N is batch size and C is number of classes.
        # In our case that is the length of the dictionary + 1
        # as we also need one more class for the blank character.
        pred_y = model(X)
        pred_y = pred_y.permute(2, 0, 1)
        pred_y_lengths = model.forward_shape(X_lengths).to(device)

        loss = loss_fn(pred_y, y, pred_y_lengths, y_lengths)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if i % 10 == 0:
          tqdm_dataloader.set_description(f"Loss: {loss.item()}")
          #print([remove_blanks(collapse(tensor_to_text(tensor[:l]))) for tensor, l in zip(pred_y.permute(1, 0, 2).argmax(dim = 2), pred_y_lengths)])
    except KeyboardInterrupt:
      print("")
      print("Training interrupted.")
      print("")
      print("Available commands:")
      print("stop \t Stops training")
      print("save \t Saves history and continues")
      print("")
      cmd = input("Command: ")
      if cmd == "stop":
        break
      elif cmd == "save":
        np.save("loss_history", loss_history)
        continue

  np.save("loss_history", loss_history)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="ASR Model Trainer")
  parser.add_argument(
      "--data-path", type=str, help="Path for data", default="../data")
  parser.add_argument(
      "--train-dataset", type=str, help="Dataset name", default="dev-clean")
  parser.add_argument("--device-name", type=str, help="Device name")
  parser.add_argument("--batch-size", type=int, help="Batch size", default=32)
  parser.add_argument(
      "--parallel",
      type=bool,
      nargs="?",
      const=True,
      help="Train in parallel",
      default=False)

  args = parser.parse_args()

  train(
      data_path=args.data_path,
      train_dataset=args.train_dataset,
      device_name=args.device_name,
      batch_size=args.batch_size,
      parallel=args.parallel)
