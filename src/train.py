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
from models import Basic, ResNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
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
  label_lengths = torch.tensor([len(utterance) for utterance in utterances])

  # We convert our label text to tensor of dictionary indicies
  # and reshape the data to (N, S) where N is batch size
  # and S is max target length. Is needed for CTCLoss:
  # https://pytorch.org/docs/stable/nn.html#torch.nn.CTCLoss
  labels = torch.cat([text_to_tensor(utterance) for utterance in utterances])

  return batch_size, waveforms, waveform_lengths, labels, label_lengths, utterances


def train(data_path="../data",
          train_dataset="dev-clean",
          num_epochs=10,
          batch_size=32,
          parallel=False,
          device_name=None,
          load=None,
          save=None,
          model=None,
          log_dir="runs",
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

  n_classes = len(dictionary) + 1
  original_model = ResNet(n_classes) if model == "ResNet" else Basic(n_classes)
  original_model = original_model.to(device)
  model = nn.DataParallel(original_model) if parallel else original_model

  if load:
    print(f"Loading model parameters from: {load}")
    model.load_state_dict(torch.load(load))

  optimizer = Adam(model.parameters(), lr=1e-3)
  loss_fn = CTCLoss()
  print(f"Using device: {device}")

  writer = SummaryWriter(log_dir)
  train_cer_history = []
  loss_history = []

  n_iter = 0
  for epoch in range(num_epochs):
    print(f"Training epoch: {epoch+1}")
    tqdm_dataloader = tqdm(dataloader)
    for i, data in enumerate(tqdm_dataloader):
      n_iter += 1

      batch_size, X, X_lengths, y, y_lengths, texts = data

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
      pred_y = model.forward(X)
      pred_y = pred_y.permute(2, 0, 1)
      pred_y_lengths = original_model.forward_shape(X_lengths).to(device)

      loss = loss_fn(pred_y, y, pred_y_lengths, y_lengths)
      loss.backward()
      optimizer.step()

      loss_history.append(loss.item())

      writer.add_scalar('Loss/train', loss.item(), n_iter)

      if n_iter % 10 == 0:
        pred_texts = ", ".join([
            remove_blanks(collapse(tensor_to_text(tensor[:l])))
            for tensor, l in zip(
                pred_y.permute(1, 0, 2).argmax(dim=2), pred_y_lengths)
        ])
        real_texts = ", ".join(texts)
        writer.add_text("Predicted", pred_texts, n_iter)
        writer.add_text("Real", real_texts, n_iter)

  if save:
    print(f"Saving model to: {save}")
    torch.save(model.state_dict(), save)

  writer.close()


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
  parser.add_argument(
      "--num-epochs",
      type=int,
      help="Number of epochs to train for",
      default=10)
  parser.add_argument(
      "--load", type=str, help="Load model parameters", default=None)

  parser.add_argument(
      "--save", type=str, help="Save model parameters", default=None)

  parser.add_argument("--model", type=str, help="Model", default=None)
  parser.add_argument(
      "--log-dir", type=str, help="Directory to save logs", default=None)

  args = parser.parse_args()

  train(
      data_path=args.data_path,
      train_dataset=args.train_dataset,
      device_name=args.device_name,
      batch_size=args.batch_size,
      parallel=args.parallel,
      num_epochs=args.num_epochs,
      load=args.load,
      save=args.save,
      log_dir=args.log_dir,
      model=args.model)
