#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torchaudio.datasets import LIBRISPEECH
from models.Basic import Basic
from tqdm import tqdm
import multiprocessing

dictionary = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "

def text_to_tensor(text, dictionary = dictionary):
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

def pad_collate(datapoints):
  waveforms, sample_rates, utterances, speaker_ids, chapter_ids, utterance_ids = zip(*datapoints)
  batch_size = len(datapoints)
  waveform_lengths = torch.tensor([waveform.shape[1] for waveform in waveforms])
  waveforms = pad_sequence([wave.T for wave in waveforms], batch_first=True)
  utterance_lengths = torch.tensor([len(utterance) for utterance in utterances])

  # We convert our label text to tensor of dictionary indicies
  # and reshape the data to (N, S) where N is batch size
  # and S is max target length. Is needed for CTCLoss:
  # https://pytorch.org/docs/stable/nn.html#torch.nn.CTCLoss
  utterances = pad_sequence([text_to_tensor(utterance) for utterance in utterances], batch_first=True)
  
  return batch_size, waveforms, waveform_lengths, utterances, utterance_lengths

def train(num_epochs=10, batch_size=64, num_workers=multiprocessing.cpu_count()):
  dataset = LIBRISPEECH("../data", "dev-clean", download=True)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate, num_workers=num_workers)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = Basic(n_classes = len(dictionary) + 1).to(device)
  optimizer = Adam(model.parameters(), lr=0.001)
  loss_fn = CTCLoss(zero_infinity=True)

  for epoch in range(num_epochs):
    for i, (batch_size, X, X_lengths, y, y_lengths) in tqdm(enumerate(dataloader)):
      # First we zero our gradients, to make everything work nicely.
      optimizer.zero_grad()

      N, T, C = X.shape
      X = X.view(N, C, T).to(device)
      y = y.to(device)

      # We predict the outputs using our model
      # and reshape the data to size (T, N, C) where
      # T is target length, N is batch size and C is number of classes.
      # In our case that is the length of the dictionary + 1
      # as we also need one more class for the blank character.
      pred_y = model(X)
      pred_y_lengths = model.forward_shape(X_lengths)
      N, C, T = pred_y.shape
      pred_y = pred_y.view(T, N, C)
      
      loss = loss_fn(pred_y, y, pred_y_lengths, y_lengths)
      loss.backward()

      optimizer.step()

if __name__ == "__main__":
  train()
