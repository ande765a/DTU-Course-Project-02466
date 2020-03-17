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

dictionary = "ABCDEFGHIJKLMNOPQRSTUVWXYZ' "

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
  utterances = torch.cat([text_to_tensor(utterance) for utterance in utterances])
  
  return batch_size, waveforms, waveform_lengths, utterances, utterance_lengths

def train(num_epochs=10, batch_size=4, num_workers=multiprocessing.cpu_count()):
  dataset = LIBRISPEECH("../data", "dev-clean", download=True)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate, num_workers=num_workers, pin_memory=True)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = Basic(n_classes = len(dictionary) + 1).to(device)
  optimizer = SGD(model.parameters(), lr=1e-3)
  loss_fn = CTCLoss()
  print(f"Using device: {device}")
  
  
  for epoch in range(num_epochs):
    print(f"Training epoch: {epoch+1}")

    tqdm_dataloader = tqdm(dataloader)
    for i, (batch_size, X, X_lengths, y, y_lengths) in enumerate(tqdm_dataloader):
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

      if i % 10 == 0:
        tqdm_dataloader.set_description(f"Loss: {loss.item()}")

if __name__ == "__main__":
  train()
