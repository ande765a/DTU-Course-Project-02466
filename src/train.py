#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torchaudio.datasets import LIBRISPEECH
from models.Basic import Basic

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

def train():
  # lets go!
  dataset = LIBRISPEECH("../data", "dev-clean", download=True)
  dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = Basic(n_classes = len(dictionary) + 1).to(device)
  optimizer = Adam(model.parameters(), lr=0.001)
  loss_fn = CTCLoss(zero_infinity=True)

  for i, (X, _, (text,), _, _, _) in enumerate(dataloader):
    # First we zero our gradients, to make everything work nicely.
    optimizer.zero_grad()

    # We convert our label text to tensor of dictionary indicies
    # and reshape the data to (N, S) where N is batch size
    # and S is max target length. Is needed for CTCLoss:
    # https://pytorch.org/docs/stable/nn.html#torch.nn.CTCLoss
    y = text_to_tensor(text).view(1, -1)

    # We predict the outputs using our model
    # and reshape the data to size (T, N, C) where
    # T is target length, N is batch size and C is number of classes.
    # In our case that is the length of the dictionary + 1
    # as we also need one more class for the blank character.
    pred_y = model(X).view(-1, 1, len(dictionary) + 1)

    loss = loss_fn(pred_y, y, (pred_y.shape[0],), (y.shape[1],))
    loss.backward()

    print(f"Loss: {loss.item()}")
    optimizer.step()

if __name__ == "__main__":
  train()
