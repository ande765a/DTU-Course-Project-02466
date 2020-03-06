import torch
import torch.nn as nn
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torchaudio.datasets import LIBRISPEECH
from models.Basic import Basic

def train():
  # lets go!
  dataset = LIBRISPEECH("data", "dev-clean", download=True)
  dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = Basic().to(device)
  optimizer = Adam(model.parameters(), lr=0.001)
  loss_fn = CTCLoss()

  for i, (X, _, y, _, _, _) in enumerate(dataloader):
    optimizer.zero_grad()
    pred_y = model(X)
    loss = loss_fn(pred_y, y, X.shape)
    loss.backward()
    print(f"Loss: {loss.item()}")
    optimizer.step()

    if i % 100 == 0:
      # Evaluate and display accuracy / word error rate / character error rate
      pass

if __name__ == "__main__":
  train()
