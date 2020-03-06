import os
from torch.utils.data import Dataset

class LibriSpeech(Dataset):
  def __init__(self, path):
    for speaker_id in os.listdir(path):
      speaker_path = os.path.join(path, speaker_id)
      for chapter_id in os.listdir(speaker_path):
        
        chapter_path = os.path.join(speaker_path, chapter_id)
        for id in os.listdir(chapter_path):
          print(f"{speaker_id}-{chapter_id}-{id}")


  def __getitem__(self, i):
    pass

  def __len__():
    pass



if __name__ == "__main__":
  dataset = LibriSpeech(path="dev-clean")
  print(dataset[0])
