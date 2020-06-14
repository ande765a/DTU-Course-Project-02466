#!/bin/env python
import sys
import os
import argparse
import json
import sys
import numpy as np
import torch
import soundfile
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from torchaudio.transforms import Resample

os.chdir("flowtron")

sys.path.insert(0, ".")
from flowtron import Flowtron
from torch.utils.data import DataLoader
from data import Data
from train import update_params

sys.path.insert(0, "tacotron2")
sys.path.insert(0, "tacotron2/waveglow")
from glow import WaveGlow
from scipy.io.wavfile import write


def chunks(lst, n):
  """Yield successive n-sized chunks from lst."""
  for i in range(0, len(lst), n):
    yield lst[i:i + n]


seed = 1234
sigma = 0.5
gate_threshold = 0.5
n_frames = 400 * 4
flowtron_speaker_id = 0
params = []

target_sample_rate = 16000

waveglow_path = "models/waveglow_256channels_universal_v4.pt"
flowtron_path = "models/flowtron_ljs.pt"
config_path = "config.json"

chunk_size = 1

with open(config_path) as f:
  data = f.read()

config = json.loads(data)
update_params(config, params)

data_config = config["data_config"]
model_config = config["model_config"]

samplerate = data_config["sampling_rate"]
hop_length = data_config["hop_length"]

# Load seeds
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Resample function
resample = Resample(orig_freq=samplerate, new_freq=target_sample_rate)

# load waveglow
waveglow = torch.load(waveglow_path)['model'].cuda().eval()
waveglow.cuda().half()
for k in waveglow.convinv:
  k.float()
waveglow.eval()

# load flowtron
model = Flowtron(**model_config).cuda()
state_dict = torch.load(flowtron_path, map_location='cpu')['state_dict']
model.load_state_dict(state_dict)
model.eval()
print("Loaded checkpoint '{}')".format(flowtron_path))

ignore_keys = ['training_files', 'validation_files']
trainset = Data(
    data_config['training_files'],
    **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
  _, path, output_path = sys.argv

  for speaker_id in tqdm(os.listdir(path)):
    speaker_path = os.path.join(path, speaker_id)

    for chapter_id in os.listdir(speaker_path):
      chapter_path = os.path.join(speaker_path, chapter_id)
      transcript_filename = f"{speaker_id}-{chapter_id}.trans.txt"
      transcript_path = os.path.join(chapter_path, transcript_filename)
      audio_output_dir = os.path.join(output_path, speaker_id, chapter_id)

      # Create output directory
      if not os.path.isdir(audio_output_dir):
        os.makedirs(audio_output_dir)
        os.chmod(audio_output_dir, 0o775)

      transcript_output_path = os.path.join(audio_output_dir,
                                            transcript_filename)

      shutil.copy(transcript_path, transcript_output_path)

      with open(transcript_path, "r") as file:
        for lines in chunks(file.readlines(), chunk_size):
          batch_size = len(lines)
          audio_names, texts = zip(*[line.split(" ", 1) for line in lines])
          texts = [text.lower() for text in texts]
          audio_filenames = [f"{audio_name}.flac" for audio_name in audio_names]
          audio_output_paths = [
              os.path.join(audio_output_dir, audio_filename)
              for audio_filename in audio_filenames
          ]

          if os.path.exists(audio_output_paths[0]):
            continue

          speaker_vecs = trainset.get_speaker_id(flowtron_speaker_id)
          speaker_vecs = speaker_vecs.repeat(batch_size, 1)
          speaker_vecs = speaker_vecs.cuda()

          text_lengths = torch.tensor([len(text) for text in texts])

          max_text_length = torch.max(text_lengths)

          encoded_texts = [trainset.get_text(text) for text in texts]
          encoded_text_lengths = torch.tensor(
              [text.size(0) for text in encoded_texts])

          max_encoded_text_length = torch.max(encoded_text_lengths)

          padded_texts = torch.LongTensor(batch_size, max_encoded_text_length)
          padded_texts.zero_()

          for i, encoded_text in enumerate(encoded_texts):
            padded_texts[i, :encoded_text.size(0)] = encoded_text

          padded_texts = padded_texts.cuda()

          frames = max_text_length * 6

          with torch.no_grad():
            residual = torch.cuda.FloatTensor(batch_size, 80,
                                              frames).normal_() * sigma

            mels, attentions, masks = model.infer(
                residual,
                speaker_vecs.T,
                padded_texts,
                gate_threshold=gate_threshold)

          audio = waveglow.infer(mels.half(), sigma=0.8).float()
          audio = resample(audio)

          audio_max, _ = audio.abs().max(dim=1, keepdim=True)
          audio = audio / audio_max
          audio = audio.cpu().numpy()

          for i, wav in enumerate(audio):
            resampled_hop_length = int(samplerate / target_sample_rate *
                                       hop_length)
            start_index = masks[i][0] * resampled_hop_length
            stop_index = masks[i][1] * resampled_hop_length
            wav = wav[start_index:stop_index]

            soundfile.write(
                audio_output_paths[i], wav, target_sample_rate, format="flac")
            print(audio_output_paths[i])