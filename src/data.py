import torch
from torch.nn.utils.rnn import pad_sequence

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


def waveforms_to_padded_mfccs(waveforms, mfcc):
  mfccs = [mfcc(wave) for wave in waveforms]
  mfcc_lenghts = torch.tensor([mfcc.shape[2] for mfcc in mfccs])
  padded_mfccs = pad_sequence([mfcc.T for mfcc in mfccs],
                              batch_first=True).permute(0, 3, 2, 1)
  return padded_mfccs, mfcc_lenghts


def encode_utterances(utterances):
  encodings = torch.cat([text_to_tensor(utterance) for utterance in utterances])
  encoding_lengths = torch.tensor([len(utterance) for utterance in utterances])
  return encodings, encoding_lengths


def pad_collate(datapoints):
  waveforms, _, utterances, *rest = zip(*datapoints)
  return waveforms, utterances