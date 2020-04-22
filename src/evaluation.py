import torch
import torch.nn as nn
import numpy as np
import editdistance


def collapse(text):
  last_char = None
  collapsed_text = ""
  for c in text:
    if c != last_char:
      collapsed_text += c
    last_char = c
  return collapsed_text


def remove_blanks(text):
  return "".join([c for c in text if c != "-"])


def WER(input, target):
  input_words = input.split(" ")
  target_words = target.split(" ")
  return editdistance.eval(input_words, target_words) / np.max(
      [1, len(target_words)])


def CER(input, target):
  return editdistance.eval(list(input), list(target)) / np.max([1, len(target)])


if __name__ == "__main__":
  print(CER("Anders er sej", "Anders er sød"))
  print(CER("hello world", "hello working parents"))
