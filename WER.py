import torch
import torch.nn as nn
resS = "---HHHHH-EEE-LLLLLLLLLL-LLLL-OOOOO-----"
resL = list(resS)

#remove adj characters
resL = [char for i, char in enumerate(resL) if i == 0 or resL[i-1] != char]

#Remove blanks
punc = "-"
resL = [char for char in resL if char not in punc]
print(resL)





