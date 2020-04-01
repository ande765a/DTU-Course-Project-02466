#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

loss_history = np.load("loss_history.npy")
plt.plot(loss_history[20:])
plt.savefig("loss_history.png")