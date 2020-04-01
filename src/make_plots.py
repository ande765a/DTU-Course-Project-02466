import numpy as np
import matplotlib.pyplot as plt

loss_history = np.load("loss_history.npy")
plt.plot(loss_history)
plt.savefig("loss_history.png")