import numpy as np
import matplotlib.pyplot as plt


def plot(f, min=0.0, max=1.0):
    x = np.arange(min, max, (max-min)/10)
    y = np.sin(x)
    plt.plot(x, y)
