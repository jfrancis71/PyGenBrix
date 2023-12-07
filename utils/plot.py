import numpy as np
import matplotlib.pyplot as plt


def plot(f, min=0.0, max=1.0, num_points=10):
    x = np.arange(min, max, (max-min)/num_points)
    y = f(x)
    plt.plot(x, y)
