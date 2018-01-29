from ReSample import ReSampler
import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import math
from copy import deepcopy

if __name__=="__main__":
    M_values = [100, 200, 500, 1000, 4000]

    for M in M_values:
        particles = np.array(range(M))
        particles = particles.reshape(particles.shape[0], 1)

        weights = np.random.rand(M)
        weights /= weights.sum()
        weights2 = deepcopy(weights)
        naive = ReSampler(particles, weights)
        naive.resample_naiive()
        plt.hist(particles, bins=np.arange(0, M))
        plt.show()
        #print(np.sum(particles) / M)

        particles = np.array(range(M))
        particles = particles.reshape(particles.shape[0], 1)

        low = ReSampler(particles, weights2)
        low.resample_low_variance()
        plt.hist(particles, bins=np.arange(0, M))
        plt.show()
        #print(np.sum(particles) / M)
