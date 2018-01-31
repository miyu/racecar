from ReSample import ReSampler
import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import math
from copy import deepcopy
from scipy.stats import norm

if __name__=="__main__":
    M_values = [4000] # [100, 200, 500, 1000, 4000]

    for M in M_values:
        indices = np.array(range(M))
        particles = indices.reshape(M, 1)

        weight_distribution_1 = norm(M / 4.0, M / 8.0)
        weight_distribution_2 = norm(M * 3.0 / 4.0, M / 6.0)
        weights_1 = np.array(weight_distribution_1.pdf(indices))
        weights_2 = np.array(weight_distribution_2.pdf(indices))
        weights_1 /= weights_1.sum()
        weights_2 /= weights_2.sum()
        weights = weights_1 + weights_2 * 1
        weights /= weights.sum()
        weights2 = deepcopy(weights)
        print(weights.shape, weights2.shape)
        """
        naive = ReSampler(particles, weights)
        naive.resample_naiive()
        plt.hist(particles, bins=np.arange(0, M))
        plt.show()
        #print(np.sum(particles) / M)

        particles = np.array(range(M))
        particles = particles.reshape(particles.shape[0], 1)
        """

        resampler = ReSampler(particles, weights2, None, None)
        resampler.resample_low_variance()
        #resampler.resample_naiive()
        plt.hist(particles, bins=np.arange(0, M, 10))
        print('Variance: ', np.std(particles) ** 2)
        plt.show()

        #while True:
            #print(weights2)
	    #plt.hist(weights2, bins=np.arange(0, 0.001, 0.00002))
            #plt.hist(particles, bins=np.arange(0, M, 10))
            #plt.show()
            #print(np.sum(particles) / M)
            #resampler.resample_low_variance()
            #resampler.resample_naiive()
