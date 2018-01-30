import rospy
import math
import numpy as np
from Debug import print_locks, print_benchmark
import time
from threading import Lock
import random

ALPHA1 = 0.2 # TUNE THESE!!! alpha1 affects fast timestamp so higher means we forget faster
ALPHA2 = 0.02 # TUNE THESE!!! alpha2 affects slow timestamp, so lower means keep more of your history
# alpha values have the same meaning, just apply to different weight_fast vs weight_slow

class ReSampler:
  def __init__(self, particles, weights, state_lock=None, map_free_space=None):
    self.particles = particles
    self.weights = weights
    self.particle_indices = None
    self.step_array = None

    self.weight_fast = 0.0
    self.weight_slow = 0.0
    
    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock

    self.map_free_space = map_free_space

  def monte_carlo_localization(self, weights_unsquashed):
    print_locks("Entering lock monte")
    self.state_lock.acquire()
    print_locks("Entered lock monte")
    start_time = time.time()

    new_particles = np.zeros(self.particles.shape)
    M = len(self.weights)
    w_avg = (1.0 / M) * weights_unsquashed.sum()
    #w_avg = -math.log10(w_avg_)
    self.weight_fast = (1.0 - ALPHA1) * self.weight_fast + ALPHA1 * w_avg
    self.weight_slow = (1.0 - ALPHA2) * self.weight_slow + ALPHA2 * w_avg
    prob_new_particle = max(0.0, 1.0 - self.weight_fast / self.weight_slow)
    print(w_avg, "fast", self.weight_fast, "slow", self.weight_slow, "ratio", self.weight_fast / self.weight_slow, "p", prob_new_particle)

    if prob_new_particle == 0.0:
      self.resample_low_variance(True)
      print_locks("Exiting lock monte (A!)")
      self.state_lock.release()   
      return    

    print("party time!")

    random_kept = np.random.rand(M) > prob_new_particle
    ok_thres = np.percentile(self.weights, 0.1)
    force_kept = self.weights > ok_thres
    kept = np.logical_or(random_kept, force_kept)
    choice = 1.0 - np.array(kept, dtype=float)
    variance_particle_indices = np.random.choice(M, size=M, replace=True, p=self.weights)
    variance_particles = self.particles[variance_particle_indices]
    new_particles_indices = np.random.randint(0, self.map_free_space.shape[0] - 1, M)
    new_particles = np.zeros((M, 3))
    new_particles[:, 0] = self.map_free_space[new_particles_indices, 0]
    new_particles[:, 1] = self.map_free_space[new_particles_indices, 1]
    new_particles[:, 2] = np.random.uniform(0, 2*np.pi, M)

    self.particles[:, :] = new_particles * choice[:, np.newaxis] + variance_particles * (1.0 - choice)[:, np.newaxis]
    self.weights[:] = (1.0 / M)

    """
    for m in range(M):
      if prob_new_particle <= np.random.rand():
        ind = random.randint(0, len(self.map_free_space) - 1)
        new_particles[m][0] = self.map_free_space[ind][0]
        new_particles[m][1] = self.map_free_space[ind][1]
        new_particles[m][2] = np.random.uniform(0, 2*np.pi)
      else:
        particle_index = np.random.choice(M, size=None, replace=True, p=self.weights)
        new_particles[m] = self.particles[particle_index]

    self.particles[:,:] = new_particles
    self.weights[:] = (1.0 / M)
    """
    
    print_locks("Exiting lock monte")
    self.state_lock.release()   
    print_benchmark("resample_monte", start_time, time.time())

  def resample_naiive(self):
    print_locks("Entering lock resample_naiive")
    self.state_lock.acquire()
    print_locks("Entered lock resample_naiive")
    start_time = time.time()
    # Use np.random.choice to re-sample
    # YOUR CODE HERE
    assert len(self.particles) == len(self.weights)
    M = len(self.particles)

    #print("RESAMLING WEIGHTS: ")
    best_weight_indices = [x for x in reversed(sorted(range(M), key=lambda i: self.weights[i]))][0:20]

    self.particle_indices = np.random.choice(M, size=M, replace=True, p=self.weights)
    self.particles[:, :] = self.particles[self.particle_indices, :]
    #print("Post-resmaple particle shape", self.particles.shape)
    #print("====")
    #print("")
    self.weights[:] = (1.0 / M)

    print_locks("Exiting lock resample_naiive")
    print_benchmark("resample_naiive", start_time, time.time())
    self.state_lock.release()

  def resample_low_variance(self, lock_taken=False):
    if not lock_taken:
      print_locks("Entering lock resample_low_variance")
      self.state_lock.acquire()
      print_locks("Entered lock resample_low_variance")

    start_time = time.time()

    M = len(self.particles)
    r = np.random.uniform(0.0, 1.0 / M)
    weights_cdf = np.cumsum(self.weights)
    search_ws = r + np.arange(M, dtype=float) / float(M)
    search_ws, ignore = np.modf(search_ws) #fractional, int
    particle_indices = np.searchsorted(weights_cdf, search_ws)
    self.particles[:, :] = self.particles[particle_indices, :]
    self.weights[:] = 1.0 / M

    # for i in range(M):
    #     ind = numpy.searchsorted(weights_cdf, r)
    #     # sanity, don't understand searchsorted propss
    #     ind = max(min(M - 1, ind), 0)
    #
    #     w = self.weights[ind]
    #     r += 1.0 / M
    #     r = r - math.floor(r)

    # M = len(self.particles)
    # r = np.random.uniform(0.0, 1.0) # Draw random number in the range [0, 1/M]
    # U = r - (1.0 / M)
    # particle_index = 0
    # weight_total = self.weights[0]
    # self.particle_indices = np.zeros(M, dtype=int)
    #
    # for new_particle_index in range(M):
    #     U += (1.0 / M)
    #     while U > weight_total:
    #         particle_index += 1
    #         weight_total += self.weights[particle_index % M]
    #     particle_index = particle_index % M
    #     self.particle_indices[new_particle_index] = particle_index
    #
    # self.particles[:, :] = self.particles[self.particle_indices, :]
    # self.weights[:] = (1.0 / M)

    if not lock_taken:
      print_locks("Exiting lock resample_low_variance")
      self.state_lock.release()
    
    print_benchmark("resample_low_variance", start_time, time.time())
