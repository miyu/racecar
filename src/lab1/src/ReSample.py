import rospy
import numpy as np
from Debug import print_locks, print_benchmark
import time

class ReSampler:
  def __init__(self, particles, weights, state_lock=None):
    self.particles = particles
    self.weights = weights
    self.particle_indices = None
    self.step_array = None

    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock

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

  def resample_low_variance(self):
    print_locks("Entering lock resample_low_variance")
    self.state_lock.acquire()
    print_locks("Entered lock resample_low_variance")
    start_time = time.time()
    
    M = len(self.particles)
    new_particles = np.array(self.particles.shape, dtype=np.float64)
    r = np.random.uniform(0, 1.0 / M) # Draw random number in the range [0, 1/M]
    U = r - (1.0 / M)
    particle_index = 0
    weight_total = self.weights[0]
    
    for new_particle_index in range(M):
        U += (1.0 / M)
        while U > weight_total:
            particle_index += 1
            weight_total += self.weights[particle_index % M]
        particle_index = particle_index % M
        print(new_particles[new_particle_index][:].shape, self.particles[particle_index][:].shape)
        new_particles[new_particle_index][:] = self.particles[particle_index][:]
        
    self.particles = new_particles

    print_locks("Exiting lock resample_low_variance")
    print_benchmark("resample_low_variance", start_time, time.time())
    self.state_lock.release()
