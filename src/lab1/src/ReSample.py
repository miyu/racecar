import rospy
import numpy as np

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
    print("Entering lock resample_naiive")
    self.state_lock.acquire()
    print("Entered lock resample_naiive")
    # Use np.random.choice to re-sample
    # YOUR CODE HERE

    assert len(self.particles) == len(self.weights)
    M = len(self.particles)
    self.particle_indices = np.random.choice(M, size=M, replace=True, p=self.weights)
    self.particles = self.particles[self.particle_indices][:]
    self.weights[:][:] = (1.0 / M)

    print("Exiting lock resample_naiive")
    self.state_lock.release()

  def resample_low_variance(self):
    print("Entering lock resample_low_variance")
    self.state_lock.acquire()
    print("Entered lock resample_low_variance")
    # Implement low variance re-sampling
    # YOUR CODE HERE

    print("Exiting lock resample_low_variance")
    self.state_lock.release()
