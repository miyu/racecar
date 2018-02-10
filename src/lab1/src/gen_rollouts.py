import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import pylab as p
import math
import itertools
from InternalMotionModel import InternalOdometryMotionModel, InternalKinematicMotionModel

def plot_init(n, title):
    plt.figure(n)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)


def plot_draw_particles(particles, directional):
    xs, ys, thetas = particles[:,0], particles[:,1], particles[:,2]
    if not directional:
        plt.scatter(xs, ys, marker=MarkerStyle('o', 'full'), facecolor='0', alpha=10.0/256, lw=0)
    else:
        dxs = np.cos(thetas) * 0.01
        dys = np.sin(thetas) * 0.01

        ax = plt.gca()
        ax.quiver(xs, ys, dxs, dys, angles='xy', scale_units='xy', scale=1)
        plt.draw()

def plot_show(axis=None):
    if axis:
        plt.axis(axis)
    plt.show()

nparticles = 100
camera_fov_horizontal = 68 * (math.pi / 180.0)
camera_fov_vertical = 41 * (math.pi / 180.0)
camera_downward_tilt_angle = 20 * (math.pi / 180.0)
camera_translate_left_right, camera_translate_down_up, camera_translate_near_far = 0, 0.2, 0

tan_fh2 = math.tan(camera_fov_horizontal / 2)
tan_fv2 = math.tan(camera_fov_vertical / 2)

for steering in np.arange(-0.17, 0.17, 0.01):
    aggregate_particle_states = np.zeros((0, 3), dtype=float)

    # simulate particles 1 meter forward
    particles = np.zeros((nparticles, 3), dtype=float)
    mm = InternalKinematicMotionModel(particles)
    speed, dt = 1.0, 0.005
    for i in range(int(5.0 / (speed * dt))):
        mm.update([speed, steering, dt])
        aggregate_particle_states = np.concatenate((aggregate_particle_states, particles), axis=0)

    # # plot particles in 2D robot-space
    # plot_init(0, "mm steering " + str(steering))
    # plot_draw_particles(aggregate_particle_states, False)
    # plot_show()

    # transform 2D (in meters) to camera-space
    naggregates = aggregate_particle_states.shape[0]
    world_particles = np.zeros((naggregates, 4))
    world_particles[:, 0] = aggregate_particle_states[:, 1] # left / right; recall positive y is left..
    world_particles[:, 1] = np.zeros(aggregate_particle_states.shape[0]) # vertical
    world_particles[:, 2] = aggregate_particle_states[:, 0] # near/far; recall moving forward increased x
    world_particles[:, 3] = 1

    camera_translate = np.array([
        [1, 0, 0, -camera_translate_left_right],
        [0, 1, 0, -camera_translate_down_up],
        [0, 0, 1, -camera_translate_near_far],
        [0, 0, 0, 1]])
    ctilt = math.cos(-camera_downward_tilt_angle)
    stilt = math.sin(-camera_downward_tilt_angle)
    # camera_rotate = np.array([
    #     [ctilt, -stilt, 0, 0],
    #     [stilt, ctilt, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]])
    camera_rotate = np.array([
        [1, 0, 0, 0],
        [0, ctilt, -stilt, 0],
        [0, stilt, ctilt, 0],
        [0, 0, 0, 1]])
    camera_particles = np.matmul(camera_rotate, np.matmul(camera_translate, world_particles.T)).T

    frustum_bound_xs = tan_fh2 * camera_particles[:, 2]
    frustum_bound_ys = tan_fv2 * camera_particles[:, 2]
    clip_space_xs = camera_particles[:, 0] / frustum_bound_xs
    clip_space_ys = camera_particles[:, 1] / frustum_bound_ys
    clip_space_particles = np.zeros((aggregate_particle_states.shape[0], 3), dtype=float)
    clip_space_particles[:, 0] = clip_space_xs
    clip_space_particles[:, 1] = clip_space_ys
    plot_init(0, "3D mm steering " + str(steering))
    plot_draw_particles(clip_space_particles, False)
    #plot_show()
    plot_show([-1, 1, -1, 1])
