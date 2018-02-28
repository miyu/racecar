import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import math
from InternalMotionModel import InternalOdometryMotionModel, InternalKinematicMotionModel, InternalLearnedMotionModel, InternalLearnedKinematicMotionModel


def plot_init(n, title):
    plt.figure(n)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)


def plot_draw_particles(particles, directional):
    xs, ys, thetas = particles[:,0], particles[:,1], particles[:,2]
    if not directional:
        plt.scatter(xs, ys)
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


def draw_odometry_plot(n, control, noise_params, axis, iters):
    pose = np.array([0,0,0], dtype=np.float64)
    particles = np.tile(pose, (1000, 1))

    plot_init(n, "Odometry")
    plot_draw_particles(particles, False)

    model = InternalOdometryMotionModel(particles, [0, 0, 0], noise_params)

    for i in range(iters):
        pose += np.array(control)
        model.update(pose)
        plot_draw_particles(particles, iters != 1)

    plot_show(axis)


def draw_kinematic_plot(n, control, noise_params, axis):
    initial_pose = np.array([0,0,0], dtype=np.float64)
    particles = np.tile(initial_pose, (1000, 1))

    plot_init(n, "Kinematic")
    plot_draw_particles(particles, False)

    model = InternalKinematicMotionModel(particles, np.array(noise_params))
    model.update(control)

    plot_draw_particles(particles, False)
    plot_show(axis)


def draw_learned_plot(n, control, axis):
    initial_pose = np.array([0,0,0], dtype=np.float64)
    particles = np.tile(initial_pose, (1000, 1))

    import torch
    model = InternalLearnedMotionModel(particles, torch.load("../../lab3/src/tanh50k.pt"))

    plot_init(n, "Learned")
    plot_draw_particles(particles, False)
    for t in range(10):
        model.update(control)
        plot_draw_particles(particles, False)

    plot_show(axis)


def draw_learned_kinematic_plot(n, control, noise_params, axis):
    initial_pose = np.array([0,0,0], dtype=np.float64)
    particles = np.tile(initial_pose, (1000, 1))

    import torch
    kinematic_motion_model = InternalKinematicMotionModel(particles, np.array(noise_params))
    learned_kinematic_motion_model = InternalLearnedKinematicMotionModel(particles, kinematic_motion_model, torch.load("../../lab1/src/KinematicDropout0.13Itr5k.pt"))

    plot_init(n, "Learned Kinematic")
    plot_draw_particles(particles, False)
    for t in range(100):
        learned_kinematic_motion_model.update([control[0], control[1], control[2]])
        plot_draw_particles(particles, False)

    plot_show(axis)


if __name__=="__main__":
    # kinematic model, noise in speed, no noise in steering
    #draw_kinematic_plot(100, [1, 0, 0.27], [[0, 0.05], [0, 0.0001]], [-0.05, 0.35, -0.10, 0.10])

    # kinematic model, no noise in speed, noise in steering
    #draw_kinematic_plot(110), [1, 0, 0.27], [[0, 0.0001], [0, 0.1]], [-0.05, 0.35, -0.10, 0.10])

    # kinematic model, noise in speed, noise in steering
    #draw_kinematic_plot(120, [1, 0, 0.27], [[0, 0.05], [0, 0.1]], [-0.05, 0.35, -0.10, 0.10])

    # odometry model, noise in speed, no noise in steering
    #draw_odometry_plot(200, [0.27, 0, 0], [[0, 0.01], [0, 0.00001], [0, 0.00001]], [-0.05, 0.35, -0.10, 0.10], 1)

    # odometry model, no noise in speed, noise in steering
    #draw_odometry_plot(200, [0.27, 0, 0], [[0, 0.00001], [0, 0.00001], [0, 0.07]], [-0.05, 0.35, -0.10, 0.10], 1)

    # odometry model, noise in speed, noise in steering
    #draw_odometry_plot(200, [0.27, 0, 0], [[0, 0.01], [0, 0.00001], [0, 0.07]], [-0.05, 0.35, -0.10, 0.10], 1)

    #draw_odometry_plot(200, [0.05, 0, 0], [[0, 0.0001], [0, 0.0001], [0, 0.1]], [-0.05, 0.35, -0.5, 0.5], 5)
    #draw_odometry_plot(200, [0.05, 0.05, 0.2], [[0, 0.01], [0, 0.01], [0, 0.1]], [-0.05, 0.35, -0.2, 0.8], 5)
    #draw_odometry_plot(200, [0.1, 0.01, 0.1], [[0, 0.005], [0, 0.001], [0, 0.05]], 10)

    #draw_learned_plot(200, [1, 0., 0.27], [-0.05, 0.35, -0.10, 0.10])
    draw_learned_kinematic_plot(100, [1, 0, 0.27], [[0, 0.05], [0, 0.1]], [-0.5, 35, -30, 30.0])
