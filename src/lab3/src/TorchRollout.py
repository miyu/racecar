#!/usr/bin/env python

import time
import sys
import rospy
import rosbag
import numpy as np
import scipy.signal
import utils as Utils
import random

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

import matplotlib.pyplot as plt

# The following are functions meant for debugging and sanity checking your
# model. You should use these and / or design your own testing tools.
# test_model starts at [0,0,0]; you can specify a control to be applied and the
# rollout() function will use that control for N timesteps.
# i.e. a velocity value of 0.7 should drive the car to a positive x value.
def rollout(m, nn_input, N):
    pose = torch.zeros(3).cuda()
    x = []
    y = []
    theta = []
    t = []
    s_values = [0.0, 0.0, 0.0, 0.0]
    print(pose.cpu().numpy())
    for i in range(N):
        out = m(Variable(nn_input))
        pose.add_(out.data)
        # Wrap pi
        if pose[2] > 3.14:
            pose[2] -= 2*np.pi
        if pose[2] < -3.14:
            pose[2] += 2*np.pi
        nn_input[0] = out.data[0]
        nn_input[1] = out.data[1]
        nn_input[2] = out.data[2]
        nn_input[3] = np.sin(pose[2])
        nn_input[4] = np.cos(pose[2])

        s_values[0] += out.data[0]
        x.append(s_values[0])
        s_values[1] += out.data[1]
        y.append(s_values[1])
        s_values[2] += out.data[2]
        theta.append(s_values[2])
        s_values[3] += 0.1
        t.append(s_values[3])

        print(pose.cpu().numpy())
        if i != 0 and i % 20 == 0:
            plt.plot(t, x)
            plt.show()
            plt.plot(t, y)
            plt.show()
            plt.plot(t, theta)
            plt.show()


def test_model(m, N, dt = 0.1):
    cos, v, st = 4, 5, 6
    s = 8
    print("Nothing")
    nn_input = torch.zeros(s).cuda()
    nn_input[cos] = 1.0
    nn_input[7] = dt
    rollout(m, nn_input, N)

    print("Forward")
    nn_input = torch.zeros(s).cuda()
    nn_input[cos] = 1.0
    nn_input[v] = 0.7 #1.0
    nn_input[7] = dt
    rollout(m, nn_input, N)

if __name__ == '__main__':
    model = torch.load('tanh.pt')
    model = model.cuda()

    test_model(model, 60)
