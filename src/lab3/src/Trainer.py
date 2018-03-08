#!/usr/bin/env python
import roslib; roslib.load_manifest('lab3')

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


PLOT_FLAG = True

SPEED_TO_ERPM_OFFSET     = 0.0
SPEED_TO_ERPM_GAIN       = 4614.0
STEERING_TO_SERVO_OFFSET = 0.5304
STEERING_TO_SERVO_GAIN   = -1.2135

argv = sys.argv
argv = [argv[0], "motion_data2.bag"]

if len(argv) < 2:
    print('Input a bag file from command line')
    print('Input a bag file from command line')
    print('Input a bag file from command line')
    print('Input a bag file from command line')
    print('Input a bag file from command line')
    sys.exit()

INPUT_SIZE=8
OUTPUT_SIZE=3
DATA_SIZE=6

def load_raw_datas():
    return np.load("/home/nvidia/our_catkin_ws/src/lab3/src/raw_datas.npy")

    bag = rosbag.Bag(argv[1])
    tandt = bag.get_type_and_topic_info()
    t1='/vesc/sensors/core'
    t2='/vesc/sensors/servo_position_command'
    t3='/pf/ta/viz/inferred_pose'
    topics = [t1,t2,t3]
    min_datas = tandt[1][t3][1] # number of t3 messages is less than t1, t2

    raw_datas = np.zeros((min_datas,DATA_SIZE))

    last_servo, last_vel = 0.0, 0.0
    n_servo, n_vel = 0, 0
    idx=0
    # The following for-loop cycles through the bag file and averages all control
    # inputs until an inferred_pose from the particle filter is recieved. We then
    # save that data into a buffer for later processing.
    # You should experiment with additional data streams to see if your model
    # performance improves.
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == t1:
            last_vel   += (msg.state.speed - SPEED_TO_ERPM_OFFSET) / SPEED_TO_ERPM_GAIN
            n_vel += 1
        elif topic == t2:
            last_servo += (msg.data - STEERING_TO_SERVO_OFFSET) / STEERING_TO_SERVO_GAIN
            n_servo += 1
        elif topic == t3 and n_vel > 0 and n_servo > 0:
            timenow = msg.header.stamp
            last_t = timenow.to_sec()
            last_vel /= n_vel
            last_servo /= n_servo
            orientation = Utils.quaternion_to_angle(msg.pose.orientation)
            data = np.array([msg.pose.position.x,
                             msg.pose.position.y,
                             orientation,
                             last_vel,
                             last_servo,
                             last_t])
            raw_datas[idx,:] = data
            last_vel = 0.0
            last_servo = 0.0
            n_vel = 0
            n_servo = 0
            idx = idx+1
            if idx % 1000==0:
                print('.')
    bag.close()

    # Pre-process the data to remove outliers, filter for smoothness, and calculate
    # values not directly measured by sensors

    # Note:
    # Neural networks and other machine learning methods would prefer terms to be
    # equally weighted, or in approximately the same range of values. Here, we can
    # keep the range of values to be between -1 and 1, but any data manipulation we
    # do here from raw values to our model input we will also need to do in our
    # MPPI code.

    # We have collected:
    # raw_datas = [ x, y, theta, v, delta, time]
    # We want to have:
    # x_datas[i,  :] = [x_dot, y_dot, theta_dot, sin(theta), cos(theta), v, delta, dt]
    # y_datas[i-1,:] = [x_dot, y_dot, theta_dot ]

    raw_datas = raw_datas[:idx, :] # Clip to only data found from bag file
    raw_datas = raw_datas[ np.abs(raw_datas[:,3]) < 0.75 ] # discard bad controls
    raw_datas = raw_datas[ np.abs(raw_datas[:,4]) < 0.36 ] # discard bad controls

    np.save("raw_datas.npy", raw_datas)
    return raw_datas

raw_datas = load_raw_datas()
x_datas = np.zeros( (raw_datas.shape[0], INPUT_SIZE) )
y_datas = np.zeros( (raw_datas.shape[0], OUTPUT_SIZE) )

dt = np.diff(raw_datas[:,5])

# TODO
# It is critical we properly handle theta-rollover:
# as -pi < theta < pi, theta_dot can be > pi, so we have to handle those
# cases to keep theta_dot also between -pi and pi

# Information about the variables.
# pose_dot[i, :]  = [x_dot, y_dot, theta_dot]
# x_dot = x_{t} - x_{t-1}
# y_dot = y_{t} - y_{t-1}
# theta_dot = theta_{t} - theta_{t-1}

# x_dot = raw_datas[1, 0] - raw_datas[0, 0]
# y_dot = raw_datas[1, 1] - raw_datas[0, 1]
# theta_dot = raw_datas[1, 2] - raw_datas[0, 2]
# pose_dot = np.array([[x_dot, y_dot, theta_dot]])

#x_datas[1:, 7] = dt

# for i in range(1,len(raw_datas)):
#     x_dot = raw_datas[i, 0] - raw_datas[i-1, 0]
#     y_dot = raw_datas[i, 1] - raw_datas[i-1, 1]
#     theta_dot = raw_datas[i, 2] - raw_datas[i-1,2]
#     pose_dot = np.append(pose_dot,[[x_dot, y_dot, theta_dot]], axis=0)
#
#     #dt = raw_datas[i, 5] - raw_datas[i-1, 5]
#     #x_datas[i, 7] = dt
x_dot = np.diff(raw_datas[:, 0])
y_dot = np.diff(raw_datas[:, 1])
theta_dot = np.diff(raw_datas[:, 2])
pose_dot = np.column_stack((x_dot, y_dot, theta_dot))

gt = pose_dot[:,2] > np.pi
pose_dot[gt,2] = pose_dot[gt,2] - 2*np.pi
lt = pose_dot[:,2] < -np.pi
pose_dot[lt,2] = pose_dot[lt,2] + 2*np.pi


# TODO
# Some raw values from sensors / particle filter may be noisy. It is safe to
# filter the raw values to make them more well behaved. We recommend something
# like a Savitzky-Golay filter. You should confirm visually (by plotting) that
# your chosen smoother works as intended.
# An example of what this may look like is in the homework document.

# raw_datas = [ x, y, theta, v, delta, time]
# x_datas[i,  :] = [x_dot, y_dot, theta_dot, sin(theta), cos(theta), v, delta, dt]
# y_datas[i-1,:] = [x_dot, y_dot, theta_dot ]

window_size = 21
poly_len = 3

x_datas[1:, 0] = scipy.signal.savgol_filter(pose_dot[:,0], window_size, poly_len)
x_datas[1:, 1] = scipy.signal.savgol_filter(pose_dot[:,1], window_size, poly_len)
x_datas[1:, 2] = scipy.signal.savgol_filter(pose_dot[:,2], window_size, poly_len)
x_datas[:, 3] = np.sin(raw_datas[:,2])#scipy.signal.savgol_filter(raw_datas[:,2], window_size, poly_len))
x_datas[:, 4] = np.cos(raw_datas[:,2])#scipy.signal.savgol_filter(raw_datas[:,2], window_size, poly_len))
x_datas[:, 5] = raw_datas[:,3]
x_datas[:, 6] = raw_datas[:,4]
x_datas[1:, 7] = scipy.signal.savgol_filter(dt, window_size, poly_len)

#dt is calculated by the previous for-loop,


#####
# Plot the raw and filtered x_dot
#####
if PLOT_FLAG and False:
    max_ind = 1000
    plt.plot(raw_datas[0:max_ind,5] - raw_datas[0,5], pose_dot[0:max_ind,0] ,    \
            raw_datas[0:max_ind,5] - raw_datas[0,5], x_datas[0:max_ind,0] )
    plt.show()

##############################################
# Is there a better way to get y_datas?
##############################################
end_ind = len(x_datas)
y_datas[:end_ind-1, 0:3] = x_datas[1:,0:3]

# Make Training robust to stasis
num_stasis_pad = 100
x_zeros = np.zeros([num_stasis_pad, INPUT_SIZE])
y_zeros = np.zeros([num_stasis_pad, OUTPUT_SIZE])
pad_pose_thetas = np.random.uniform(0, np.pi * 2, num_stasis_pad)
x_zeros[:, 3] = np.sin(pad_pose_thetas) # cos 0 = 1
x_zeros[:, 4] = np.cos(pad_pose_thetas) # cos 0 = 1
x_zeros[:, 7] = np.clip(np.random.normal(0.25, 0.1, num_stasis_pad), 0.001, 0.5) # dt = 0.1

x_datas = np.append(x_datas, x_zeros, axis=0)
y_datas = np.append(y_datas, y_zeros, axis=0)

# Convince yourself that input/output values are not strange
print("Xdot  ", np.min(x_datas[:,0]), np.max(x_datas[:,0]))
print("Ydot  ", np.min(x_datas[:,1]), np.max(x_datas[:,1]))
print("Tdot  ", np.min(x_datas[:,2]), np.max(x_datas[:,2]))
print("sin   ", np.min(x_datas[:,3]), np.max(x_datas[:,3]))
print("cos   ", np.min(x_datas[:,4]), np.max(x_datas[:,4]))
print("vel   ", np.min(x_datas[:,5]), np.max(x_datas[:,5]))
print("delt  ", np.min(x_datas[:,6]), np.max(x_datas[:,6]))
print("dt    ", np.min(x_datas[:,7]), np.max(x_datas[:,7]))
print()
print("y Xdot", np.min(y_datas[:,0]), np.max(y_datas[:,0]))
print("y Ydot", np.min(y_datas[:,1]), np.max(y_datas[:,1]))
print("y Tdot", np.min(y_datas[:,2]), np.max(y_datas[:,2]))

######### NN stuff
dtype = torch.cuda.FloatTensor
# D_in, H1, H2, H3, D_out = INPUT_SIZE, 32, 32, 32, OUTPUT_SIZE
#D_in, H1, H2, H3, D_out = INPUT_SIZE, 32, 32, 32, OUTPUT_SIZE # 0.000513, with 5e-5
D_in, H1, H2, H3, D_out = INPUT_SIZE, 80, 64, 64, OUTPUT_SIZE # 0.000525, with 5e-5

# Make validation set
num_samples = x_datas.shape[0]
rand_idx = np.random.permutation(num_samples)
x_d = x_datas[rand_idx,:]
y_d = y_datas[rand_idx,:]
split = int(0.9*num_samples) # 90% Training Set, 10% Validation Set
x_tr = x_d[:split]
y_tr = y_d[:split]
x_tt = x_d[split:]
y_tt = y_d[split:]

# TODO
# specify your neural network (or other) model here.
# model = torch

model = nn.Sequential(
    nn.Linear(INPUT_SIZE, H1),
    #nn.Tanh(), #relu
    nn.Dropout(0.13), #works best around 0.1 - 0.13
    # nn.ReLU(),

    nn.Linear(H1, H3),
    nn.ReLU(),

    # pretty terrible results
    # nn.Linear(H1, H2),
    # nn.ReLU(),
    # nn.Linear(H2, H3),
    # nn.ReLU(),

    nn.Linear(H3, OUTPUT_SIZE)
)
model = model.cuda()

#x = Variable(x, requires_grad=True) # Needs Gradient
#y = Variable(y, requires_grad=False) # Target does not need gradient
#x_val = Variable(x_val, requires_grad=False) # No Gradient for test data
#y_val = Variable(y_val, requires_grad=False) # Target does not need gradient

loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 3e-3
opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0) #learning_rate)

filename = 'tanh50k.pt'

def doTraining(model, filename, optimizer, N=10000):
    x = torch.from_numpy(x_tr.astype('float32')).type(dtype)
    y = torch.from_numpy(y_tr.astype('float32')).type(dtype)
    x_val = torch.from_numpy(x_tt.astype('float32')).type(dtype)
    y_val = torch.from_numpy(y_tt.astype('float32')).type(dtype)

    t_list = []
    vloss_list = []
    for t in range(N):
        num_samples = x.shape[0]
        rand_idx = np.random.permutation(num_samples)
        x = x[rand_idx,:]
        y = y[rand_idx,:]

        y_pred = model(Variable(x, requires_grad=True)) # gets to 9.399e-5 w/ rate 1e-3, then 5.88E-05 w/ rate 5e-3, 5e-05 w/ tweaks to dropout.
        loss = loss_fn(y_pred, Variable(y, requires_grad=False))
        if t % 50 == 0:
            val = model(Variable(x_val))
            vloss = loss_fn(val, Variable(y_val, requires_grad=False))

            # t_list.append(t)
            # vloss_list.append(vloss.data[0]/x_val.shape[0])
            # First print is the loss
            # Second print is the validation loss
            print(t, loss.data[0]/x.shape[0], vloss.data[0]/x_val.shape[0])


        # Optimize
        optimizer.zero_grad() # clear out old computed gradients
        loss.backward()       # apply the loss function backprop
        optimizer.step()      # take a gradient step for model's parameters

        # if not PLOT_FLAG and t != 0 and t % 5000 == 0 :
        #     plt.plot(t_list, vloss_list)
        #     plt.show()

    torch.save(model, filename)

#doTraining(model, filename, opt)
model = torch.load(filename)
model = model.cuda()
def error(model):
    x = torch.from_numpy(x_tr.astype('float32')).type(dtype)
    y = torch.from_numpy(y_tr.astype('float32')).type(dtype)
    x_val = torch.from_numpy(x_tt.astype('float32')).type(dtype)
    y_val = torch.from_numpy(y_tt.astype('float32')).type(dtype)

    val_pred = model(Variable(x_val))
    error = val_pred.cpu().data.numpy() - y_val.numpy()
    return np.mean(np.absolute(error))

print(error(model))

# The following are functions meant for debugging and sanity checking your
# model. You should use these and / or design your own testing tools.
# test_model starts at [0,0,0]; you can specify a control to be applied and the
# rollout() function will use that control for N steps.
# i.e. a velocity value of 0.7 should drive the car to a positive x value.
def rollout(model, nn_input, steps):
    print('Generating Rollouts: ')
    pose = torch.zeros(3).cuda()
    x = []
    y = []
    theta = []
    t = []
    s_values = [0.0, 0.0, 0.0, 0.0]
    print(pose.cpu().numpy())
    for i in range(steps):
        out = model(Variable(nn_input))
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
    return (x, y, theta, t)
        # print(pose.cpu().numpy())
        # if i != 0 and i % 20 == 0:
        #     plt.plot(t, x)
        #     plt.show()
        #     plt.plot(t, y)
        #     plt.show()
        #     plt.plot(t, theta)
        #     plt.show()
        #     plt.plot(x, y)
        #     plt.show()


def test_model(model, steps, dt = 0.1):
    cos, v, st = 4, 5, 6
    s = INPUT_SIZE
    print("Testing No Velocity")
    nn_input = torch.zeros(s).cuda()
    nn_input[cos] = 1.0
    nn_input[7] = dt
    idle_xs, idle_ys, idle_thetas, idle_ts = rollout(m, nn_input, N)

    print("Testing Forward Velocity")
    nn_input = torch.zeros(s).cuda()
    nn_input[cos] = 1.0
    nn_input[v] = 0.7 #1.0
    nn_input[7] = dt
    forward_xs, forward_ys, forward_thetas, forward_ts = rollout(m, nn_input, N)

    print("Backward")
    nn_input = torch.zeros(s).cuda()
    nn_input[cos] = 1.0
    nn_input[v] = -0.7 #1.0
    nn_input[7] = dt
    backward_xs, backward_ys, backward_thetas, backward_ts = rollout(m, nn_input, N)

    print("Left Turn")
    nn_input = torch.zeros(s).cuda()
    nn_input[cos] = 1.0
    nn_input[v] = 0.7 #1.0
    nn_input[st] = 0.26
    nn_input[7] = dt
    left_xs, left_ys, left_thetas, left_ts = rollout(m, nn_input, N)

    print("Right Turn")
    nn_input = torch.zeros(s).cuda()
    nn_input[cos] = 1.0
    nn_input[v] = 0.7 #1.0
    nn_input[st] = -0.26
    nn_input[7] = dt
    right_xs, right_ys, right_thetas, right_ts = rollout(m, nn_input, N)

    plt.plot(idle_xs, idle_ys, "r", forward_xs, forward_ys, "g", backward_xs, backward_ys, "b", left_xs, left_ys, "y", right_xs, right_ys, "k")
    plt.show()


test_model(model, 11)
