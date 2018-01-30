#!/usr/bin/env python

import rospy
import rosbag
import time
import numpy as np
import matplotlib.pyplot as plt
from threading import Lock

import matplotlib.colors as colors
import matplotlib.cm as cm
from std_msgs.msg import Int32, String
from SensorModel import SensorModel
from nav_msgs.srv import GetMap




class ParticleLikelihood():
    """
        Initalize the ParticleLikelihood, by getting the bag and message

        Input:
            message: The message from the bag file

    """
    def __init__(self, message):
        #print(message.ranges)
        self.message = message

        # Need to run the map server to get the map
        rospy.wait_for_service('/static_map')
        get_map = rospy.ServiceProxy('/static_map', GetMap)
        self.map_msg = get_map().map

        #print("Map msg header: ",map_msg.header)
        #print("length of map msg data: ", len(map_msg.data))
        #print("Map msg info: ", map_msg.info)

        self.width = self.map_msg.info.width
        #print("Width: ", width)

        self.height = self.map_msg.info.height
        #print("Height: ", height)

        self.resolution = self.map_msg.info.resolution
        #print("Resolution: ", resolution )

        self.origin = self.map_msg.info.origin.position
        #print("Origin: ", origin )

    """
        Discretize the map and get the particle. Then apply the sensor MODEL
        to get the particle likelihood.

        Input:
            meter_gap: the distance between each particle in meter
            theta_step_size: The number of orientation for each x,y position
            plot_likelihood: Plot the particle likelihood?
        Output:
            The time it took to discretize and observe the map

    """
    def apply_sensor_model(self,meter_gap=.02, theta_step_size=10,plot_likelihood = True):
        start = time.time()

        particles = []
        max_particles = 0

        #theta_step_size = 10    # The number of different orientation
        #meter_gap = .02         # The discretization of the map in meter
        cell_gap = int(meter_gap/self.resolution)

        # Discretize the map with spceified meter
        for i in range(0,self.height, cell_gap):
            for j in range(0,self.width, cell_gap):
                if(self.map_msg.data[i*self.height+j] != -1):
                    # Place a particle in all/some x,y valid position
                    #print("Known space: (", i, ", " ,j , ") with value ",
                    #      map_msg.data[i*height+j])
                    x = j * self.resolution + self.origin.x
                    y = i * self.resolution + self.origin.y
                    # Iterate through some theta.
                    for theta in range(1, theta_step_size+1):
                        particles.append([x, y, np.pi*2 * theta / theta_step_size])
                        max_particles += 1
        #print("Done getting particles")
        weights = np.ones(max_particles, dtype=np.float64) / max_particles
        np_particles = np.array(particles, dtype=np.float64)

        #print("Old weights ",weights)
        #print("Old position ",np_particles)
        #print("Maximum particles ",max_particles)

        sensorModel = SensorModel(self.map_msg, np_particles, weights )
        sensorModel.lidar_cb(self.message)
        new_particles = sensorModel.particles
        new_weights = sensorModel.weights

        #print("New weights ", new_weights)
        #print("New particles ", new_particles)

        #x = new_particles[:,0]
        #y = new_particles[:,1]

        max_weight = []
        max_particle = []
        for i in range(max_particles/theta_step_size):
            weight_xy = new_weights[i*theta_step_size: (i+1)*theta_step_size]
            #print("Weight xy: ", weight_xy)
            ind = np.argmax(weight_xy)
            max_weight_xy = weight_xy[ind]
            #print("Max weight xy: ", max_weight_xy)
            #print("Index: ", ind)
            max_weight.append(max_weight_xy)
            max_particle.append(list(new_particles[i*theta_step_size + ind]))

        max_particle = np.array(max_particle)
        max_weight = np.array(max_weight)

        #print("Max particle:", max_particle)
        #print("Max weight:", max_weight)

        #print("Max particle len: ", len(max_particle))
        #print("Max weight len: ", len(max_weight))
        x = max_particle[:, 0]
        y = max_particle[:, 1]

        np.amax(max_weight)
        #print("The largest weight: ", np.amax(max_weight))
        #print("The smallest weight: ", np.amin(max_weight))

        end = time.time()

        if plot_likelihood:
            plt.scatter(x, y, c=max_weight, vmin = np.amin(max_weight), vmax = np.amax(max_weight),cmap=cm.get_cmap('cool'),edgecolor='')
            plt.gray()
            plt.show()

        #print("Got to update sensor model")

        return end - start

if __name__=='__main__':
    rospy.init_node("particle_likelihood", anonymous=True)
    BAG_FILENAME = str(rospy.get_param("~bag_file"))
    bag = rosbag.Bag(BAG_FILENAME)
    message = None
    try:
        for topic, msg, t in bag.read_messages():
            message = msg
    finally:
        bag.close()
    pl = ParticleLikelihood(message)

    pl.apply_sensor_model(.02)

    """
    MIN_METER = .02
    MAX_METER = .5
    METER_STEP = .02
    time_list = []
    meter_dis = []

    for meter in np.arange(MIN_METER, MAX_METER, METER_STEP):
        delta_time = pl.apply_sensor_model(meter, plot_likelihood = F alse)
        time_list.append(delta_time)
        meter_dis.append(meter)
        print('Completed discretization with meter = ', meter)
        print('Time it took to complete is ', delta_time ,' sec')
    plt.plot(meter_dis, time_list)
    plt.xlabel("Distance between each particle (meter)")
    plt.ylabel("Time (sec)")
    plt.title("Time vs Discretization")
    plt.show()
    """
