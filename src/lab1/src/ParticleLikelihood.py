#!/usr/bin/env python

import rospy
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from threading import Lock

import matplotlib.colors as colors
from std_msgs.msg import Int32, String
from SensorModel import SensorModel
from nav_msgs.srv import GetMap




class ParticleLikelihood():
    def __init__(self):
        BAG_FILENAME = str(rospy.get_param("~bag_file"))
        bag = rosbag.Bag(BAG_FILENAME)
        message = None
        try:
            for topic, msg, t in bag.read_messages():
                message = msg
        finally:
            bag.close()

        #print(message.ranges)

        rospy.wait_for_service('/static_map')
        get_map = rospy.ServiceProxy('/static_map', GetMap)
        map_msg = get_map().map

        print(map_msg.header)
        print(len(map_msg.data))

        print(map_msg.info)

        width = map_msg.info.width
        print("Width: ", width)

        height = map_msg.info.height
        print("Height: ", height)

        resolution = map_msg.info.resolution
        print("Resolution: ", resolution )

        origin = map_msg.info.origin.position
        print("Origin: ", origin )


        particles = []
        max_particles = 0

        theta_step_size = 10    # The number of different orientation
        meter_var = .02         # The discretization of the map in meter
        cell_var = int(meter_var/resolution)

        for i in range(0,height, cell_var):
            for j in range(0,width, cell_var):
                if(map_msg.data[i*height+j] != -1):
                    # Place a particle in all/some x,y valid position
                    #print("Known space: (", i, ", " ,j , ") with value ",
                    #      map_msg.data[i*height+j])
                    x = j * resolution + origin.x
                    y = i * resolution + origin.y
                    # Iterate through some theta.
                    for theta in range(1, theta_step_size+1):
                        particles.append([x, y, np.pi*2 * theta / theta_step_size])
                        max_particles += 1

        print("Done getting particles")
        weights = np.ones(max_particles, dtype=np.float64) / max_particles
        np_particles = np.array(particles, dtype=np.float64)

        print("Old weights ",weights)
        print("Old position ",np_particles)
        print("Maximum particles ",max_particles)

        state_lock = Lock()
        sensorModel = SensorModel(map_msg, np_particles, weights )
        sensorModel.lidar_cb(message)
        new_particles = sensorModel.particles
        new_weights = sensorModel.weights

        print("New weights ", new_weights)
        print("New particles ", new_particles)

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

        #max_weight *= (1/np.amax(max_weight))



        print("Max particle:", max_particle)
        print("Max weight:", max_weight)

        print("Max particle len: ", len(max_particle))
        print("Max weight len: ", len(max_weight))
        x = max_particle[:, 0]
        y = max_particle[:, 1]

        print("The largest weight: ", np.amax(max_weight))
        print("The smallest weight: ", np.amin(max_weight))

        plt.scatter(x, y, c=max_weight,edgecolor='')
        plt.gray()
        plt.show()

        print("Got to update sensor model")

if __name__=='__main__':
    rospy.init_node("particle_likelihood", anonymous=True)
    pl = ParticleLikelihood()
