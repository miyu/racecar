#!/usr/bin/env python

import math
import rospy
import numpy as np
import time
import utils as Utils
import tf.transformations
from tf.transformations import euler_from_quaternion
import tf
from threading import Lock
from copy import deepcopy

from vesc_msgs.msg import VescStateStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped, Pose

from ReSample import ReSampler
from SensorModel import SensorModel
from MotionModel import OdometryMotionModel, KinematicMotionModel
import rosbag
import random
from Debug import print_locks, print_benchmark
import matplotlib.pyplot as plt


class ParticleFilterTest():
    def __init__(self):
        self.MAX_PARTICLES = int(rospy.get_param("~max_particles")) # The maximum number of particles
        self.MAX_VIZ_PARTICLES = int(rospy.get_param("~max_viz_particles")) # The maximum number of particles to visualize

        self.particle_indices = np.arange(self.MAX_PARTICLES)
        self.particles = np.zeros((self.MAX_PARTICLES,3)) # Numpy matrix of dimension MAX_PARTICLES x 3
        self.weights = np.ones(self.MAX_PARTICLES, dtype=np.float64) / float(self.MAX_PARTICLES) # Numpy matrix containig weight for each particle

        self.state_lock = Lock() # A lock used to prevent concurrency issues. You do not need to worry about this

        # Use the 'static_map' service (launched by MapServer.launch) to get the map
        # Will be used to initialize particles and SensorModel
        # Store map in variable called 'map_msg'
        # YOUR CODE HERE
        rospy.wait_for_service('/static_map')
        get_map = rospy.ServiceProxy('/static_map', GetMap)
        map_msg = get_map().map

        # Use dir to show msg fields
        # print(dir(map_msg))
        # print(dir(map_msg))

        # Globally initialize the particles
        self.map_free_space = []
        self.initialize_global(map_msg)
        print("Z!")


        # Publish particle filter state
        self.pose_pub      = rospy.Publisher("/pf/viz/inferred_pose", PoseStamped, queue_size = 1) # Publishes the expected pose
        self.particle_pub  = rospy.Publisher("/pf/viz/particles", PoseArray, queue_size = 1) # Publishes a subsample of the particles
        self.pub_tf = tf.TransformBroadcaster() # Used to create a tf between the map and the laser for visualization
        self.pub_laser     = rospy.Publisher("/pf/viz/scan", LaserScan, queue_size = 1) # Publishes the most recent laser scan
        print("Y!")

        self.RESAMPLE_TYPE = rospy.get_param("~resample_type", "naiive") # Whether to use naiive or low variance sampling
        self.resampler = ReSampler(self.particles, self.weights, self.state_lock, self.map_free_space)  # An object used for resampling
        print("X!")

        self.sensor_model = SensorModel(map_msg, self.particles, self.weights, self.state_lock) # An object used for applying sensor model
        self.laser_sub = rospy.Subscriber(rospy.get_param("~scan_topic", "/scan"), LaserScan, self.sensor_model.lidar_cb, queue_size=1)
        print("XZ!")

        self.MOTION_MODEL_TYPE = rospy.get_param("~motion_model", "kinematic") # Whether to use the odometry or kinematics based motion model
        print("!~!@#@! MOTION MODEL TYPE", self.MOTION_MODEL_TYPE)
        if self.MOTION_MODEL_TYPE == "kinematic":
            self.motion_model = KinematicMotionModel(self.particles, self.state_lock) # An object used for applying kinematic motion model
            self.motion_sub = rospy.Subscriber(rospy.get_param("~motion_topic", "/vesc/sensors/core"), VescStateStamped, self.motion_model.motion_cb, queue_size=1)
        elif self.MOTION_MODEL_TYPE == "odometry":
            self.motion_model = OdometryMotionModel(self.particles, self.state_lock)# An object used for applying odometry motion model
            self.motion_sub = rospy.Subscriber(rospy.get_param("~motion_topic", "/vesc/odom"), Odometry, self.motion_model.motion_cb, queue_size=1)
        else:
            print "Unrecognized motion model: "+ self.MOTION_MODEL_TYPE
            assert(False)

        print("A!")
        # Use to initialize through rviz. Check clicked_pose_cb for more info
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.clicked_pose_cb, queue_size=1)

    # Initialize the particles to cover the map
    def initialize_global(self, map_msg):
        for grid_x in range(map_msg.info.height):
            for grid_y in range(map_msg.info.width):
                if map_msg.data[grid_x + grid_y * map_msg.info.width] == -1:
                    continue
		world_x = grid_x * map_msg.info.resolution + map_msg.info.origin.position.x
		world_y = grid_y * map_msg.info.resolution + map_msg.info.origin.position.y
                self.map_free_space.append((world_x, world_y))


        for i in range(0, self.MAX_PARTICLES):
            ind = random.randint(0, len(self.map_free_space) - 1) #who the hell does an inclsuive end of range? omg
            self.particles[i][0] = self.map_free_space[ind][0]
            self.particles[i][1] = self.map_free_space[ind][1]
            self.particles[i][2] = np.random.uniform(0, 2*np.pi)

        self.map_free_space = np.array(self.map_free_space)


    # Publish a tf between the laser and the map
    # This is necessary in order to visualize the laser scan within the map
    def publish_tf(self, pose):
        # Use self.pub_tf
        # YOUR CODE HERE
        self.pub_tf.sendTransform( \
            (pose[0], pose[1], 0), \
            tf.transformations.quaternion_from_euler(0, 0, pose[2]), \
            rospy.Time.now(), \
            "/laser",
            "/map")

            # "map",
            # child_frame_id) # /scan => /map

    # Returns the expected pose given the current particles and weights
    def expected_pose(self, particles=None, weights=None):
        particles = particles if particles is not None else self.particles
        weights = weights if weights is not None else self.weights

        # YOUR CODE HERE
        num_particles = particles.shape[0]
        posx = (particles[:, 0] * weights).sum()
        posy = (particles[:, 1] * weights).sum()
        orienxacc = (np.cos(particles[:, 2]) * weights).sum()
        orienyacc = (np.sin(particles[:, 2]) * weights).sum()
        orien = math.atan2(orienyacc, orienxacc)
        return np.array([posx, posy, orien])

        # for i in range(0, len(particles)):
        #     [x, y, theta] = particles[i]
        #     w = weights[i]
        #     pos_acc += np.array([x, y], dtype=float) * float(w)
        #     orien_acc += np.array([math.cos(theta), math.sin(theta)], dtype=float) * w
        #
        # pos_expected = pos_acc / np.sum(weights)
        # theta_expected = math.atan2(orien_acc[1], orien_acc[0])
        #
        # return np.array([pos_expected[0], pos_expected[1], theta_expected], dtype=float)

    # Callback for '/initialpose' topic. RVIZ publishes a message to this topic when you specify an initial pose using its GUI
    # Reinitialize your particles and weights according to the received initial pose
    # Remember to apply a reasonable amount of Gaussian noise to each particle's pose
    def clicked_pose_cb(self, msg):
        print("Entering lock clicked_pose_cb")
        self.state_lock.acquire()
        print("Entered lock clicked_pose_cb")

        print(msg)

        center_x = msg.pose.pose.position.x
        center_y = msg.pose.pose.position.y
        theta = euler_from_quaternion([ \
            msg.pose.pose.orientation.x, \
            msg.pose.pose.orientation.y, \
            msg.pose.pose.orientation.z, \
            msg.pose.pose.orientation.w])[2]

        for i in range(0, self.MAX_PARTICLES):
            self.particles[i][0] = center_x + np.random.normal(0.0, 1.0)
            self.particles[i][1] = center_y + np.random.normal(0.0, 1.0)
            self.particles[i][2] = theta + np.random.normal(0.0, 0.3)
            self.weights[i] = 1.0 / self.MAX_PARTICLES

        print("Exiting lock clicked_pose_cb")
        self.state_lock.release()

    # Visualize the current state of the filter
    # (1) Publishes a tf between the map and the laser. Necessary for visualizing the laser scan in the map
    # (2) Publishes the most recent laser measurement. Note that the frame_id of this message should be the child_frame_id of the tf from (1)
    # (3) Publishes a PoseStamped message indicating the expected pose of the car
    # (4) Publishes a subsample of the particles (use self.MAX_VIZ_PARTICLES).
    #     Sample so that particles with higher weights are more likely to be sampled.
    def visualize(self):
        def make_pose_from_xy_theta(xytheta):
            pose = Pose()
            pose.position.x = xytheta[0]
            pose.position.y = xytheta[1]
            pose.position.z = 0
            orien = tf.transformations.quaternion_from_euler(0, 0, xytheta[2])
            pose.orientation.x = orien[0]
            pose.orientation.y = orien[1]
            pose.orientation.z = orien[2]
            pose.orientation.w = orien[3]
            return pose

        print_locks("Entering lock visualize")
        self.state_lock.acquire()
        print_locks("Entered lock visualize")
        start_time = time.time()
        laser = self.sensor_model.last_laser # no deepcopy
        particles = np.copy(self.particles)
        weights = np.copy(self.weights)
        print_locks("Exiting lock visualize (computation will continue)")
        self.state_lock.release()

        # compute pose
        pose = self.expected_pose(particles, weights)

        # (1) Publishes a tf between the map and the laser. Necessary for visualizing the laser scan in the map
        self.publish_tf(pose)

        # (2) Publishes the most recent laser measurement. Note that the frame_id of this message should be the child_frame_id of the tf from (1)
        laser.header.frame_id = "/laser" #"laser"
        self.pub_laser.publish(laser) # /scan /map /map

        # (3) Publishes a PoseStamped message indicating the expected pose of the car
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = '/map'
        pose_msg.pose = make_pose_from_xy_theta(pose)
        self.pose_pub.publish(pose_msg) #PoseStamped(pose)

        # (4) Publishes a subsample of the particles (use self.MAX_VIZ_PARTICLES).
        #     Sample so that particles with higher weights are more likely to be sampled.
        particle_indices = np.random.choice(self.MAX_PARTICLES, size=self.MAX_VIZ_PARTICLES, replace=False, p=weights)
        sampled_particles = particles[particle_indices, :]
        particles_msg = PoseArray()
        particles_msg.header.stamp = rospy.Time.now()
        particles_msg.header.frame_id = '/map'
        particles_msg.poses = [make_pose_from_xy_theta(sampled_particles[i]) for i in range(len(sampled_particles))]
        self.particle_pub.publish(particles_msg)

        print_locks("Finished visualize")
        print_benchmark("visualize", start_time, time.time())

# Suggested main
if __name__ == '__main__':
    rospy.init_node("particle_test", anonymous=True)
    BAG_FILENAME = str(rospy.get_param("~bag_file"))
    bag = rosbag.Bag(BAG_FILENAME)
    message = None

    map_free_space = []

    # Need to run the map server to get the map
    rospy.wait_for_service('/static_map')
    get_map = rospy.ServiceProxy('/static_map', GetMap)
    map_msg = get_map().map

    sensorModel = None#SensorModel(map_msg)
    kinematicModel =  None #KinematicMotionModel(paticles, state_lock)
    resampler = None
    numParticles = 8000
    state_lock = Lock() # A lock used to prevent concurrency issues. You do not need to worry about this

    st = None
    timer = []
    RMSE = []
    expected_pose = None
    computation_time = []

    num_particles_list = [200, 800, 1600, 4000, 8000]

    for numParticles in num_particles_list:
        start = time.time()
        rospy.init_node("particle_test", anonymous=True)
        BAG_FILENAME = str(rospy.get_param("~bag_file"))
        bag = rosbag.Bag(BAG_FILENAME)
        message = None

        try:
            for topic, msg, t in bag.read_messages():
                """
                if topic == '/vesc/odom' and noiseProp is not None:
                    print("time: ", msg.header.stamp.to_sec())
                    odomControlCount += 1
                    noiseProp.addOdomMessage(msg)
                    print("odom")
                """
                if topic == '/initialpose':
                    x = msg.pose.pose.position.x
                    y = msg.pose.pose.position.y
                    theta = Utils.quaternion_to_angle(msg.pose.pose.orientation)
                    particles = np.repeat(np.array([[x,y,theta]], dtype=np.float64),numParticles,axis=0)
                    weights = np.ones(numParticles) / numParticles
                    #print("Particles: ", particles)
                    sensorModel = SensorModel(map_msg, particles, weights, state_lock )
                    kinematicModel = KinematicMotionModel(particles, state_lock)
                    resampler = ReSampler(particles, weights,state_lock, None)  # An object used for resampling
                    st = t
                    #print("initalpose")
                elif topic == '/pf/ta/viz/inferred_pose' :
                    x = msg.pose.position.x
                    y = msg.pose.position.y
                    expected_pose = [x, y]

                elif topic == "/vesc/sensors/core" and kinematicModel is not None:
                    kinematicModel.motion_cb(msg)
                elif topic == "/vesc/sensors/servo_position_command" and kinematicModel is not None:
                    kinematicModel.servo_cb(msg)
                elif topic == "/scan" and sensorModel is not None:
                    sensorModel.lidar_cb(msg)
                    resampler.resample_low_variance(state_lock)
                    timer.append((t-st).to_sec())

                    mean = 0
                    error_list = []
                    for i in range(len(particles)):
                        par_x = particles[i][0]
                        par_y = particles[i][1]
                        error = (par_x - expected_pose[0]) ** 2 + (par_y - expected_pose[1]) ** 2
                        error_list.append(error)
                    RMSE.append(np.sqrt(np.mean(error_list)))



        finally:
            bag.close()
        end = time.time()
        computation_time.append(end - start)
        print("Next particle size")
    #print("Time: ", timer)
    #print(RMSE)
    #plt.plot(timer, RMSE)
    plt.plot(num_particles_list, computation_time)
    plt.show()
    print("done")
