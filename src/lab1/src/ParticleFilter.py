#!/usr/bin/env python

import rospy
import numpy as np
import time
import utils as Utils
import tf.transformations
import tf
from threading import Lock

from vesc_msgs.msg import VescStateStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped

from ReSample import ReSampler
from SensorModel import SensorModel
from MotionModel import OdometryMotionModel, KinematicMotionModel

import random


class ParticleFilter():
    def __init__(self):
        self.MAX_PARTICLES = int(rospy.get_param("~max_particles")) # The maximum number of particles
        self.MAX_VIZ_PARTICLES = int(rospy.get_param("~max_viz_particles")) # The maximum number of particles to visualize

        self.particle_indices = np.arange(self.MAX_PARTICLES)
        self.particles = np.zeros((self.MAX_PARTICLES,3)) # Numpy matrix of dimension MAX_PARTICLES x 3
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES) # Numpy matrix containig weight for each particle

        self.state_lock = Lock() # A lock used to prevent concurrency issues. You do not need to worry about this

        # Use the 'static_map' service (launched by MapServer.launch) to get the map
        # Will be used to initialize particles and SensorModel
        # Store map in variable called 'map_msg'
        # YOUR CODE HERE
        rospy.wait_for_service('/static_map')
        get_map = rospy.ServiceProxy('/static_map', GetMap)
        map_msg = get_map()

        # Globally initialize the particles
        self.initialize_global(map_msg)

        # Publish particle filter state
        self.pose_pub      = rospy.Publisher("/pf/viz/inferred_pose", PoseStamped, queue_size = 1) # Publishes the expected pose
        self.particle_pub  = rospy.Publisher("/pf/viz/particles", PoseArray, queue_size = 1) # Publishes a subsample of the particles
        self.pub_tf = tf.TransformBroadcaster() # Used to create a tf between the map and the laser for visualization
        self.pub_laser     = rospy.Publisher("/pf/viz/scan", LaserScan, queue_size = 1) # Publishes the most recent laser scan

        self.RESAMPLE_TYPE = rospy.get_param("~resample_type", "naiive") # Whether to use naiive or low variance sampling
        self.resampler = ReSampler(self.particles, self.weights, self.state_lock)  # An object used for resampling

        self.sensor_model = SensorModel(map_msg, self.particles, self.weights, self.state_lock) # An object used for applying sensor model
        self.laser_sub = rospy.Subscriber(rospy.get_param("~scan_topic", "/scan"), LaserScan, self.sensor_model.lidar_cb, queue_size=1)

        self.MOTION_MODEL_TYPE = rospy.get_param("~motion_model", "kinematic") # Whether to use the odometry or kinematics based motion model

        if self.MOTION_MODEL_TYPE == "kinematic":
            self.motion_model = KinematicMotionModel(self.particles, self.state_lock) # An object used for applying kinematic motion model
            self.motion_sub = rospy.Subscriber(rospy.get_param("~motion_topic", "/vesc/sensors/core"), VescStateStamped, self.motion_model.motion_cb, queue_size=1)
        elif self.MOTION_MODEL_TYPE == "odometry":
            self.motion_model = OdometryMotionModel(self.particles, self.state_lock)# An object used for applying odometry motion model
            self.motion_sub = rospy.Subscriber(rospy.get_param("~motion_topic", "/vesc/odom"), Odometry, self.motion_model.motion_cb, queue_size=1)
        else:
            print "Unrecognized motion model: "+ self.MOTION_MODEL_TYPE
            assert(False)

        # Use to initialize through rviz. Check clicked_pose_cb for more info
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.clicked_pose_cb, queue_size=1)

    # Initialize the particles to cover the map
    def initialize_global(self, map_msg):
        # self.particle_indices = np.arange(self.MAX_PARTICLES)
        # self.particles = np.zeros((self.MAX_PARTICLES,3)) # Numpy matrix of dimension MAX_PARTICLES x 3
        # self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES) # Numpy matrix containig weight for each particle
        for i in range(0, self.MAX_PARTICLES):
            while True:
                ind = random.randint(0, len(map_msg.data))
                if map_msg.data[ind] != -1:
                    grid_x = i % map_msg.info.width
                    grid_y = i / map_msg.info.width
                    world_x = grid_x * map_msg.info.resolution + map_msg.info.origin.position.x
                    world_y = grid_y * map_msg.info.resolution + map_msg.info.origin.position.y
                    self.particles[i][0] = world_x
                    self.particles[i][1] = world_y
                    self.particles[i][2] = random.uniform(0, 6.28)
                    break

    # Publish a tf between the laser and the map
    # This is necessary in order to visualize the laser scan within the map
    def publish_tf(self, pose, child_frame_id):
        # Use self.pub_tf
        # YOUR CODE HERE
        self.pub_tf.sendTransform( \
            (pose[0], pose[1], 0), \
            tf.transformations.quaternion_from_euler(0, 0, pose[2]), \
            rospy.Time.now(), \
            child_frame_id, \
            "world")

    # Returns the expected pose given the current particles and weights
    def expected_pose(self):
        # YOUR CODE HERE
        pos_acc = np.array([0, 0])
        orien_acc = np.array([0, 0])

        for i in range(0, len(self.particles)):
            [x, y, theta] = self.particles
            w = self.weights[i]
            pos_acc += np.array([x, y], dtype=float) * w
            orien_acc += np.array([math.cos(theta), math.sin(theta)], dtype=float) * w

        pos_expected = pos_acc / np.sum(self.weights)
        theta_expected = math.atan2(orien.acc[1], orien_acc[0])

        return np.array([pos_expected[0], pos_expected[1], theta_expected], dtype=float)

    # Callback for '/initialpose' topic. RVIZ publishes a message to this topic when you specify an initial pose using its GUI
    # Reinitialize your particles and weights according to the received initial pose
    # Remember to apply a reasonable amount of Gaussian noise to each particle's pose
    def clicked_pose_cb(self, msg):
        self.state_lock.acquire()

        center_x = msg.x
        center_y = msg.y

        for i in range(0, self.MAX_PARTICLES):
            self.particles[i][0] = center_x + np.random.normal(0.0, 10.0)
            self.particles[i][1] = center_y + np.random.normal(0.0, 10.0)
            self.particles[i][2] = np.random.uniform(0, 6.28)
            self.weights[i] = 1

        self.state_lock.release()

    # Visualize the current state of the filter
    # (1) Publishes a tf between the map and the laser. Necessary for visualizing the laser scan in the map
    # (2) Publishes the most recent laser measurement. Note that the frame_id of this message should be the child_frame_id of the tf from (1)
    # (3) Publishes a PoseStamped message indicating the expected pose of the car
    # (4) Publishes a subsample of the particles (use self.MAX_VIZ_PARTICLES).
    #     Sample so that particles with higher weights are more likely to be sampled.
    def visualize(self):
        self.state_lock.acquire()

        pose = self.expected_pose()

        # (1) Publishes a tf between the map and the laser. Necessary for visualizing the laser scan in the map
        self.publish_tf(pose, self.sensor_model.last_laser.header.frame_id)

        # (2) Publishes the most recent laser measurement. Note that the frame_id of this message should be the child_frame_id of the tf from (1)
        self.pub_laser.publish(self.sensor_model.last_laser)

        # (3) Publishes a PoseStamped message indicating the expected pose of the car
        pose_msg = PoseStamped()
        print("POSE MSG BEFORE", pose_msg)
        pose_msg.header = std_msgs.msg.Header()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose = pose
        self.pose_pub.publish(pose_msg) #PoseStamped(pose)

        # (4) Publishes a subsample of the particles (use self.MAX_VIZ_PARTICLES).
        #     Sample so that particles with higher weights are more likely to be sampled.
        def make_pose(particle):
            pose = Pose()
            pose.position.x = particle[0]
            pose.position.y = particle[1]
            pose.position.z = 0
            pose.orientation = tf.transformations.quaternion_from_euler(0, 0, particle[2])
            return pose

        sampled_particles = np.random.choice(self.particles, size=self.MAX_VIZ_PARTICLES, replace=False, p=self.weights)
        particles_msg = PoseArray()
        particles_msg.header = std_msgs.msg.Header()
        particles_msg.header.stamp = rospy.Time.now()
        particles_msg.poses = [make_pose(sampled_particles[i]) for i in range(len(sampled_particles))]
        self.particle_pub.publish(particles_msg)

        self.state_lock.release()

# Suggested main
if __name__ == '__main__':
    rospy.init_node("particle_filter", anonymous=True) # Initialize the node
    pf = ParticleFilter() # Create the particle filter

    while not rospy.is_shutdown(): # Keep going until we kill it
        # Callbacks are running in separate threads
        if pf.sensor_model.do_resample: # Check if the sensor model says it's time to resample
            pf.sensor_model.do_resample = False # Reset so that we don't keep resampling

        # Resample
        if pf.RESAMPLE_TYPE == "naiive":
            pf.resampler.resample_naiive()
        elif pf.RESAMPLE_TYPE == "low_variance":
            pf.resampler.resample_low_variance()
        else:
            print "Unrecognized resampling method: "+ pf.RESAMPLE_TYPE

      pf.visualize() # Perform visualization
