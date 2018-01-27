#!/usr/bin/env python
import math
import numpy as np
import rospy
import range_libc
import time
from sensor_msgs.msg import LaserScan
from threading import Lock

from Debug import print_locks

THETA_DISCRETIZATION = 112 # Discretization of scanning angle
SQUASH_FACTOR_N = 1.5
INV_SQUASH_FACTOR = 1 / SQUASH_FACTOR_N    # Factor for helping the weight distribution to be less peaked

Z_HIT = 5 # Weight for hit reading, hit obj properly
Z_SHORT = 1 # Weight for short reading, something in the way
Z_MAX = 0.05 # Weight for max reading, out of range
Z_RAND = 2 # Weight for random reading, random noise

Z_NORM = float(Z_HIT + Z_SHORT + Z_MAX + Z_RAND)
Z_HIT /= Z_NORM
Z_SHORT /= Z_NORM
Z_MAX /= Z_NORM
Z_RAND /= Z_NORM

SIGMA_HIT = 20 # Noise value for hit reading

# Z_HIT = 0
# Z_SHORT = 0
# Z_MAX = 0
# Z_RAND = 1

LAMBDA_SHORT = 0.01

class SensorModel:
    def __init__(self, map_msg, particles, weights, state_lock=None):
        if state_lock is None:
            self.state_lock = Lock()
        else:
            self.state_lock = state_lock

        self.particles = particles
        self.weights = weights

        self.LASER_RAY_STEP = int(rospy.get_param("~laser_ray_step")) # Step for downsampling laser scans
        self.MAX_RANGE_METERS = float(rospy.get_param("~max_range_meters")) # The max range of the laser

        print("ZZ!")
        oMap = range_libc.PyOMap(map_msg) # A version of the map that range_libc can understand
        print("ZA!")
        print("ZA!")
        max_range_px = int(self.MAX_RANGE_METERS / map_msg.info.resolution) # The max range in pixels of the laser
        self.range_method = range_libc.PyCDDTCast(oMap, max_range_px, THETA_DISCRETIZATION) # The range method that will be used for ray casting
        print("ZZZZZ!")
        self.range_method.set_sensor_model(self.precompute_sensor_model(max_range_px)) # Load the sensor model expressed as a table
        print("ZB!")
        self.queries = None
        self.ranges = None
        self.laser_angles = None # The angles of each ray
        self.downsampled_angles = None # The angles of the downsampled rays
        self.do_resample = False # Set so that outside code can know that it's time to resample
        print("ZC!")

        self.laser_sub = rospy.Subscriber(rospy.get_param("~scan_topic", "/scan"), LaserScan, self.lidar_cb, queue_size=1)
        print("ZD!")
        #print(self.precompute_sensor_model(7))

    def lidar_cb(self, msg):
        # # comment back in to no-op
        # self.last_laser = msg
        # self.do_resample = True
        # return

        print_locks("Entering lock lidar_cb")
        self.state_lock.acquire()
        print_locks("Entered lock lidar_cb")

        # Compute the observation
        # obs is a a two element tuple
        # obs[0] is the downsampled ranges
        # obs[1] is the downsampled angles
        # Each element of obs must be a numpy array of type np.float32 (this is a requirement for GPU processing)
        # Use self.LASER_RAY_STEP as the downsampling step
        # Keep efficiency in mind, including by caching certain things that won't change across future iterations of this callback
        obs = [[], []]
        for i in range(0, len(msg.ranges), self.LASER_RAY_STEP):
            #if not math.isnan(msg.ranges[i]):
            obs[0].append(msg.ranges[i])
            obs[1].append(msg.angle_min + i * msg.angle_increment)

        obs = [np.array(obs[0], dtype=np.float32), np.array(obs[1], dtype=np.float32)]

        prior = np.array(self.weights, dtype=np.float32)
        self.apply_sensor_model(self.particles, obs, self.weights)
        self.weights /= np.sum(self.weights)
        after = np.array(self.weights, dtype=np.float32)

        #print("PRIOR", [x for x in reversed(sorted(prior))][0:20])
        #print("AFTER", [x for x in reversed(sorted(after))][0:20])
        #print("DELTA", [x for x in reversed(sorted(after - prior))][0:20])

        self.last_laser = msg
        self.do_resample = True

        print_locks("Exiting lock lidar_cb")
        self.state_lock.release()
        #print(self.precompute_sensor_model(280))

    def precompute_sensor_model(self, max_range_px):
        table_width = int(max_range_px) + 1
        sensor_model_table = np.zeros((table_width,table_width))

        # ch 6.2 - have distribution for what a range would be
        # given map and location.

        # Populate sensor model table as specified
        # Note that the row corresponds to the observed measurement and the column corresponds to the expected measurement
        # YOUR CODE HERE

        assert abs((Z_HIT + Z_SHORT + Z_MAX + Z_RAND) - 1.0) < 1E-4

        def compute_eng(ztk, ztkstar, varhit):
            denom1 = math.sqrt(2.0 * np.pi * varhit)
            numer2 = math.pow(ztk - ztkstar, 2.0)
            denom2 = varhit
            return (1.0 / denom1) * np.exp((-0.5) * numer2 / denom2)


        def compute_p_short_unnormalized(ztk, ztkstar):
            if 0.0 <= ztk <= ztkstar:
                return LAMBDA_SHORT * np.exp(-LAMBDA_SHORT * ztk)
            else:
                return 0.0

        engs = np.zeros((table_width, table_width))
        p_short_unnormalized = np.zeros((table_width, table_width))
        for i in range(table_width):
            for j in range(table_width):
                engs[i][j] = compute_eng(float(i), float(j), SIGMA_HIT * SIGMA_HIT)
                p_short_unnormalized[i][j] = compute_p_short_unnormalized(float(i), float(j))

        engs = engs / engs.sum(axis=0, dtype=float)
        p_short_normalized = p_short_unnormalized / p_short_unnormalized.sum(axis=0, dtype=float)

        def compute_p_hit(i, j, max_range_px):
            return engs[i][j] if 0.0 <= i <= max_range_px else 0.0

        def compute_p_short(i, j):
            return p_short_normalized[i][j]

        def compute_p_max(i, max_range_px):
            return 1 if i == max_range_px else 0.0

        def compute_p_rand(i, max_range_px):
            return 1.0 / max_range_px if 0 <= i < max_range_px else 0.0

        for i in range(table_width): # observed
            for j in range(table_width): # expected
                p_hit = compute_p_hit(i, j, max_range_px)
                p_short = compute_p_short(i, j)
                p_max = compute_p_max(i, max_range_px)
                p_rand = compute_p_rand(i, max_range_px)

                #print(i, j, max_range_px, LAMBDA_SHORT, p_hit, p_short, p_max, p_rand)

                assert 0 <= p_hit <= 1
                assert 0 <= p_rand <= 1
                assert 0 <= p_short <= 1
                assert 0 <= p_max <= 1

                sensor_model_table[i][j] = Z_HIT*p_hit + Z_SHORT*p_short + Z_MAX*p_max + Z_RAND*p_rand

        column_sums = sensor_model_table.sum(axis=0)
        row_sums = sensor_model_table.sum(axis=1)
        #print("Column Sums: ", column_sums)

        print("SMT:")
        print(sensor_model_table)
        for j in range(table_width):
            assert abs(column_sums[j] - 1.0) <= 0.02

        return sensor_model_table


    def apply_sensor_model(self, proposal_dist, obs, weights):
        obs_ranges = obs[0]
        obs_angles = obs[1]
        num_rays = obs_angles.shape[0]

        # Only allocate buffers once to avoid slowness
        if not isinstance(self.queries, np.ndarray):
            self.queries = np.zeros((proposal_dist.shape[0],3), dtype=np.float32)
            self.ranges = np.zeros(num_rays*proposal_dist.shape[0], dtype=np.float32)

        self.queries[:,:] = proposal_dist[:,:]

        self.range_method.calc_range_repeat_angles(self.queries, obs_angles, self.ranges)

        # Evaluate the sensor model on the GPU
        self.range_method.eval_sensor_model(obs_ranges, self.ranges, weights, num_rays, proposal_dist.shape[0])

        np.power(weights, INV_SQUASH_FACTOR, weights)

if __name__ == '__main__':
    rospy.init_node('SensorTest1', anonymous=True)
    print("Enter main")
    Sensor = SensorModel(None, None, None)
    rospy.spin()
