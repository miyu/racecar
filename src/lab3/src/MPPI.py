#!/usr/bin/env python

import time
import sys
import numpy as np
import utils as Utils

import torch
import torch.utils.data
from torch.autograd import Variable
from InternalMotionModel import InternalKinematicMotionModel

IS_ON_ROBOT = True

if IS_ON_ROBOT:
    import rospy
    import rosbag
    from nav_msgs.srv import GetMap
    from ackermann_msgs.msg import AckermannDriveStamped
    from vesc_msgs.msg import VescStateStamped
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped

CUDA = torch.cuda.is_available()
if CUDA:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

CONTROL_SIZE = 2

MODEL_FILENAME = '/home/nvidia/our_catkin_ws/src/lab3/src/tanh50k.pt'

def wrap_pi_pi_number(x):
    x = np.fmod(x, np.pi * 2) + np.pi * 4
    x = np.fmod(x, np.pi * 2)
    if x > np.pi:
        x -= 2 * np.pi
    return x

def wrap_pi_pi_tensor(x):
    x = torch.fmod(x, np.pi * 2)
    x *= ((0 <= x) * (x < 2*np.pi)).type(dtype)
    x -= (x > np.pi).type(dtype) * 2 * np.pi
    return x

class MPPIController:

  def __init__(self, T, K, sigma=0.5, _lambda=0.5):
    if IS_ON_ROBOT:
        self.SPEED_TO_ERPM_OFFSET = float(rospy.get_param("/vesc/speed_to_erpm_offset", 0.0))
        self.SPEED_TO_ERPM_GAIN   = float(rospy.get_param("/vesc/speed_to_erpm_gain", 4614.0))
        self.STEERING_TO_SERVO_OFFSET = float(rospy.get_param("/vesc/steering_angle_to_servo_offset", 0.5304))
        self.STEERING_TO_SERVO_GAIN   = float(rospy.get_param("/vesc/steering_angle_to_servo_gain", -1.2135))
    else:
        self.SPEED_TO_ERPM_OFFSET = 0.0
        self.SPEED_TO_ERPM_GAIN   = 4614.0
        self.STEERING_TO_SERVO_OFFSET = 0.5304
        self.STEERING_TO_SERVO_GAIN   = -1.2135

    self.CAR_LENGTH = 0.33

    self.last_pose = None
    # MPPI params
    self.T = T # Length of rollout horizon
    self.K = K # Number of sample rollouts
    self.sigma = sigma
    self._lambda = _lambda

    self.goal = None # Lets keep track of the goal pose (world frame) over time
    self.lasttime = None
    self.last_control = None

    # PyTorch / GPU data configuration
    # TODO
    # you should pre-allocate GPU memory when you can, and re-use it when
    # possible for arrays storing your controls or calculated MPPI costs, etc
    if CUDA:
        self.model = torch.load(MODEL_FILENAME)
    else:
        self.model = torch.load(MODEL_FILENAME, map_location=lambda storage, loc: storage) # Maps CPU storage and serialized location back to CPU storage


    sigma_data = [[1.0, 0.2], [0.2, 1.0]]

    if CUDA:
        self.model.cuda() # tell torch to run the network on the GPU
        self.dtype = torch.cuda.FloatTensor

        self.Sigma = torch.cuda.FloatTensor(sigma_data) # Covariance Matrix shape: (CONTROL_SIZE, CONTROL_SIZE)
        self.SigmaInv = torch.inverse(self.Sigma)
        self.U = torch.cuda.FloatTensor(CONTROL_SIZE, self.K, self.T).zero_()
        self.Epsilon = torch.cuda.FloatTensor(CONTROL_SIZE, self.K, self.T).zero_()
        self.Trajectory_cost = torch.cuda.FloatTensor(1, self.K).zero_()
    else:
        self.dtype = torch.FloatTensor

        self.Sigma = torch.FloatTensor(sigma_data) # Covariance Matrix shape: (CONTROL_SIZE, CONTROL_SIZE)
        self.SigmaInv = torch.inverse(self.Sigma)
        self.U = torch.FloatTensor(CONTROL_SIZE, self.K, self.T).zero_()
        self.Epsilon = torch.FloatTensor(CONTROL_SIZE, self.K, self.T).zero_()
        self.Trajectory_cost = torch.FloatTensor(1, self.K).zero_()

    print("Loading:", MODEL_FILENAME)
    print("Model:\n",self.model)
    print("Torch Datatype:", self.dtype)

    # control outputs
    self.msgid = 0

    # Store last control input
    self.last_control = None
    # visualization parameters
    self.num_viz_paths = 40
    if self.K < self.num_viz_paths:
        self.num_viz_paths = self.K

    if IS_ON_ROBOT:
        # We will publish control messages and a way to visualize a subset of our
        # rollouts, much like the particle filter
        self.ctrl_pub = rospy.Publisher(rospy.get_param("~ctrl_topic", "/vesc/high_level/ackermann_cmd_mux/input/nav0"),
                AckermannDriveStamped, queue_size=2)
        self.path_pub = rospy.Publisher("/mppi/paths", Path, queue_size = self.num_viz_paths)

        # Use the 'static_map' service (launched by MapServer.launch) to get the map
        map_service_name = rospy.get_param("~static_map", "static_map")
        print("Getting map from service: ", map_service_name)
        rospy.wait_for_service(map_service_name)
        map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map # The map, will get passed to init of sensor model
        self.map_info = map_msg.info # Save info about map for later use
        print("Map Information:\n",self.map_info)

        # Create numpy array representing map for later use
        self.map_height = map_msg.info.height
        self.map_width = map_msg.info.width
        array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
        self.permissible_region = np.zeros_like(array_255, dtype=bool)
        self.permissible_region[array_255==0] = 1 # Numpy array of dimension (map_msg.info.height, map_msg.info.width),
                                                  # With values 0: not permissible, 1: permissible
        self.permissible_region = np.negative(self.permissible_region) # 0 is permissible, 1 is not

        print("Making callbacks")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal",
                PoseStamped, self.clicked_goal_cb, queue_size=1)
        self.pose_sub  = rospy.Subscriber("/pf/viz/inferred_pose",
                PoseStamped, self.mppi_cb, queue_size=1)


  def update_lambda(self, new_lambda):
    self._lambda = new_lambda

  # TODO
  # You may want to debug your bounds checking code here, by clicking on a part
  # of the map and convincing yourself that you are correctly mapping the
  # click, and thus the goal pose, to accessible places in the map
  def clicked_goal_cb(self, msg):
    self.goal = np.array([msg.pose.position.x,
                          msg.pose.position.y,
                          Utils.quaternion_to_angle(msg.pose.orientation)])
    print("Current Pose: ", self.last_pose)
    print("SETTING Goal: ", self.goal)

  def running_cost(self, pose, goal, ctrl=None, noise=None):
    # TODO
    # This cost function drives the behavior of the car. You want to specify a
    # cost function that penalizes behavior that is bad with high cost, and
    # encourages good behavior with low cost.
    # We have split up the cost function for you to a) get the car to the goal
    # b) avoid driving into walls and c) the MPPI control penalty to stay
    # smooth
    # You should feel free to explore other terms to get better or unique
    # behavior
    pose_cost = 0.0
    bounds_check = 0.0
    ctrl_cost = 0.0 # We can tweak this later

    # print('First Pose: ', pose[0, :])
    # print('Goal: ', goal)

    dx = pose[:, 0] - goal[0]
    dy = pose[:, 1] - goal[1]
    dtheta = wrap_pi_pi_tensor(pose[:, 2] - goal[2])
    dtheta *= 0.1

    distance = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))

    pose_cost = torch.sqrt(torch.pow(distance, 2) + torch.pow(dtheta, 2))

    # print("and costs:", pose_cost)

    grid_poses = pose.clone() #.cpu().numpy()
    if IS_ON_ROBOT:
        Utils.world_to_map(grid_poses, self.map_info)
        print('Grid Poses: ', grid_poses)
        permissibles = self.permissible_region[grid_poses[:, 0], grid_poses[:, 1]]
        bounds_check = torch.from_numpy(permissibles.astype(float) * 1E5).type(self.dtype)
        print('Permissibles Size: ', permissibles.shape)
        # print('Permissible Region: ', self.permissible_region)
        # print(temp_pose, self.map_info)
        # if self.permissible_region[int(temp_pose[0,0])][int(temp_pose[0,1])]:
        #     bounds_check = 1.0e5

    return pose_cost + ctrl_cost + bounds_check

  def mppi(self, init_pose, neural_net_input):
    t0 = time.time()
    # Network input can be:
    #   0    1       2          3           4        5      6   7
    # xdot, ydot, thetadot, sin(theta), cos(theta), vel, delta, dt

    # MPPI should
    # generate noise according to sigma
    # combine that noise with your central control sequence
    # Perform rollouts with those controls from your current pose
    # Calculate costs for each of K trajectories
    # Perform the MPPI weighting on your calculated costs
    # Scale the added noise by the weighting and add to your control sequence
    # Apply the first control values, and shift your control trajectory

    # Notes:
    # MPPI can be assisted by carefully choosing lambda, and sigma
    # It is advisable to clamp the control values to be within the feasible range
    # of controls sent to the Vesc
    # Your code should account for theta being between -pi and pi. This is
    # important.
    # The more code that uses pytorch's cuda abilities, the better; every line in
    # python will slow down the control calculations. You should be able to keep a
    # reasonable amount of calculations done (T = 40, K = 2000) within the 100ms
    # between inferred-poses from the particle filter.

    if CUDA:
        poses = torch.cuda.FloatTensor(self.T, 3).zero_()
        self.Trajectory_cost = torch.cuda.FloatTensor(1, self.K).zero_()
    else:
        poses = torch.FloatTensor(self.T, 3).zero_()
        self.Trajectory_cost = torch.FloatTensor(1, self.K).zero_()
    self.Epsilon.normal_(std=self.sigma)
    # Epsilon shape: (CONTROL_SIZE, K, T)
    noisyU = self.U + self.Epsilon # self.U -> All K samples SHOULD BE IDENTICAL
    # noisyU shape: (CONTROL_SIZE, K, T)

    neural_net_input = np.tile(neural_net_input, (self.K, 1)) # We should convert these numpy operations to CUDA later
    # neural_net_input shape: (8) -> (K, 8)

    neural_net_input_torch = torch.from_numpy(neural_net_input.astype('float32')).type(self.dtype)
    # neural_net_input_torch shape: (K, 8)

    x_tminus1 = torch.from_numpy(np.tile(init_pose, (self.K, 1)).astype('float32')).type(self.dtype)
    # x_tminus1 shape: (K, 3)

    for t in range(1, self.T):
        #print('Neural Net Input Size: ', neural_net_input_torch.size())
        #print('Noisy U Size:', noisyU.size())
        neural_net_input_torch[:, 5] = noisyU[0, :, t-1]
        neural_net_input_torch[:, 6] = noisyU[1, :, t-1]

        neural_net_output = self.model(Variable(neural_net_input_torch))
        # neural_net_output shape: (K, 3)

        x_t = x_tminus1 + neural_net_output.data
        # x_t shape: (K, 3)

        # print("@t", t, x_t)

        if CUDA:
            u_tminus1 = self.U[:,0,t-1].view(1, CONTROL_SIZE)
        else:
            u_tminus1 = self.U[:,0,t-1].contiguous().view(1, CONTROL_SIZE)
        # u_tminus1 shape: (1, CONTROL_SIZE)
        intermediate = torch.mm(u_tminus1, self.SigmaInv)
        # self.Sigma shape: (CONTROL_SIZE, CONTROL_SIZE)

        # intermediate shape: (1, CONTROL_SIZE)

        intermediate = self._lambda * torch.mm(intermediate, self.Epsilon[:,:,t-1])
        # self.Epsilon[:,:,t-1] shape: (CONTROL_SIZE, K)
        # intermediate shape: (1, K)
        # Lambda: Scalar

        current_cost = self.running_cost(x_t, self.goal).view(1, self.K)
        # print('COST: ', current_cost)
        self.Trajectory_cost += current_cost + intermediate
        #print('Current Cost Size: ', current_cost.size())

        # running_cost: want this to be (1, K)
        # self._lambda * intermediate: (1, K)
        # self.Trajectory_cost shape: (1, K)

        x_tminus1 = x_t

    beta = torch.min(self.Trajectory_cost)
    #print('Beta: ', beta)
    trajectoryMinusMin = self.Trajectory_cost - beta
    trajectoryMinusMin *= (-1.0 / self._lambda)
    #print('Trajectory - Beta: ', trajectoryMinusMin)
    n = torch.sum(torch.exp(trajectoryMinusMin))
    #print('n: ', n)
    omega = (1.0/ n) * torch.exp(trajectoryMinusMin)
    #print('omega: ', omega)
    # omega shape: (1, K)

    for t in range(self.T):
        omega = omega.expand(CONTROL_SIZE, -1) # Check this!
        #print('Omega shape: ', omega.size())
        #print('Omega: ', omega)
        # omega shape: (CONTROL_SIZE, K)
        delta_control = torch.sum(omega * self.Epsilon[:,:,t], dim=1).view(CONTROL_SIZE, 1)
        #print('Delta Control: ', delta_control)
        #print('Delta control shape: ', delta_control.size())
        # self.Epsilon[:, :, t] shape: (CONTROL_SIZE, K)
        # delta_control shape: (CONTROL_SIZE, 1)
        self.U[:, :, t] += delta_control
        # self.U[:, :, t] shape: (CONTROL_SIZE, K)

    # print("Validate U:", self.U)

    controls = noisyU[:, :, 0]
    # print("Controls:", controls)
    # print("Trajectory costs:", self.Trajectory_cost)
    best_cost, best_index = torch.min(self.Trajectory_cost, 1)
    # print("Best index: ", best_index, "has cost", best_cost)
    run_ctrl = controls[:, best_index]
    # print("Which is control", run_ctrl)
    # run_ctrl shape: (CONTROL_SIZE)

    # print("MPPI: %4.5f ms" % ((time.time()-t0)*1000.0))

    return run_ctrl, poses

  # Reads Particle Filter Messages
  # ALSO do we need to make sure our Thetas are between -pi and pi???????
  def mppi_cb(self, msg):
    new_lambda = mp._lambda * 0.99 # This wasn't in skeleton code: Decay Lambda
    mp.update_lambda(new_lambda) # This wasn't in skeleton code: Decay Lambda
    print('New Lambda: ', mp._lambda) # This wasn't in skeleton code: Decay Lambda

    if self.last_pose is None:
      self.last_pose = np.array([msg.pose.position.x,
                                 msg.pose.position.y,
                                 Utils.quaternion_to_angle(msg.pose.orientation)])
      # Default: initial goal to be where the car is when MPPI node is
      # initialized
      self.goal = self.last_pose
      self.lasttime = msg.header.stamp.to_sec()
      return

    theta = Utils.quaternion_to_angle(msg.pose.orientation)
    curr_pose = np.array([msg.pose.position.x,
                          msg.pose.position.y,
                          theta])

    pose_dot = curr_pose - self.last_pose # get state
    pose_dot[2] = wrap_pi_pi_number(pose_dot[2]) # This was not in skeleton code: Clamp Theta between -pi and pi
    self.last_pose = curr_pose

    timenow = msg.header.stamp.to_sec()
    dt = timenow - self.lasttime
    self.lasttime = timenow
    nn_input = np.array([pose_dot[0], pose_dot[1], pose_dot[2],
                         np.sin(theta),
                         np.cos(theta), 0.0, 0.0, dt])

    run_ctrl, poses = self.mppi(curr_pose, nn_input)
    run_ctrl = run_ctrl.view(CONTROL_SIZE)
    #run_ctrl_np = run_ctrl.cpu().numpy().reshape((2,))

    self.U[:,:,0:self.T-1] = self.U[:,:,1:self.T]
    self.U[:,:,self.T-1] = 0.0

    self.send_controls( run_ctrl[0], run_ctrl[1] )


    self.visualize(poses)

  def send_controls(self, speed, steer):
    print("Speed:", speed, "Steering:", steer)
    ctrlmsg = AckermannDriveStamped()
    ctrlmsg.header.seq = self.msgid
    ctrlmsg.drive.steering_angle = steer
    ctrlmsg.drive.speed = speed
    self.ctrl_pub.publish(ctrlmsg)
    self.msgid += 1

  # Publish some paths to RVIZ to visualize rollouts
  def visualize(self, poses):
    if self.path_pub.get_num_connections() > 0:
      frame_id = 'map'
      for i in range(0, self.num_viz_paths):
        pa = Path()
        pa.header = Utils.make_header(frame_id)
        pa.poses = map(Utils.particle_to_posestamped, poses[i,:,:], [frame_id]*self.T)
        self.path_pub.publish(pa)

def test_MPPI(mp, N, goal=np.array([0.,0.,0.])):
  init_input = np.array([0.,0.,0.,0.,1.,0.,0.,0.])
  pose = np.array([0.,0.,0.])
  mp.goal = goal
  print("Start:", pose)
  mp.ctrl.zero_()
  last_pose = np.array([0.,0.,0.])
  for i in range(0,N):
    # ROLLOUT your MPPI function to go from a known location to a specified
    # goal pose. Convince yourself that it works.

    print("Now:", pose)
  print("End:", pose)

def small_test_MPPI(mp, motion_model, i):
  new_lambda = mp._lambda * 0.99 # * np.exp(-i / 4.0)
  mp.update_lambda(new_lambda)
  print('New Lambda: ', mp._lambda)
  if mp.last_pose is None:
    mp.last_pose = np.array([0., 0., 0.])
    mp.lasttime = time.time()
    return

  curr_pose = motion_model.particles[0]
  theta = motion_model.particles[0][2]

  pose_dot = curr_pose - mp.last_pose # get state
  mp.last_pose = curr_pose

  dt = 0.1#time.time() - mp.lasttime
  mp.lasttime = time.time()

  pose_dot[2] = wrap_pi_pi_number(pose_dot[2])

  # print("Pose dot: ", pose_dot)

  if True:#mp.last_control is None: TURNS OUT MOVING WITH PURE NOISE WORKS A LOT BETTER
      nn_input = np.array([pose_dot[0], pose_dot[1], pose_dot[2], np.sin(theta), np.cos(theta), 0.0, 0.0, dt])
  else:
      nn_input = np.array([pose_dot[0], pose_dot[1], pose_dot[2], np.sin(theta), np.cos(theta), mp.last_control[0], mp.last_control[1], dt])

  # print("NN input", nn_input)

  run_ctrl, poses = mp.mppi(curr_pose, nn_input)
  run_ctrl = run_ctrl.view(CONTROL_SIZE)

  print("Decided control", run_ctrl)

  mp.U[:,:,0:mp.T-1] = mp.U[:,:,1:mp.T]
  mp.U[:,:,mp.T-1] = 0.0

  motion_model.update([run_ctrl[0], run_ctrl[1], dt]) # Speed, Steering, dt
  mp.last_control = run_ctrl

  print("Moved with control", run_ctrl, "to", motion_model.particles[0], " ", wrap_pi_pi_number(motion_model.particles[0][2]))

if __name__ == '__main__':
  if CUDA:
    print('CUDA is available')
  else:
    print('CUDA is NOT available')

  T = 5
  K = 10000
  sigma = 0.5 # These values will need to be tuned
  _lambda = 1e-4 # 1.0

  if IS_ON_ROBOT:
    rospy.init_node("mppi_control", anonymous=True) # Initialize the node
    mp = MPPIController(T, K, sigma, _lambda)
    rospy.spin()
  else:
    # test & DEBUG
    mp = MPPIController(T, K, sigma, _lambda)
    if CUDA:
        mp.goal = torch.cuda.FloatTensor([2., 2., 0.])
    else:
        mp.goal = torch.FloatTensor([2., 2., 0.])
    nparticles = 1#1000
    particles = np.zeros((nparticles, 3), dtype=float)
    motion_model = InternalKinematicMotionModel(particles, np.array([[0.0, 0.003], [0.0, 0.003]]), useNoise=False)
    i = 1
    while(True):
        small_test_MPPI(mp, motion_model, i)
        i += 1
    #test_MPPI(mp, 10, np.array([0.,0.,0.]))
