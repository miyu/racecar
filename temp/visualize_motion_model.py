import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import math


class OdometryMotionModel:
    def __init__(self, initial_pose, particles , state_lock=None):
        self.last_pose = initial_pose # The last pose thatwas received
    	self.particles = particles if particles is not None else np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    	#self.sub = rospy.Subscriber("/vesc/odom",Odometry, self.motion_cb )
    	#if state_lock is None:
        #    self.state_lock = Lock()
        #else:
	#        self.state_lock = state_lock

    def motion_cb(self, msg):
        # # uncomment to no-op
        # return

        #self.state_lock.acquire()

    	pose = None
    	control = None
    	if isinstance(self.last_pose, np.ndarray):
            print("A!")
            # Compute the control from the msg and last_pose
            # YOUR CODE HERE
            # reference frame pose position [x,y,z]-> starting position of car, z is static (2D motion)
            # pose orientation [x,y,z,w]. x, y are 0. z^2 + w^2 = 1.
            # difference in orientation between last pose and current?
            #print(self.last_pose);
            x1 = self.last_pose[0]
            y1 = self.last_pose[1]
            theta1 = self.last_pose[2]

            #x2 = msg.pose.pose.position.x
            #y2 = msg.pose.pose.position.y

            #x_o = msg.pose.pose.orientation.x
            #y_o = msg.pose.pose.orientation.y
            #z_o = msg.pose.pose.orientation.z
            #w_o = msg.pose.pose.orientation.w

            #angle = euler_from_quaternion([x_o, y_o, z_o, w_o])
            #theta2 = angle[2]

            # Just moving forward
            x2 = x1 + .5
            y2  =y1+.5
            theta2 = theta1
            
            pose = np.array([x2, y2, theta2], dtype=np.float64)
            control = np.array([x2 - x1, y2 - y1, theta2 - theta1], dtype=np.float64)

            # print("Control in if ", control)
        else:
            print("B!")
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y

            x_o = msg.pose.pose.orientation.x
            y_o = msg.pose.pose.orientation.y
            z_o = msg.pose.pose.orientation.z
            w_o = msg.pose.pose.orientation.w

            angle = euler_from_quaternion([x_o, y_o, z_o, w_o])
            '''
            print("x: ",x);
            print("y: ", y);
            print("x_o", x_o)
            print("y_o", y_o)
            print("z_o", z_o)
            print("w_o", w_o)
            print("delta: ", angle[2]);
            '''
            pose = np.array([x,y,angle[2]], dtype=np.float64);

            control = np.array([0, 0, 0], dtype=np.float64)

            #print("Conrtol in else: ", control)

	#print("Check")

	self.apply_motion_model(self.particles, control)

        self.plot_particles()
	#self.last_pose = pose
	#self.state_lock.release()

    def apply_motion_model(self, proposal_dist, control):
    	# Update the proposal distribution by applying the control to each particle
    	# YOUR CODE HERE
    	# pdist has dim MAX_PARTICLES x 3 => Individual particle is 1 x 3.
    	# result should be dim MAX_PARTICLES x 3 => result particle is 1 x 3.
    	# Hence, control should be 1 x 3. => Dot product
    	control = np.reshape(control, (1, 3))
        print("Control: ",control)
        print("self.particles: ", self.particles[0])
        noise = np.array([np.random.normal(0, var) for var in [0.01, 0.01, 0.05]], dtype=np.float64)
        print("new: ", np.array(proposal_dist + control + noise, dtype=np.float64)) 
        for i in range(len(proposal_dist)):
    	    noise = np.array([np.random.normal(0, var) for var in [0.01, 0.01, 0.05]], dtype=np.float64)
    	    #print("In apply_motion_model, control is  ", control)
    	    #print("Proposal dist", proposal_dist)
    	    self.particles[i] = np.array(proposal_dist[i] + control + noise, dtype=np.float64)

    def plot_particles(self):        
        pre_x = self.particles[:,0]
        pre_y = self.particles[:,1]
        #print("predeicted x: ",pre_x)
        #print("predeicted y: ",pre_y)
        plt.scatter(pre_x, pre_y)
        plt.scatter(self.last_pose[0], self.last_pose[1])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

class KinematicMotionModel:
    def __init__(self, initial_pose,  particles=None, state_lock=None):
        #self.last_servo_cmd = None # The most recent servo command
        #self.last_vesc_stamp = None # The time stamp from the previous vesc state msg
        self.particles = particles if particles is not None else np.array([[0, 0, 0]], dtype=np.float64)
        self.initial_pose = initial_pose
        #self.SPEED_TO_ERPM_OFFSET = float(rospy.get_param("/vesc/speed_to_erpm_offset")) # Offset conversion param from rpm to speed
        #self.SPEED_TO_ERPM_GAIN   = float(rospy.get_param("/vesc/speed_to_erpm_gain"))   # Gain conversion param from rpm to speed
        #self.STEERING_TO_SERVO_OFFSET = float(rospy.get_param("/vesc/steering_angle_to_servo_offset")) # Offset conversion param from servo position to steering angle
        #self.STEERING_TO_SERVO_GAIN   = float(rospy.get_param("/vesc/steering_angle_to_servo_gain")) # Gain conversion param from servo position to steering angle

        #if state_lock is None:
        #    self.state_lock = Lock()
        #else:
        #    self.state_lock = state_lock

        # This subscriber just caches the most recent servo position command
        #self.servo_pos_sub = rospy.Subscriber(rospy.get_param("~servo_pos_topic", "/vesc/sensors/servo_position_command"), Float64,self.servo_cb, queue_size=1)

        #To get velocity
        #self.motion_vel_sub = rospy.Subscriber(rospy.get_param("~vel", "/vesc/sensors/core"), VescStateStamped, self.motion_cb)

    def servo_cb(self, msg):
        self.last_servo_cmd = msg.data # Just update servo command

    def motion_cb(self, msg):
        # # uncomment to no-op
        # return

        #print("Entering lock motion_cb")
        #self.state_lock.acquire()
        #print("Entered lock motion_cb")

        # if no servo info, skip
        #if self.last_servo_cmd is None:
        #    print("Releasing lock motion_cb early (no servo command)")
        #    self.state_lock.release()
        #    return

        # if no prev timestamp, then play as if dt is 0
        #if self.last_vesc_stamp is None:
        #    self.last_vesc_stamp = msg.header.stamp

        #dt = (msg.header.stamp - self.last_vesc_stamp).to_sec()
        #self.last_vesc_stamp = msg.header.stamp

        # Convert raw msgs to controls
        # Note that control = (raw_msg_val - offset_param) / gain_param
        # YOUR CODE HERE

        #curr_speed = float(float(msg.state.speed) - float(self.SPEED_TO_ERPM_OFFSET)) / float(self.SPEED_TO_ERPM_GAIN)
        #curr_steering_angle = float(float(self.last_servo_cmd) - float(self.STEERING_TO_SERVO_OFFSET)) / float(self.STEERING_TO_SERVO_GAIN)

        curr_speed = 1  # Move by m/s
        curr_steering_angle = 0   # The steering angle
        dt = .2 # 50 Hz
        
        #print("Velocity ", curr_speed)
        #print("Steering Angle: ", curr_steering_angle, " -- ", self.last_servo_cmd, " -- ", self.STEERING_TO_SERVO_OFFSET)
        #print("Delta Time: ", dt)

        self.apply_motion_model(self.particles, [curr_speed, curr_steering_angle, dt])
        #print("Releasing lock motion_cb")
        #self.state_lock.release()

    def apply_motion_model(self, proposal_dist, control):
        # Update the proposal distribution by applying the control to each particle
        # YOUR CODE HERE
        for i in range(len(proposal_dist)):
            noise = np.array([np.random.normal(0, var) for var in [.0001,1, .00001]], dtype=np.float64)
            [curr_speed, curr_steering_angle, dt] = control + noise

            [x,y,theta] = proposal_dist[i]
            dx = (curr_speed) * math.cos(theta) # added - shit was flipped
            dy = (curr_speed) * math.sin(theta)
            beta = math.atan(math.tan(curr_steering_angle) / 2.0)
            dtheta = (curr_speed / 0.33) * math.sin(2.0 * beta) # Length = 0.33 m

            

            #self.particles[i] = proposal_dist[i] + np.array([dx * dt, dy * dt, dtheta * dt], dtype=np.float64)

            L = .33   # length of the car
            prior_theta = self.particles[i][2]
            theta = (curr_speed/L) * (np.sin(2*beta))  *dt + prior_theta
            x = (L/np.sin(2*beta)) * (np.sin(theta) - np.sin(prior_theta)) + self.particles[i][0]
            y = (L/np.sin(2*beta)) * (np.cos(prior_theta) - np.cos(theta)) + self.particles[i][1]
            self.particles[i] = np.array([x, y, theta])
        
                                     

        self.plot_particles()
            #print("DELTA: ", np.array([dx, dy, dtheta], dtype=float) * dt)
            #print("  RES: ", self.particles[i])

    def plot_particles(self):
        pred_x = self.particles[:,0]
        pred_y = self.particles[:,1]
        #print("predeicted x: ",pre_x)
        #print("predeicted y: ",pre_y)
        plt.scatter(pred_x, pred_y)
        plt.scatter(self.initial_pose[0], self.initial_pose[1])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        

        
if __name__=="__main__":
    initial_pose = np.array([1,1,0], dtype=np.float64)
    #particles = np.repeat(initial_pose,1000, axis=1)
    particles = np.tile(initial_pose, (1000, 1))
    
    print(particles)
    #odo = OdometryMotionModel(initial_pose, particles)
    #odo.motion_cb(None)

    kin = KinematicMotionModel(initial_pose, particles)
    kin.motion_cb(None)
