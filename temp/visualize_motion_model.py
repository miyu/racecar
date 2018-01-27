import numpy as np
import matplotlib as plt



class OdometryMotionModel:
    def __init__(self, intial_pose, particles , state_lock=None):
        self.last_pose = inital_pose # The last pose thatwas received
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
            x2 = x1 + .02
            y2  =y1
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

	#self.last_pose = pose
	#self.state_lock.release()

    def apply_motion_model(self, proposal_dist, control):
    	# Update the proposal distribution by applying the control to each particle
    	# YOUR CODE HERE
    	# pdist has dim MAX_PARTICLES x 3 => Individual particle is 1 x 3.
    	# result should be dim MAX_PARTICLES x 3 => result particle is 1 x 3.
    	# Hence, control should be 1 x 3. => Dot product
    	control = np.reshape(control, (1, 3))
    	noise = np.array([np.random.normal(0, var) for var in [0.01, 0.01, 0.05]], dtype=np.float64)
    	#print("In apply_motion_model, control is  ", control)
    	#print("Proposal dist", proposal_dist)

    	self.particles[:][:] = np.array(proposal_dist + control + noise, dtype=np.float64)


if __name__=="__main__":
    initial_pose = np.array([1,1,0]);
    particles = np.repeat(initial_pose,1000, axis=1)

    print(particles)
