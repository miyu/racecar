import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import math


def rotate_2d(x, y, theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return (c * x - s * y, s * x + c * y)

class OdometryMotionModel:
    def __init__(self, initial_pose, particles , state_lock=None):
        self.last_pose = initial_pose # The last pose thatwas received
    	self.particles = particles if particles is not None else np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    	
    def motion_cb(self, msg):
    	pose = None
    	control = None
    	if isinstance(self.last_pose, np.ndarray):
            x1 = self.last_pose[0]
            y1 = self.last_pose[1]
            theta1 = self.last_pose[2]

            # Just moving forward
            x2 = x1 + .5
            y2  =y1+.5
            theta2 = theta1
            
            pose = np.array([x2, y2, theta2], dtype=np.float64)
            control = np.array([x2 - x1, y2 - y1, theta2 - theta1], dtype=np.float64)
	self.apply_motion_model(self.particles, control)
        self.plot_particles()

    def apply_motion_model(self, proposal_dist, control):
    	# Update the proposal distribution by applying the control to each particle
    	# YOUR CODE HERE
    	# pdist has dim MAX_PARTICLES x 3 => Individual particle is 1 x 3.
    	# result should be dim MAX_PARTICLES x 3 => result particle is 1 x 3.
    	# Hence, control should be 1 x 3. => Dot product
        num_particles = proposal_dist.shape[0]
        base_dx, base_dy, dtheta = control
        for i in range(num_particles):
            cx, cy, ctheta = proposal_dist[i][0], proposal_dist[i][1], proposal_dist[i][2]
            applied_dx = base_dx + np.random.normal(0, abs(base_dx * 0.25) + 0.03)
            applied_dy = base_dy + np.random.normal(0, abs(base_dy * 0.25) + 0.03)
            applied_dtheta = dtheta + np.random.normal(0, abs(dtheta * 0.2) + 0.03)
            rx, ry = rotate_2d(applied_dx, applied_dy, ctheta)
            self.particles[i][0] = cx + rx
            self.particles[i][1] = cy + ry
            self.particles[i][2] = ctheta + applied_dtheta

    def plot_particles(self):        
        pre_x = self.particles[:,0]
        pre_y = self.particles[:,1]
        plt.scatter(pre_x, pre_y)
        plt.scatter(self.last_pose[0], self.last_pose[1])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

class KinematicMotionModel:
    def __init__(self, initial_pose,  particles=None, state_lock=None):
        self.particles = particles if particles is not None else np.array([[0, 0, 0]], dtype=np.float64)
        self.initial_pose = initial_pose

    def servo_cb(self, msg):
        self.last_servo_cmd = msg.data # Just update servo command

    def motion_cb(self, msg):
        curr_speed = 5  # Move by m/s
        curr_steering_angle = .02   # The steering angle
        dt = .2 # 50 Hz
        
        self.apply_motion_model(self.particles, [curr_speed, curr_steering_angle, dt])
        
    def apply_motion_model(self, proposal_dist, control):
        # Update the proposal distribution by applying the control to each particle
        # YOUR CODE HERE
        noise_speed = .2
        noise_steering = .05
        """
        for i in range(len(proposal_dist)):
            noise = np.array([np.random.normal(0, var) for var in [noise_speed,noise_steering, .00001]], dtype=np.float64)
            [curr_speed, curr_steering_angle, dt] = control + noise

            beta = math.atan(math.tan(curr_steering_angle) / 2.0)
            dtheta = (curr_speed / 0.33) * math.sin(2.0 * beta) # Length = 0.33 m
       

            #self.particles[i] = proposal_dist[i] + np.array([dx * dt, dy * dt, dtheta * dt], dtype=np.float64)
        
            L = .33   # length of the car
            prior_theta = self.particles[i][2]
            theta = (curr_speed/L) * (np.sin(2*beta))  *dt + prior_theta
            x = (L/np.sin(2*beta)) * (np.sin(theta) - np.sin(prior_theta)) + self.particles[i][0]
            y = (L/np.sin(2*beta)) * (np.cos(prior_theta) - np.cos(theta)) + self.particles[i][1]
            self.particles[i] = np.array([x, y, theta])
        """
        
        num_particles = proposal_dist.shape[0]

        # TODO: consider applying same control noise to all particles.
        control_speeds = np.random.normal(0, noise_speed, size=num_particles) + control[0]
        control_steerings = np.random.normal(0, noise_steering, size=num_particles) + control[1]
        dt = control[2]
    
        beta = np.arctan(np.tan(control_steerings) / 2.0)
        dtheta = (control_speeds / 0.33) * np.sin(2.0 * beta) #0.33 is len between car wheels front/back
        self.particles[:,2] += dtheta * dt

        current_thetas = self.particles[:,2]
        dx = control_speeds * np.cos(current_thetas)
        dy = control_speeds * np.sin(current_thetas)
        
        self.particles[:, 0] += dx * dt
        self.particles[:, 1] += dy * dt
        
        self.plot_particles(control[1])

    
    def plot_particles(self, steering_angle):
        pred_x = self.particles[:,0]
        pred_y = self.particles[:,1]
        plt.scatter(pred_x, pred_y)
        plt.scatter(self.initial_pose[0], self.initial_pose[1])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Left turn by " + str(steering_angle) + " rad")
        plt.show()
        

        
if __name__=="__main__":
    initial_pose = np.array([0,0,0], dtype=np.float64)
    particles = np.tile(initial_pose, (1000, 1))
    #print(particles)
    odo = OdometryMotionModel(initial_pose, particles)
    odo.motion_cb(None)
    #kin = KinematicMotionModel(initial_pose, particles)
    #kin.motion_cb(None)
