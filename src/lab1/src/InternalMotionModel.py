import numpy as np
import math

from TorchInclude import torch, FloatTensor, Variable

def rotate_2d(x, y, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return (c * x - s * y, s * x + c * y)

def np_array_or(x, y):
    return x if x is not None else y

class InternalOdometryMotionModel:
    def __init__(self, particles, initial_pose, noise_params=None):
        self.last_poses = initial_pose
        self.particles = particles
        #self.noise_params = np_array_or(noise_params,np.array([[0.1, 0.03], [0.1, 0.03], [0.3, 0.03]]))
        self.noise_params = np_array_or(noise_params,np.array([[0.0, 0.3], [0.0, 0.3], [0.0, 0.3]]))


    def update(self, pose):
        # find delta between last pose in odometry-space
        x1, y1, theta1 = self.last_poses
        x2, y2, theta2 = pose
        self.last_poses = np.array(pose)

        # transform odometry-space delta to local-relative-space delta
        local_relative_dx, local_relative_dy = rotate_2d(x2 - x1, y2 - y1, -theta1)
        dtheta = theta2 - theta1

        # derive control parameters
        control = [local_relative_dx, local_relative_dy, dtheta]
        #print("ROBOT AT", x2, y2, "theta", theta2, "LR", local_relative_dx, local_relative_dy, dtheta)
        self.apply_motion_model(self.particles, control)

    def apply_motion_model(self, proposal_dist, control):
        # Update the proposal distribution by applying the control to each particle
        # YOUR CODE HERE
        # pdist has dim MAX_PARTICLES x 3 => Individual particle is 1 x 3.
        # result should be dim MAX_PARTICLES x 3 => result particle is 1 x 3.
        # Hence, control should be 1 x 3. => Dot product

        # get control params
        num_particles = proposal_dist.shape[0]
        base_local_dx, base_local_dy, base_dtheta = control
        local_dxs = base_local_dx + np.random.normal(0, abs(base_local_dx * self.noise_params[0][0]) + self.noise_params[0][1], num_particles)
        local_dys = base_local_dy + np.random.normal(0, abs(base_local_dy * self.noise_params[1][0]) + self.noise_params[1][1], num_particles)
        local_dthetas = base_dtheta + np.random.normal(0, abs(base_dtheta * self.noise_params[2][0]) + self.noise_params[2][1], num_particles)

        # update thetas before xs and ys
        self.particles[:, 2] += local_dthetas

        # update xs, ys
        current_thetas = self.particles[:, 2]
        rxs, rys = rotate_2d(local_dxs, local_dys, current_thetas)
        self.particles[:, 0] += rxs
        self.particles[:, 1] += rys

        # for i in range(num_particles):
        #     cx, cy, ctheta = proposal_dist[i]
        #     applied_dx = local_dx + np.random.normal(0, abs(local_dx * self.noise_params[0][0]) + self.noise_params[0][1])
        #     applied_dy = local_dy + np.random.normal(0, abs(local_dy * self.noise_params[1][0]) + self.noise_params[1][1])
        #     applied_dtheta = dtheta + np.random.normal(0, abs(dtheta * self.noise_params[2][0]) + self.noise_params[2][1])
        #     rx, ry = rotate_2d(applied_dx, applied_dy, ctheta)
        #     self.particles[i][0] = cx + rx
        #     self.particles[i][1] = cy + ry
        #     self.particles[i][2] = ctheta + applied_dtheta

class InternalKinematicMotionModel:
    def __init__(self, particles, noise_params=None):
        self.particles = particles
        self.noise_params = np_array_or(noise_params, np.array([[0.1, 0.1], [0.1, 0.3]]))

    def update(self, control):
        num_particles = self.particles.shape[0]

        # get control params
        control_speeds = np.random.normal(0, abs(control[0] * self.noise_params[0][0]) + self.noise_params[0][1], size=num_particles) + control[0]
        control_steerings = np.random.normal(0, abs(control[1] * self.noise_params[1][0]) + self.noise_params[1][1], size=num_particles) + control[1]
        #control_steerings = control[1]
        #control_speeds = control[0]
        dt = control[2]

        # update thetas before xs and ys
        beta = np.arctan(np.tan(control_steerings) / 2.0)
        dtheta = (control_speeds / 0.33) * np.sin(2.0 * beta) #0.33 is len between car wheels front/back
        self.particles[:, 2] += dtheta * dt

        # update xs, ys
        current_thetas = self.particles[:, 2]
        dx = control_speeds * np.cos(current_thetas)
        dy = control_speeds * np.sin(current_thetas)
        self.particles[:, 0] += dx * dt
        self.particles[:, 1] += dy * dt


class InternalLearnedMotionModel:
    def __init__(self, particles, prediction_model):
        self.particles = particles
        self.prediction_model = prediction_model.eval()

        self.num_particles = particles.shape[0]

        # columns: vx, vy, vtheta, sin(theta), cos(theta), control v, steering, dt
        self.nn_inputs_np = np.zeros((self.particles.shape[0], 8))
        self.nn_inputs_cuda = FloatTensor(self.particles.shape[0], 8).cuda()

        self.last_poses = None

    def update(self, control):
        # compute delta pose
        last_poses = self.last_poses if self.last_poses is not None else np.zeros((self.num_particles, 3))
        current_poses = self.particles[:, :]
        thetas = current_poses[:, 2]

        # compute neural network inputs
        self.nn_inputs_np[:, 0:3] = current_poses - last_poses
        self.nn_inputs_np[:, 3] = np.sin(thetas)
        self.nn_inputs_np[:, 4] = np.cos(thetas)
        self.nn_inputs_np[:, 5] = control[0]
        self.nn_inputs_np[:, 6] = control[1]
        self.nn_inputs_np[:, 7] = control[2]

        # push to gpu, compute
        self.nn_inputs_cuda[:, :] = torch.from_numpy(self.nn_inputs_np)
        nn_result = self.prediction_model(Variable(self.nn_inputs_cuda))
        pose_delta = nn_result.data.cpu().numpy()

        # apply update to particles
        np.add(self.particles, pose_delta, out=self.particles)

        # store current pose for future
        self.last_poses = self.particles[:, :]


class InternalLearnedKinematicMotionModel:
    def __init__(self, particles, kinematic_motion_model, residual_model):
        self.particles = particles
        self.kinematic_motion_model = kinematic_motion_model
        self.residual_model = residual_model.eval()

        self.num_particles = particles.shape[0]

        # columns: vx, vy, vtheta, sin(theta), cos(theta), control v, steering, dt
        self.nn_inputs_np = np.zeros((self.particles.shape[0], 8))
        self.nn_inputs_cuda = FloatTensor(self.particles.shape[0], 8).cuda()

        self.last_poses = None

    def update(self, control):
        # compute delta pose and store current pose for future
        last_poses = self.last_poses if self.last_poses is not None else np.zeros((self.num_particles, 3))
        current_poses = self.particles[:, 0:3]
        thetas = current_poses[:, 2]
        self.last_poses = current_poses

        # compute neural network inputs
        self.nn_inputs_np[:, 0:3] = current_poses - last_poses
        self.nn_inputs_np[:, 3] = np.sin(thetas)
        self.nn_inputs_np[:, 4] = np.cos(thetas)
        self.nn_inputs_np[:, 5] = control[0]
        self.nn_inputs_np[:, 6] = control[1]
        self.nn_inputs_np[:, 7] = control[2]

        # push to gpu, compute
        self.nn_inputs_cuda[:, :] = torch.from_numpy(self.nn_inputs_np)
        nn_result = self.residual_model(Variable(self.nn_inputs_cuda))
        pose_delta_residuals = nn_result.data.cpu().numpy()

        # now apply kinematic model
        self.kinematic_motion_model.update(control)

        # and apply residuals on top of that
        np.subtract(self.particles, pose_delta_residuals, out=self.particles)

        # store current pose for future
        self.last_poses = self.particles[:, :]
