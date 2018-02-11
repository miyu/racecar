import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import pylab as p
import math
import cv2
import itertools
from InternalMotionModel import InternalOdometryMotionModel, InternalKinematicMotionModel

def plot_init(n, title):
    plt.figure(n)
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.title(title)


def plot_draw_particles(particles, directional):
    #xs, ys, thetas = particles[:,0], particles[:,1], particles[:,2]
    xs = 320 * particles[:,0]
    ys = particles[:,1]
    zs = 640 * particles[:,2]
    if not directional:
        plt.scatter(xs, zs, marker=MarkerStyle('o', 'full'), facecolor='0', alpha=1, lw=0)
    else:
        dxs = np.cos(thetas) * 0.01
        dys = np.sin(thetas) * 0.01

        ax = plt.gca()
        ax.quiver(xs, ys, dxs, dys, angles='xy', scale_units='xy', scale=1)
        plt.draw()

def plot_show(axis=None):
    if axis:
        plt.axis(axis)
    plt.show()

def compute_templates():
    nparticles = 1
    camera_fov_horizontal = 68 * (math.pi / 180.0)
    camera_fov_vertical = 41 * (math.pi / 180.0)
    camera_downward_tilt_angle = 20 * (math.pi / 180.0)
    camera_translate_left_right, camera_translate_down_up, camera_translate_near_far = 0, 0.2, 0

    tan_fh2 = math.tan(camera_fov_horizontal / 2)
    tan_fv2 = math.tan(camera_fov_vertical / 2)

    for steering in np.arange(-0.25, 0.25, 0.01):
        aggregate_particle_states = np.zeros((0, 3), dtype=float)

        # simulate particles 1 meter forward
        particles = np.zeros((nparticles, 3), dtype=float)
        mm = InternalKinematicMotionModel(particles)
        speed, dt = 1.0, 0.005
        for i in range(int(5.0 / (speed * dt))):
            mm.update([speed, steering, dt])
            # particle.x = np.average(particles)
            aggregate_particle_states = np.concatenate((aggregate_particle_states, particles), axis=0)

        # # plot particles in 2D robot-space
        # plot_init(0, "mm steering " + str(steering))
        # plot_draw_particles(aggregate_particle_states, False)
        # plot_show()

        # transform 2D (in meters) to camera-space
        naggregates = aggregate_particle_states.shape[0]
        world_particles = np.zeros((naggregates, 4))
        world_particles[:, 0] = -aggregate_particle_states[:, 1] # left / right; recall positive y is left..
        world_particles[:, 1] = np.zeros(aggregate_particle_states.shape[0]) # vertical
        world_particles[:, 2] = aggregate_particle_states[:, 0] # near/far; recall moving forward increased x
        world_particles[:, 3] = 1

        ctilt = math.cos(-camera_downward_tilt_angle)
        stilt = math.sin(-camera_downward_tilt_angle)
        camera_transform = np.array([
            [1, 0, 0, -0],
            [0, ctilt, -stilt, -0],
            [0, stilt, ctilt, -0],
            [0, 0, 0, 1]])

        # camera_rotate = np.array([
        #     [ctilt, -stilt, 0, 0],
        #     [stilt, ctilt, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]])
        # camera_rotate = np.array([
        #     [1, 0, 0, 0],
        #     [0, ctilt, -stilt, 0],
        #     [0, stilt, ctilt, 0],
        #     [0, 0, 0, 1]])
        camera_particles = np.matmul(camera_transform, world_particles.T).T

        # frustum_bound_xs = tan_fh2 * camera_particles[:, 2]
        # frustum_bound_ys = tan_fv2 * camera_particles[:, 2]
        # clip_space_xs = camera_particles[:, 0] / frustum_bound_xs
        # clip_space_ys = camera_particles[:, 1] / frustum_bound_ys
        # clip_space_particles = np.zeros((aggregate_particle_states.shape[0], 3), dtype=float)
        # clip_space_particles[:, 0] = clip_space_xs
        # clip_space_particles[:, 1] = clip_space_ys
        # print(clip_space_particles.shape)
        # plot_init(0, "3D mm steering " + str(steering))
        particlesx = 320 * camera_particles[0]
        particlesz = 480 * camera_particles[2]
        new_image = np.zeros((480,640), dtype='uint8')
        # plot_draw_particles(camera_particles, False)
        # print(len(camera_particles))
        for i in range(1000):
            # particle = camera_particles[i]
            x = int(particlesx[i])
            z = int(particlesz[i])
            print(x, z)
            if -320 <= x <= 320 and 0 <= z <= 479:
                new_image[z][x] = 1
            else:
                print(i)
                break

        # cv2.imshow('image', new_image)
        # k = cv2.waitKey()
        # if k == 27:
        #     return
        #plot_show()
        #480 by 640, rgb
        plot_show([-320, 320, 0 , 480])

class TemplateMatcher:
    def __init__(self, sub_topic, pub_topic):
        print("Subscribing to", sub_topic)
        print("Publishing to", pub_topic)

        self.pub = rospy.Publisher(pub_topic, Image, queue_size=10)
        self.sub = rospy.Subscriber(sub_topic, Image, self.apply_filter_cb)
        self.bridge = CvBridge()

    """
    A callback to remove all except some range of colors.

    Assuming that msg is sensor_msgs/Image
    """
    def apply_filter_cb(self, msg):
        cv_image = None
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        except CvBridgeError as e:
            print(e)

        cv_image = cv2.GaussianBlur(cv_image, (3,3), 0)

        hsv  = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)

        # To see the red tape
        # lower_l = np.array([0, 100, 100])
        # lower_h = np.array([20, 255, 255])
        higher_l = np.array([170, 100, 100])
        higher_r = np.array([179, 255, 255])
        # mask_rl = cv2.inRange(hsv, lower_l, lower_h)
        mask_r = cv2.inRange(hsv, higher_l, higher_r)
        # mask_r = cv2.addWeighted(mask_rl, 1.0, mask_rh, 1.0)

        mask = mask_r

        res = cv2.bitwise_and(cv_image, cv_image, mask = mask)
        #print("shape is ", res.shape)
        #print("Publishing:")
        try:
            # res[:,:,0] = 0
            # res[:,:,1] = 0
            self.drawROI(res)

            # self.pub.publish(self.bridge.cv2_to_imgmsg(res, 'rgb8'))
        except CvBridgeError as e:
            print(e)
        print("DONE!")

    def getError(self):
        return self.error


    """
        Draw out thecv2.inRange(hsv, lower, upper) ROI, and get the center
    """
    def drawROI(self, img):
        height = None
        width = None
        channels = 1

        edges = None


        if len(img.shape) == 2:
            height, width = img.shape
            #edges = img.copy()
        else:
            height, width, channels  = img.shape

            #edges = self.getEdges(img)

        # Used to set the lines of the ROI
        bottomLineHeight = height - 10
        topLineHeight = height - height/6

        # Mask img for ROI
        mask = np.zeros((height, width), dtype = 'uint8')
        mask[(topLineHeight):(bottomLineHeight) ,:] = 1
        ROIimg = cv2.bitwise_and(img ,img , mask = mask)

        gray = cv2.cvtColor(ROIimg, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    	M = cv2.moments(cnts)
        #print("M is ", M)
        # If the tape is not within the ROI, then don't calculate moment
        if M["m00"] == 0:
            print("Not on top of tape")
            self.error = 0.0
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
            self.error = float(width/2.0 - float(cX))
            print('Cx: ', float(cX))
            print('Center:', width/2.0)

        # negative error, turn right... hopefully
        control = self.PID.calc_control(self.error)
        self.PID.drive(control)

        cv2.line(img, (0,topLineHeight), (width,topLineHeight), color = (0, 255, 0))
        cv2.line(img, (0,bottomLineHeight), (width, bottomLineHeight), color = (0, 255, 0))

if __name__ == '__main__':
    compute_templates()
