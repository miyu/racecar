#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import pylab as p
import math
import cv2
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from InternalMotionModel import InternalOdometryMotionModel, InternalKinematicMotionModel
import time

FILTER_SIZE = (480,640)

IS_ON_ROBOT = True

def plot_init(n, title):
    plt.figure(n)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)


def plot_draw_particles(particles, directional):
    #xs, ys, thetas = particles[:,0], particles[:,1], particles[:,2]
    xs = particles[:,0]
    ys = particles[:,1]
    zs = particles[:,2]
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
    nparticles = 1000
    camera_fov_horizontal = 68 * (math.pi / 180.0)
    camera_fov_vertical = 41 * (math.pi / 180.0)
    camera_downward_tilt_angle = 35.0 * (math.pi / 180.0)

    tan_fh2 = math.tan(camera_fov_horizontal / 2)
    tan_fv2 = math.tan(camera_fov_vertical / 2)
    # aggregate_particle = np.zeros((0, 3), dtype=float)
    # aggregate_particle2 = np.zeros((0, 4), dtype=float)
    templates = np.zeros((len(np.arange(-0.28, 0.30, 0.02)),FILTER_SIZE[0],FILTER_SIZE[1]), dtype='uint8')
    template_number = 0
    for steering in np.arange(-0.28, 0.30, 0.02): # arange is exclusive
        aggregate_particle_states = np.zeros((0, 3), dtype=float)
        particle = list()
        # simulate particles 1 meter forward
        particles = np.zeros((nparticles, 3), dtype=float)
        mm = InternalKinematicMotionModel(particles)
        speed, dt = 1.0, 0.005
        t = 0.0
        t1 = 1.0
        t += t1
        for i in range(int(t1 / (speed * dt))):
            mm.update([speed, steering, dt])
            # particle.x = np.average(particles)
            aggregate_particle_states = np.concatenate((aggregate_particle_states, particles), axis=0)

        t2 = 1.0 - t1
        t += t2
        for i in range(int(t2 / (speed * dt))):
            mm.update([speed, -steering, dt])
            # particle.x = np.average(particles)
            aggregate_particle_states = np.concatenate((aggregate_particle_states, particles), axis=0)

        # # plot particles in 2D robot-space
        # plot_init(0, "mm steering " + str(steering))
        # plot_draw_particles(aggregate_particle_states, False)
        # plot_show([0,1,-1, 1])
        aggregate_particle = np.concatenate((aggregate_particle, aggregate_particle_states), axis=0)
        r_y = np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]])
        r_z = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]])
        r_x = np.array([
            [1, 0, 0],
            [0, np.cos(-camera_downward_tilt_angle), np.sin(-camera_downward_tilt_angle)],
            [0, -np.sin(-camera_downward_tilt_angle), np.cos(-camera_downward_tilt_angle)]])
        R = r_x.dot(r_z.dot(r_y))
        T = np.array([[0.0], [0.2], [0.2]])
        tbf = np.concatenate((R, T), axis=1)
        camera_transform = np.concatenate((tbf, np.array([[0.0, 0.0, 0.0, 1.0]])), axis = 0)
        # print(camera_transform)
        # camera_particles = np.zeros((aggregate_particle_states.shape[0], 4))
        # camera_particles[:, 0] = aggregate_particle_states[:, 0]
        # camera_particles[:, 1] = aggregate_particle_states[:, 1]
        # camera_particles[:, 2] = aggregate_particle_states[:, 2]
        # camera_particles[:, 3] = 1.0
        # camera_particles = camera_transform.dot(camera_particles.T).T
        # aggregate_particle2 = np.concatenate((aggregate_particle2, camera_particles), axis=0)
        #print(camera_particles.shape)


        #transform 2D (in meters) to camera-space
        naggregates = aggregate_particle_states.shape[0]
        world_particles = np.zeros((naggregates, 4))
        world_particles[:, 0] = -aggregate_particle_states[:, 1] # left / right; recall positive y is left..
        world_particles[:, 1] = np.zeros(aggregate_particle_states.shape[0]) # vertical
        world_particles[:, 2] = aggregate_particle_states[:, 0] # near/far; recall moving forward increased x
        world_particles[:, 3] = 1

        ctilt = math.cos(-camera_downward_tilt_angle)
        stilt = math.sin(-camera_downward_tilt_angle)
        camera_transform = np.array([ # negatives because translating to camera space
            [1, 0, 0, -0.04],
            [0, ctilt, -stilt, -0.2],
            [0, stilt, ctilt, -0.08], # 0.50 pushes origin to center of screen, probably should be -0.08
            [0, 0, 0, 1]]) # 8 14 4

        # camera_rotate = np.array([
        #     [ctilt, -stilt, 0, 0],
        #     [stilt, ctilt, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]])
        camera_rotate = np.array([
            [1, 0, 0, 0],
            [0, ctilt, -stilt, 0],
            [0, stilt, ctilt, 0],
            [0, 0, 0, 1]])
        camera_particles = np.matmul(camera_transform, world_particles.T).T
        # plot_draw_particles(camera_particles, False)
        # plot_show([0,1,-1, 1])
        aggregate_particle2 = np.concatenate((aggregate_particle2, camera_particles), axis=0)

        camera_particles[:,0] = (FILTER_SIZE[1]/2) * camera_particles[:,0]
        camera_particles[:,2] = FILTER_SIZE[0] * camera_particles[:,2]
        new_image = np.zeros(FILTER_SIZE, dtype='uint8')
        #plot_draw_particles(camera_particles, False)
        for i in range(len(camera_particles)):
            x = int(camera_particles[i][0])
            z = int(camera_particles[i][2])
            if -FILTER_SIZE[1]/2 <= x < FILTER_SIZE[1]/2 and 0 < z <= FILTER_SIZE[0]:
                new_image[int(FILTER_SIZE[0]-z)][int(x+FILTER_SIZE[1]/2)] = 255
            else:
                #print('Else:')
                continue


        if not IS_ON_ROBOT:
            print("showing image")
            cv2.imshow('image', new_image)
            k = cv2.waitKey()
            if k == 27:
                continue

        # frustum_bound_xs = tan_fh2 * camera_particles[:, 2]
        # frustum_bound_ys = tan_fv2 * camera_particles[:, 2]
        # clip_space_xs = camera_particles[:, 0] / frustum_bound_xs
        # clip_space_ys = camera_particles[:, 1] / frustum_bound_ys
        # clip_space_particles = np.zeros((aggregate_particle_states.shape[0], 3), dtype=float)
        # clip_space_particles[:, 0] = clip_space_xs
        # clip_space_particles[:, 1] = clip_space_ys
        # print(clip_space_particles.shape)
        # plot_init(0, "3D mm steering " + str(steering))
        templates[template_number] = new_image
        template_number += 1

    # 8 14 4
    # plot_init(0, "mm steering")
    # plot_draw_particles(aggregate_particle2, False)
    # plot_show([-1, 1, -1, 1])
    return templates

class TemplateMatcher:
    def __init__(self, sub_topic, pub_topic):
        print("Subscribing to", sub_topic)
        print("Publishing to", pub_topic)

        self.templates = compute_templates()
        assert self.templates is not None

        self.pub = rospy.Publisher(pub_topic, Image, queue_size=10)
        self.sub = rospy.Subscriber(sub_topic, Image, self.apply_filter_cb)
        self.bridge = CvBridge()
        self.teleop_pub = rospy.Publisher('/vesc/high_level/ackermann_cmd_mux/input/nav_0', AckermannDriveStamped, queue_size=10)

        print('Done!')

    def show(self, img):
        cv2.imshow('stuff', img)
        k = cv2.waitKey()
        if k == 27:
            return

    def pick_template(self, img):
        start_time = time.time()
        best_overlap = 0
        best_template = None
        overlayed_image = None
        index = -1
        alpha = 0.5

        # img should be grayscale, single plane
        assert len(img.shape) == 2

        gray_plane = img
        for i in range(len(self.templates)):
            template = self.templates[i]
            #self.pub.publish(self.bridge.cv2_to_imgmsg(template))
            # overlayed_image = cv2.addWeighted(template, alpha, img[:,:,0], 1.0 - alpha, 0.0)
            #assert overlayed_image is not None
            #self.show(overlayed_image)
            overlap = np.sum(gray_plane * template)
            #overlap = np.sum(convolve(img[:,:,0], template[np.newaxis,:,:]))
            #overlap = np.sum(signal.convolve(img[:,:,0],np.flipud(np.fliplr(template)), mode='valid'))
            #print('Overlap', overlap)
            if overlap > best_overlap:
                best_template = template
                best_overlap = overlap
                index = i

        template_picked_time = time.time()

        print(best_overlap)
        if best_template is not None and index > -1:
            best_overlay = cv2.addWeighted(best_template, alpha, gray_plane, 1.0 - alpha, 0.0)
            self.pub.publish(self.bridge.cv2_to_imgmsg(best_overlay))
            self.drive(forward_velocity=0.3, steering_angle=float(-0.28+index*0.02), dt=0.005)
        end_time = time.time()

        print("pick_template", template_picked_time - start_time, "end", end_time - start_time)
        #self.show(self.templates[best_template])

    """
    A callback to remove all except some range of colors.

    Assuming that msg is sensor_msgs/Image
    """
    def apply_filter_cb(self, msg):
        start_time = time.time()
        # cv_image = cv2.imread('111.jpg', 1)
        # cv_image = cv2.resize(cv_image, (640, 480))
        # self.show(cv_image)
        cv_image = None
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        except CvBridgeError as e:
            print(e)

        image_decoded_time = time.time()

        #cv_image = cv2.GaussianBlur(cv_image, (3,3), 0)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
        hsv_time = time.time()

        # To see the red tape
        #lower_l = np.array([0, 100, 100])
        #lower_h = np.array([20, 255, 255])
        # higher_l = np.array([120, 100, 100]) # THESE WORKED WITH IMAGES AT HOME
        # higher_r = np.array([180, 255, 255])

        # #######################
        # # Known good red
        # higher_l = np.array([150, 100, 100], np.uint8) #THESE WORKED WITH CAMERA
        # higher_r = np.array([179, 255, 255], np.uint8)
        # mask_r = cv2.inRange(hsv, higher_l, higher_r)
        # mask = mask_r

        #######################
        # Known good blue is hue (0-360): 166-187, but cv divides range by 2
        lower = np.array([(166 - 20) / 2, 100, 100], np.uint8)
        upper = np.array([(187 + 20) / 2, 255, 255], np.uint8)
        mask_b = cv2.inRange(hsv, lower, upper)
        mask = mask_b

        #mask_rl = cv2.inRange(hsv, lower_l, lower_h)
        #mask_r = cv2.addWeighted(mask_rl, 1.0, mask_rh, 1.0, 0.0)
        filtered_time = time.time()

        res = cv2.bitwise_and(cv_image, cv_image, mask = mask)
        and_time = time.time()
        #print("shape is ", res.shape)
        #print("Publishing:")

        # self.drawROI(res)
        # try:
        #     res[:,:,0] = 0
        #     res[:,:,1] = 0
        #     self.drawROI(res)
        #     #self.pub.publish(self.bridge.cv2_to_imgmsg(res, 'rgb8'))
        # except CvBridgeError as e:
        #     print(e)
        self.pick_template(mask)
        roi_drawn_time = time.time()


        print("DONE! ", "IDT", image_decoded_time - start_time, "HSV", hsv_time - start_time, "Filtered", filtered_time - start_time, "And", and_time - start_time, "ROI", roi_drawn_time - start_time)

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
        bottomLineHeight = int(height - 10)
        topLineHeight = int(height - height/6)

        # Mask img for ROI
        mask = np.zeros((height, width), dtype = 'uint8')
        mask[(topLineHeight):(bottomLineHeight) ,:] = 1
        ROIimg = cv2.bitwise_and(img ,img , mask = mask)

        #self.show(img)
        #self.pub.publish(self.bridge.cv2_to_imgmsg(img, 'rgb8')) #UNCOMMENT THIS TO PUBLISH TO RVIZ

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

        cv2.line(img, (0,topLineHeight), (width,topLineHeight), color = (0, 255, 0))
        cv2.line(img, (0,bottomLineHeight), (width, bottomLineHeight), color = (0, 255, 0))

        #self.show(img)

    def drive(self, forward_velocity, steering_angle, dt):
        msg = AckermannDriveStamped()
        msg.drive.speed = forward_velocity
        msg.drive.steering_angle = steering_angle
        self.teleop_pub.publish(msg)
        #rospy.sleep(dt) # dt must be in seconds. Consider using r = rospy.rate(RATE in Hz), r.sleep() if this doesn't work as intended

if __name__ == '__main__':
    print("Main")
    sub_topic = None # The image topic to be subscribed to
    pub_topic = None # The topic to publish filtered images to

    if IS_ON_ROBOT:
        rospy.init_node('gen_rollouts', anonymous=True)

        #Populate params with values passed by launch file
        sub_topic = rospy.get_param('~sub_topic')
        pub_topic = rospy.get_param('~pub_topic')

    t = TemplateMatcher(sub_topic, pub_topic)

    if IS_ON_ROBOT:
        rospy.spin()
    else:
        compute_templates()
        t.apply_filter_cb(None)
