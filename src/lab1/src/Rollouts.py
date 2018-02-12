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
IS_GENERATE_RUN = False
camera_fov_horizontal = 68 * (math.pi / 180.0)
camera_fov_vertical = 41 * (math.pi / 180.0)
camera_downward_tilt_angle = 0.0 * (math.pi / 180.0) # should be negative if looking down

def plot_init(n, title):
    plt.figure(n)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)


def plot_draw_particles(particles, directional):
    if not directional:
        xs = particles[:,0]
        ys = particles[:,1]
        zs = particles[:,2]

        valid_xs = np.extract(zs > 0, xs)
        valid_ys = np.extract(zs > 0, ys)

        plt.scatter(valid_xs, valid_ys, marker=MarkerStyle('o', 'full'), facecolor='0', alpha=0.01, lw=0)
    else:
        xs, ys, thetas = particles[:,0], particles[:,1], particles[:,2]

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
    print("Computing templates")

    # particle rollouts
    def generate_rollout(steering):
        # simulate particles 1 meter forward
        nparticles = 1000
        particles = np.zeros((nparticles, 3), dtype=float)
        mm = InternalKinematicMotionModel(particles) #, np.array([[0.0, 0.001], [0.0, 0.001]])
        speed, dt = 1.0, 0.005
        rollout_meters = 2
        rollout_particles = np.zeros((0, 3), dtype=float)
        for i in range(int(rollout_meters / (speed * dt))):
            mm.update([speed, steering, dt])
            rollout_particles = np.concatenate((rollout_particles, particles), axis=0)

        return rollout_particles

    # # plot particles in 2D robot-space
    # for steering, rollout_particles in rollout_particles_by_steering.iteritems():
    #     plot_init(0, "mm steering " + str(steering))
    #     plot_draw_particles(rollout_particles, False)
    #     plot_show([-1, 1,-1, 1])

    # take particles to world space
    def rollout_particles_to_world(rollout_particles):
        num_rollout_particles = rollout_particles.shape[0]

        world_particles = np.zeros((num_rollout_particles, 4))
        world_particles[:, 0] = -rollout_particles[:, 1] # left / right; recall positive y is left..
        world_particles[:, 1] = 0 # vertical
        world_particles[:, 2] = rollout_particles[:, 0] # near/far; recall moving forward increased x
        world_particles[:, 3] = 1

        return world_particles

    # take particles to camera space
    def world_particles_to_camera(world_particles):
        camera_position = [0.04, 0.2, -0.08] # with left hand, X right, Y up, Z outward. Not same as robot coords

        ctilt = math.cos(camera_downward_tilt_angle)
        stilt = math.sin(camera_downward_tilt_angle)

        camera_transform = np.array([ # negatives because translating to camera space
            [1, 0, 0, -camera_position[0]],
            [0, ctilt, -stilt, -camera_position[1]],
            [0, stilt, ctilt, -camera_position[2]], # 0.50 pushes origin to center of screen, probably should be -0.08
            [0, 0, 0, 1]])

        return np.matmul(camera_transform, world_particles.T).T

    # draw camera space to screen / image
    def camera_particles_to_clip(camera_particles):
        tan_fh2 = math.tan(camera_fov_horizontal / 2)
        tan_fv2 = math.tan(camera_fov_vertical / 2)
        frustum_bound_xs = tan_fh2 * camera_particles[:, 2]
        frustum_bound_ys = tan_fv2 * camera_particles[:, 2]
        clip_space_xs = camera_particles[:, 0] / frustum_bound_xs
        clip_space_ys = camera_particles[:, 1] / frustum_bound_ys
        clip_space_particles = np.zeros((camera_particles.shape[0], 3), dtype=float)
        clip_space_particles[:, 0] = clip_space_xs
        clip_space_particles[:, 1] = clip_space_ys
        clip_space_particles[:, 2] = camera_particles[:, 2]
        return clip_space_particles

    def offset_z(camera_particles, offset):
        copy = np.copy(camera_particles)
        copy[:, 2] += offset
        return copy

    def render_camera_particles_plot(camera_particles, steering, offset):
        clip_particles = camera_particles_to_clip(camera_particles)
        plot_init(0, "3D camera - steering " + str(steering) + " offset " + str(offset))
        plot_draw_particles(clip_particles, False)
        plot_show([-1, 1, -1, 1])

    def render_camera_particles_bitmap(camera_particles, steering, offset):
        clip_particles = camera_particles_to_clip(camera_particles)

        new_image = np.zeros(FILTER_SIZE, dtype=float)
        base_weight = 0
        for i in range(len(clip_particles)):
            x = int((clip_particles[i][0] * 0.5 + 0.5) * FILTER_SIZE[1])
            y = int((clip_particles[i][1] * -0.5 + 0.5) * FILTER_SIZE[0])
            if 0 <= x < FILTER_SIZE[1] and 0 <= y < FILTER_SIZE[0] and clip_particles[i][2] > 0:
                p = float(i) / len(clip_particles)
                new_image[y][x] = min(1.0, max(0.0, new_image[y][x] + p * p * p))
            else:
                continue
        new_image *= 1.0 / max(0.0001, np.amax(new_image))

        displayed_image = np.zeros(FILTER_SIZE, dtype='uint8')
        for y in range(FILTER_SIZE[0]):
            for x in range(FILTER_SIZE[1]):
                displayed_image[y][x] = int(new_image[y][x] * 255)

        if not IS_ON_ROBOT and not IS_GENERATE_RUN:
            print("showing image", "steering", steering, "offset", offset)
            cv2.imshow('image', displayed_image)
            while True:
                k = cv2.waitKey()
                if k == 27:
                    break

        return displayed_image

    if not IS_ON_ROBOT and not IS_GENERATE_RUN:
        steering = 0.27
        offset = 0.0 # offset negative means "if we went backward then we could do this steering"
        rollout = generate_rollout(steering)
        world_particles = rollout_particles_to_world(rollout)
        camera_particles = world_particles_to_camera(world_particles)

        while True:
            render_camera_particles_bitmap(offset_z(camera_particles, offset), steering, offset)
            render_camera_particles_plot(offset_z(camera_particles, offset), steering, offset)
            render_camera_particles_bitmap(offset_z(camera_particles, 0), steering, 0)
            render_camera_particles_plot(offset_z(camera_particles, 0), steering, 0)
    else:
        path = "/home/nvidia/our_catkin_ws/src/lab1/src/template_and_controls.pickle"
        if IS_GENERATE_RUN:
            path = "template_and_controls.pickle"

        import pickle
        try:
            with open(path, "rb") as fd:
                template_and_controls = pickle.load(fd)
                print ("Happily deserialized ", path, len(template_and_controls), template_and_controls[0][0].shape)
                return template_and_controls
        except (OSError, IOError) as e:
            if not IS_GENERATE_RUN:
                raise "couldn't find template and controls file"

            template_and_controls = [] # array of (image, steering | None)
            for steering in np.arange(-0.28, 0.281, 0.02): # arange is exclusive
                rollout = generate_rollout(steering)
                world_particles = rollout_particles_to_world(rollout)
                camera_particles = world_particles_to_camera(world_particles)

                # template for if we followed this steering
                template = render_camera_particles_bitmap(camera_particles, steering, 0.0)
                template_and_controls.append((template, steering))

                # templates for if we moved back, then followed steering
                for meters_backwards in np.arange(0.10, 0.201, 0.1):
                    print("Meters backwards", meters_backwards)
                    offset = -meters_backwards
                    template = render_camera_particles_bitmap(offset_z(camera_particles, offset), steering, offset)
                    template_and_controls.append((template, None))

            with open(path, "wb") as fd:
                pickle.dump(template_and_controls, fd)

            return template_and_controls

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
        best_score = 0
        best_template = None
        overlayed_image = None
        index = -1
        alpha = 0.5

        # img should be grayscale, single plane
        assert len(img.shape) == 2

        gray_plane = img
        for i in range(len(self.templates)):
            template, steering = self.templates[i]
            #self.pub.publish(self.bridge.cv2_to_imgmsg(template))
            # overlayed_image = cv2.addWeighted(template, alpha, img[:,:,0], 1.0 - alpha, 0.0)
            #assert overlayed_image is not None
            #self.show(overlayed_image)
            overlap = np.sum(gray_plane * template)
            #overlap = np.sum(convolve(img[:,:,0], template[np.newaxis,:,:]))
            #overlap = np.sum(signal.convolve(img[:,:,0],np.flipud(np.fliplr(template)), mode='valid'))
            #print('Overlap', overlap)

            score = overlap
            if steering is None:
                score /= 100.0
            else:
                print("steering ", steering, "score", score)
                if -0.245 <= steering <= -0.235:
                    score = float('inf')

            if score > best_score:
                best_template = template
                best_score = score
                index = i

        template_picked_time = time.time()

        if best_template is not None and index > -1:
            best_overlay = cv2.addWeighted(best_template, alpha, gray_plane, 1.0 - alpha, 0.0)
            self.pub.publish(self.bridge.cv2_to_imgmsg(best_overlay))

            template, steering = self.templates[index]
            if steering is None:
                self.drive(forward_velocity=-0.3, steering_angle=0.0, dt=0.005)
            else:
                self.drive(forward_velocity=0.3, steering_angle=steering, dt=0.005)

        end_time = time.time()

        print("pick_template", template_picked_time - start_time, "end", end_time - start_time, "best steering is", self.templates[index][1], "score", best_score)
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
