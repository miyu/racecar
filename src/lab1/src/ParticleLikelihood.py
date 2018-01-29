import rospy
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from threading import Lock

from std_msgs.msg import Int32, String
from SensorModel import SensorModel
from nav_msgs.srv import GetMap


MAP_FILENAME = '../maps/sieg_floor3.pgm'

bag = rosbag.Bag('../bags/laser_scans/laser_scan1.bag')
message = None
try:
    for topic, msg, t in bag.read_messages():
        message = msg
finally:
    bag.close()

#print(message.ranges)

rospy.wait_for_service('/static_map')
get_map = rospy.ServiceProxy('/static_map', GetMap)
map_msg = get_map().map

print(map_msg.header)
print(len(map_msg.data))

print(map_msg.info)

width = map_msg.info.width
print("Width: ", width)

height = map_msg.info.height
print("Height: ", height)

resolution = map_msg.info.resolution
print("Resolution: ", resolution )

origin = map_msg.info.origin.position
print("Origin: ", origin )


particles = []
max_particles = 0

for i in range(height):
    for j in range(width):
        if(map_msg.data[i*height+j] != -1):
            #print("Known space: (", i, ", " ,j , ") with value ",
            #      map_msg.data[i*height+j])
            x = j *resolution + origin.x 
            y = i * resolution + origin.y
            for theta in range(1,10):
                particles.append([x, y, 6.28/theta])
                max_particles += 1


print("Done getting particles")
weights = np.ones(max_particles, dtype=np.float64) / max_particles
np_particles = np.array(particles, dtype=np.float64)

state_lock = Lock()

sensorModel = SensorModel(map_msg, np_particles, weights )

sensorModel.lidar_cb(message)

new_particles = sensorModel.particles

print("Got to update sensor model")

#with open(MAP_FILE_NAME, 'rb') as f:
#    buffer = f.read()

#image = plt.imread(MAP_FILENAME)

#l = []
#for i in range(len(image)):
#    for j in range(len(image[0])):
#        if image[i][j] not in l:
#            l.append(image[i][j])
            
#plt.imshow(image, plt.cm.gray)
#plt.show()

