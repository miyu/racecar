import rospy
import rosbag
from std_msgs.msg import Int32, String
from SensorModel import SensorModel

bag = rosbag.Bag('../bags/laser_scans/laser_scan1.bag')

try:
    for topic, msg, t in bag.read_messages():
        message = msg
finally:
    bag.close()

#print(message.ranges)

rospy.wait_for_service('/static_map')
get_map = rospy.ServiceProxy('/static_map', GetMap)
map_msg = get_map().map

print(map_msg)

sensorModel = SensorModel()
