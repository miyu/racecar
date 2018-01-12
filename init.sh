echo_error() {
	echo -e "\033[31m$@\e[0m"
}
if [ $SHLVL -ne 1 ]; then
	echo_error "Usage: . init.sh"
	echo_error "So don't do ./init.sh"
	exit 1
fi

# build locally, load ros dev script
. /opt/ros/kinetic/setup.bash
catkin_make
. devel/setup.bash

# setup wireless network
echo "Wireless: Make sure to connect to Jetson10 password nv1d1410"

# configure ROS_IP and ROS_MASTER_URI
export ROS_IP=$(ifconfig | grep wlx -a1 | grep addr | egrep -o "[0-9]+(\.[0-9]+)+" | head -n 1)
export ROS_MASTER_URI="http://10.42.0.1:11311"
echo "Set ROS_IP=$ROS_IP and ROS_MASTER_URI=$ROS_MASTER_URI"

