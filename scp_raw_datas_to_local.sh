# Note: Do not copy to remote's catkin_ws; that's already
# used by the racecar stuff provided to us.
echo "ssh nvidia@10.42.0.1 password is nvidia10"

scp nvidia@10.42.0.1:~/our_catkin_ws/src/lab3/src/raw_datas.npy ~/catkin_ws/src/lab3/src/raw_datas.npy
