# Note: Do not copy to remote's catkin_ws; that's already
# used by the racecar stuff provided to us.
echo "ssh nvidia@10.42.0.1 password is nvidia10"
#rsync -rav -e ssh --include='*/' --include='*.launch' --exclude='*' . nvidia@10.42.0.1:~/our_catkin_ws
#rsync -rav -e ssh --include='*/' --include='*.pickle' --exclude='*' . nvidia@10.42.0.1:~/our_catkin_ws
rsync -rav -e ssh --include='*/' --include='*.py' --exclude='*' . nvidia@10.42.0.1:~/our_catkin_ws
#rsync -rav -e ssh --include='*/' --include='CMakeLists.txt' --exclude='*' . nvidia@10.42.0.1:~/our_catkin_ws
# if you sync sh files, need to rm -R build and devel directories in remote
# then rebuild with catkin_make
# rsync -rav -e ssh --include='*/' --include='*.sh' --exclude='*' . nvidia@10.42.0.1:~/our_catkin_ws
