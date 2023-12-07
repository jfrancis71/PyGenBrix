# Must be run in conda RobotStack environment.
ros2 run image_tools cam2image &
ros2 run image_transport republish raw compressed --ros-args --remap in:=/image --remap out/compressed:=thomas/compressed &
