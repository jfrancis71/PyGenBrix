# Must be run in conda RoboStack environment.
ros2 run image_transport republish compressed raw --ros-args --remap in/compressed:=/thomas/compressed --remap out:=server/raw

