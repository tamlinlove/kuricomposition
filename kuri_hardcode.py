#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import time

def move(x, y, z, ax, ay, az):
    # Starts a new node
    rospy.init_node('turtlebot_node', anonymous=True)
    velocity_publisher = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
    velocity_msg = Twist()

    print("Moving turtlebot")
    # velocity_msg.linear.x = 0.3
    # velocity_msg.linear.x = x


    while not rospy.is_shutdown():
        # time_end = time.time() + 1
        # while(time.time() < time_end):
        velocity_msg.linear.x = x;
        velocity_msg.linear.y = y;
        velocity_msg.linear.z = z;
        velocity_msg.angular.x = ax;
        velocity_msg.angular.y = ay;
        velocity_msg.angular.z = az;
        velocity_publisher.publish(velocity_msg)
        # exiting the loop
        break

if __name__ == '__main__':
    move(1,0,0,0,0,0)
    move(0,0,0,0,0,-1.6) # Turn right
    move(1,0,0,0,0,0)
    move(0,0,0,0,0,1.6) # Turn left
    move(1,0,0,0,0,0)
