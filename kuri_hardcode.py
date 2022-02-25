#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import time

def move(publisher, x, y, z, ax, ay, az):
    print("Executing a move")
# Starts a new node
    #rospy.init_node('robot_cleaner', anonymous=True)
    #velocity_publisher = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=100)

    velocity_msg = Twist()

     #Need to compensate for subscriber
    # Need this for the robot to do anything
    rate = rospy.Rate(5) # 10hz
    rate.sleep()

    velocity_msg.linear.x = x;
    velocity_msg.linear.y = y;
    velocity_msg.linear.z = z;
    velocity_msg.angular.x = ax;
    velocity_msg.angular.y = ay;
    velocity_msg.angular.z = az;

    publisher.publish(velocity_msg)
    time.sleep(1)

if __name__ == '__main__':
    rospy.init_node('robot_cleaner',anonymous=True)
    vpub = rospy.Publisher('/mobile_base/commands/velocity',Twist,queue_size=100)
    time.sleep(1)
    move(vpub,1,0,0,0,0,0)
    move(vpub,0,0,0,0,0,-1.6) # Turn right
    move(vpub,1,0,0,0,0,0)
    move(vpub,0,0,0,0,0,1.6) # Turn left
    move(vpub,1,0,0,0,0,0)
    move(vpub,1,0,0,0,0,0)
    move(vpub,1,0,0,0,0,0)
    move(vpub,0,0,0,0,0,-1.6) # Turn right
    move(vpub,1,0,0,0,0,0)
    move(vpub,1,0,0,0,0,0)
    move(vpub,1,0,0,0,0,0)
    time.sleep(10)
