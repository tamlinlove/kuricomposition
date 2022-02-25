#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import time

def move(x, y, z, ax, ay, az):
    # Starts a new node
    rospy.init_node('robot_cleaner', anonymous=True)
    velocity_publisher = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)

    velocity_msg = Twist()

     #Need to compensate for subscriber
    rate = rospy.Rate(10) # 10hz
    rate.sleep()
    time = time_in
    #print("Moving Bot for {0}s".format(time))

    print("Moving turtlebot")
    # velocity_msg.linear.x = 0.3
    # velocity_msg.linear.x = x

    #run for specific time
    t0 = rospy.Time.now().to_sec()
    t1 = t0

    velocity_msg.linear.x = x;
    velocity_msg.linear.y = y;
    velocity_msg.linear.z = z;
    velocity_msg.angular.x = ax;
    velocity_msg.angular.y = ay;
    velocity_msg.angular.z = az;


    while(t1-t0 < time):
        #Publish the velocity
        velocity_publisher.publish(vel_msg)
        t1=rospy.Time.now().to_sec()
    #After the loop, stops the robot
    vel_msg.linear.x = 0
    #Force the robot to stop
    velocity_publisher.publish(vel_msg)

if __name__ == '__main__':
    move(1,0,0,0,0,0)
    move(0,0,0,0,0,-1.6) # Turn right
    move(1,0,0,0,0,0)
    move(0,0,0,0,0,1.6) # Turn left
    move(1,0,0,0,0,0)
