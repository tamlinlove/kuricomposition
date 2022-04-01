#!/usr/bin/env python
from socket import *
import rospy
from geometry_msgs.msg import Twist
from mobile_base_driver.msg import ChestLeds
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time
import pickle #Pickle used for data transfering.

def move(publisher, x, y, z, ax, ay, az):
    print("Executing a move")
    # Starts a new node
    #rospy.init_node('robot_cleaner', anonymous=True)
    #velocity_publisher = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=100)

    velocity_msg = Twist()
    print(velocity_msg)

    #Need to compensate for subscriber
    # Need this for the robot to do anything
    rate = rospy.Rate(5) # 10hz
    rate.sleep()

    velocity_msg.linear.x = x
    velocity_msg.linear.y = y
    velocity_msg.linear.z = z
    velocity_msg.angular.x = ax
    velocity_msg.angular.y = ay
    velocity_msg.angular.z = az

    publisher.publish(velocity_msg)
    #time.sleep(1)

    if ax != 0 or ay != 0 or az != 0:
        # Send 0 rotation
        time.sleep(1)
        velocity_msg.linear.x = 0
        velocity_msg.linear.y = 0
        velocity_msg.linear.z = 0
        velocity_msg.angular.x = 0
        velocity_msg.angular.y = 0
        velocity_msg.angular.z = 0
        publisher.publish(velocity_msg)

def head_move(publisher, positions):
    
    joint_msg = JointTrajectory()
    joint_point = JointTrajectoryPoint()

    rate = rospy.Rate(5)

    joint_point.positions = positions
    joint_point.velocities = [0,0]
    joint_point.accelerations = [0,0]
    joint_point.effort = [0,0]
    joint_point.time_from_start.secs = 1

    joint_msg.joint_names = ['head_1_joint','head_2_joint']
    joint_msg.points = [joint_point]
    
    publisher.publish(joint_msg)
    print(joint_msg)

def eye_move(publisher, position):
    
    joint_msg = JointTrajectory()
    joint_point = JointTrajectoryPoint()

    rate = rospy.Rate(5)

    joint_point.positions = [position]
    joint_point.velocities = [0]
    joint_point.accelerations = [0]
    joint_point.effort = [0]
    joint_point.time_from_start.secs = 1

    joint_msg.joint_names = ['eyelids_joint']
    joint_msg.points = [joint_point]
    
    publisher.publish(joint_msg)
    print(joint_msg)


def update_light(publisher, colour):
    print("Updating chest light colour to: ", colour)

    light_msg = ChestLeds()

    rate = rospy.Rate(15)

    interp_factor = 1.0*(colour-8)/2.0
    for i in range(15):
        light_msg.leds[i].red = int(255 * (1 - interp_factor))
        light_msg.leds[i].blue = int(255 * interp_factor)
    publisher.publish(light_msg)

if __name__ == '__main__':
    #Set up ros stuff
    rospy.init_node('robot_cleaner',anonymous=True)
    vpub = rospy.Publisher('/mobile_base/commands/velocity',Twist,queue_size=100)
    lpub = rospy.Publisher("/mobile_base/commands/chest_leds",ChestLeds, latch=True, queue_size=1)
    hpub = rospy.Publisher('/head_controller/command',JointTrajectory,queue_size=100)
    epub = rospy.Publisher('/eyelids_controller/command',JointTrajectory,queue_size=100)
    #time.sleep(1)
    #Set server socket settings.
    serverPort = 12007
    serverSocket = socket(AF_INET, SOCK_STREAM)
    serverSocket.bind(("",serverPort))
    serverSocket.listen(1)
    print("The server is ready to receive")
    while 1:
        connectionSocket, addr = serverSocket.accept() #connecting to
        while True:
            pickledLimits = connectionSocket.recv(2048)#Recieve data in Pickle form. 1024
            limits = pickle.loads(pickledLimits)
            print("Received: ", limits)
            x, y, z, ax, ay, az, value, hpos, epos = limits
            print("move and value: ({},{},{},{},{},{},{})".format(x, y, z, ax, ay, az,value))
            toSend = [1, 21]
            connectionSocket.send(pickle.dumps(toSend))
            
            try:
                eye_move(epub, epos)
                head_move(hpub, hpos)
                move(vpub, x, y, z, ax, ay, az)
                update_light(lpub, value)
                toSend = [1, 21]
                connectionSocket.send(pickle.dumps(toSend))
            except rospy.ROSInterruptException: pass
        
        connectionSocket.close()#Closing connections.
