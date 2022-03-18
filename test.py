#!/usr/bin/env python
from socket import *
import rospy
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory
import time
import pickle #Pickle used for data transfering.

def move_head(publisher,p1,p2):
    print("Moving head")

    head_msg = JointTrajectory()

    # Need this for the robot to do anything
    rate = rospy.Rate(5) # 10hz
    rate.sleep()

    head_msg.positions = [p1,p2]

    publisher.publish(head_msg)

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

    velocity_msg.linear.x = x
    velocity_msg.linear.y = y
    velocity_msg.linear.z = z
    velocity_msg.angular.x = ax
    velocity_msg.angular.y = ay
    velocity_msg.angular.z = az

    publisher.publish(velocity_msg)
    #time.sleep(1)


if __name__ == '__main__':
    #Set up ros stuff
    rospy.init_node('robot_cleaner',anonymous=True)
    vpub = rospy.Publisher('/mobile_base/commands/velocity',Twist,queue_size=100)
    hpub = rospy.Publisher('/head_controller/command',JointTrajectory,queue_size=100)
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
            pickledLimits = connectionSocket.recv(1024)#Recieve data in Pickle form.
            limits = pickle.loads(pickledLimits)
            x, y, z, ax, ay, az = limits
            print("move({},{},{},{},{},{})".format(x, y, z, ax, ay, az))
            toSend = [1, 21]
            connectionSocket.send(pickle.dumps(toSend))
            
            try:
                '''
                move(vpub, x, y, z, ax, ay, az)
                toSend = [1, 21]
                connectionSocket.send(pickle.dumps(toSend))
                '''
                move_head(hpub,1,1)
                toSend = [1, 21]
                connectionSocket.send(pickle.dumps(toSend))
            except rospy.ROSInterruptException: pass
        
        connectionSocket.close()#Closing connections.
