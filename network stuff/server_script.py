#!/usr/bin/env python
from socket import *
import pickle #Pickle used for data transfering.


if __name__ == '__main__':
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
            '''
            try:
                move(x, y, z, ax, ay, az)
                toSend = [1, 21]
                connectionSocket.send(pickle.dumps(toSend))
            except rospy.ROSInterruptException: pass
            '''
        connectionSocket.close()#Closing connections.
