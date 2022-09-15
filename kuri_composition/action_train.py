import time
import client
import motion

import matplotlib.pyplot as plt

#servers = ["localhost"]
servers = ["192.168.1.2"]
ports = [12007]
sockets = [0]

ROBOT = "Turtlebot"
# ROBOT = "Kuri"

if ROBOT == "Kuri":
    TURN_MAGNITUDE = 1.48
    NO_MOVEMENT = [0,0,0,0,0,0]
    FORWARD = [0.3,0,0,0,0,0]
    TURN_LEFT = [0,0,0,0,0,TURN_MAGNITUDE]
    TURN_RIGHT = [0,0,0,0,0,-TURN_MAGNITUDE]
    HEAD_POSITION_STRAIGHT = [0,0]
    HEAD_POSITION_LEFT = [1,0]
    HEAD_POSITION_RIGHT = [-1,0]
    HEAD_POSITION_DOWN = [0,1]
    HEAD_POSITION_UP = [0,-1]
    EYES_CLOSED = 1
    EYES_OPEN = 0
    EYES_SMILE = -0.3
    SLEEP_TIME = 1.2
if ROBOT == "Turtlebot":
    TURN_MAGNITUDE_LEFT = 3.7
    TURN_MAGNITUDE_RIGHT = 3.7
    FORWARD_MAGNITUDE = 0.7
    NO_MOVEMENT = [0,0,0,0,0,0]
    FORWARD = [FORWARD_MAGNITUDE,0,0,0,0,0]
    TURN_LEFT = [0,0,0,0,0,TURN_MAGNITUDE_LEFT]
    TURN_RIGHT = [0,0,0,0,0,-TURN_MAGNITUDE_RIGHT]
    HEAD_POSITION_STRAIGHT = [0,0]
    HEAD_POSITION_LEFT = [1,0]
    HEAD_POSITION_RIGHT = [-1,0]
    HEAD_POSITION_DOWN = [0,1]
    HEAD_POSITION_UP = [0,-1]
    EYES_CLOSED = 1
    EYES_OPEN = 0
    EYES_SMILE = -0.3
    SLEEP_TIME = 1.5

cur_led = 9

def reset_behaviour():
    print("Running reset behaviour")
    limitSet = NO_MOVEMENT.copy()
    limitSet.append(cur_led)
    limitSet.append(HEAD_POSITION_DOWN)
    limitSet.append(EYES_CLOSED)
    client.send_to_server(limitSet, servers[0], sockets[0]) #Sending sets to servers
    results = client.get_replies(sockets[0])
    time.sleep(2)

def startup_behaviour():
    print("Running startup behaviour")
    limitSet = NO_MOVEMENT.copy()
    limitSet.append(cur_led)
    limitSet.append(HEAD_POSITION_STRAIGHT)
    limitSet.append(EYES_OPEN)
    client.send_to_server(limitSet, servers[0], sockets[0]) #Sending sets to servers
    results = client.get_replies(sockets[0])
    time.sleep(2)

def end_behaviour():
    print("Running end behaviour")
    limitSet = NO_MOVEMENT.copy()
    limitSet.append(cur_led)
    limitSet.append(HEAD_POSITION_UP)
    limitSet.append(EYES_SMILE)
    client.send_to_server(limitSet, servers[0], sockets[0]) #Sending sets to servers
    results = client.get_replies(sockets[0])
    time.sleep(2)

def move_forward():
    limitSet = [motion.FORWARD_MAGNITUDE*1000,0,0,0,0,0]
    limitSet.append(cur_led)
    limitSet.append(HEAD_POSITION_UP)
    limitSet.append(EYES_SMILE)
    client.send_to_server(limitSet, servers[0], sockets[0]) #Sending sets to servers
    results = client.get_replies(sockets[0])
    time.sleep(SLEEP_TIME)

def turn_left():
    limitSet = [0,0,0,0,0,motion.TURN_MAGNITUDE*1000]
    limitSet.append(cur_led)
    limitSet.append(HEAD_POSITION_UP)
    limitSet.append(EYES_SMILE)
    client.send_to_server(limitSet, servers[0], sockets[0]) #Sending sets to servers
    results = client.get_replies(sockets[0])
    time.sleep(SLEEP_TIME)

def turn_right():
    limitSet = [0,0,0,0,0,-motion.TURN_MAGNITUDE*1000]
    limitSet.append(cur_led)
    limitSet.append(HEAD_POSITION_UP)
    limitSet.append(EYES_SMILE)
    client.send_to_server(limitSet, servers[0], sockets[0]) #Sending sets to servers
    results = client.get_replies(sockets[0])
    time.sleep(SLEEP_TIME)


def run():
    #move_forward()
    turn_right()
    #turn_left()

if __name__ == "__main__":
    client.create_connections(servers[0], ports[0], sockets[0])
    #reset_behaviour()
    #startup_behaviour()
    run()
    #end_behaviour()
    # closing connections to servers
    client.close_connections(sockets[0])
