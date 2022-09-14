import numpy as np
import math
import time
import cv2

import camera
import client

STATE_DISTANCE = 50

servers = ["192.168.1.5"]
ports = [12007]
sockets = [0]

ROBOT = "Turtlebot"
# ROBOT = "Kuri"

if ROBOT == "Kuri":
    TURN_MAGNITUDE = 1.48
    FORWARD_MAGNITUDE = 0.3
    NO_MOVEMENT = [0,0,0,0,0,0]
    FORWARD = [FORWARD_MAGNITUDE,0,0,0,0,0]
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
    TURN_MAGNITUDE = 3.7
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

directions = ["UP","DOWN","LEFT","RIGHT"]

def get_cardinal_goal(back,direction):
    distance = 100
    goal = (back[0]+distance,back[1])
    if direction == "RIGHT":
        goal = (back[0]+distance,back[1])
    elif direction == "UP":
        goal = (back[0],back[1]+distance)
    elif direction == "LEFT":
        goal = (back[0]-distance,back[1])
    elif direction == "DOWN":
        goal = (back[0],back[1]-distance)
    return goal
    


def angle_to_goal(back,front,goal):
    a_term = math.atan2(front[1]-back[1],front[0]-back[0])
    b_term = math.atan2(goal[1]-back[1],goal[0]-back[0])
    ang = a_term - b_term

    if ang<-math.pi:
        ang+=2*math.pi
    elif ang>math.pi:
        ang-=2*math.pi
    return ang

def get_rotation_to_goal(angle):
    angle_proportion = angle/(math.pi/2)
    turn_velocity = angle_proportion * TURN_MAGNITUDE
    return [0,0,0,0,0,turn_velocity]

def dist(a,b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def get_distance_to_goal(back,front,goal):
    centre_point = [(back[0]+front[0])/2,(back[1]+front[1])/2]
    distance = dist(centre_point,goal)
    return distance

def get_forward_motion_to_goal(distance):
    #FORWARD_MAGNITUDE = min(np.random.rand()+0.5,1)
    distance_proportion = distance/STATE_DISTANCE
    forward_magnitude = distance_proportion * FORWARD_MAGNITUDE

    forward_magnitude = min(forward_magnitude,FORWARD_MAGNITUDE)
    print("MAX: {}, current: {}, distance: {}".format(FORWARD_MAGNITUDE,forward_magnitude,distance))

    return [forward_magnitude,0,0,0,0,0]

def send_movement_command(command):
    SLEEP_TIME = 2

    limitSet = command
    limitSet.append(cur_led)
    limitSet.append(HEAD_POSITION_UP)
    limitSet.append(EYES_SMILE)
    print(command)
    client.send_to_server(limitSet, servers[0], sockets[0]) #Sending sets to servers
    results = client.get_replies(sockets[0])
    print(results)
    time.sleep(SLEEP_TIME)

def angle_correct(cap,goal,front=(0,0),back=(0,0),max_tries=10,draw_circles=None):
    image,frame = camera.update_image(cap)
    front,back = camera.get_position(image,front=front,back=back)
    
    angle = angle_to_goal(back,front,goal)

    if draw_circles is not None:
        draw_circles["Front"] = front
        draw_circles["Back"] = back

    send_movement_command(get_rotation_to_goal(angle))
    
    
    #angle_tries += 1

    #camera.display_text(frame,"Angle: {} degrees".format(math.degrees(angle)))
    camera.draw_frame(frame,draw_cirlces=draw_circles)

    return front,back

def distance_correct(cap,goal,front=(0,0),back=(0,0),max_tries=10,draw_circles=None):
    image,frame = camera.update_image(cap)
    front,back = camera.get_position(image,front=front,back=back)
    distance = get_distance_to_goal(back,front,goal)

    if draw_circles is not None:
        draw_circles["Front"] = front
        draw_circles["Back"] = back

    command = get_forward_motion_to_goal(distance)
    send_movement_command(command)

    #move_tries += 1

    #camera.display_text(frame,"Distance: {}".format(distance))
    camera.draw_frame(frame,draw_cirlces=draw_circles)

    return front,back


def error_correct(cap,goal,goal_direction,angle_tolerance=0.15,distance_tolerance=10,max_tries=5,draw_circles=None):
    image,frame = camera.update_image(cap)
    camera.draw_frame(frame)

    front = (0,0)
    back = (0,0)

    while True:
        front,back = angle_correct(cap,goal,front=front,back=back,draw_circles=draw_circles)
        front,back = distance_correct(cap,goal,front=front,back=back,draw_circles=draw_circles)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break





def error_correct_1(cap,goal,goal_direction,angle_tolerance=0.15,distance_tolerance=10,max_tries=5,draw_circles=None):

    image,frame = camera.update_image(cap)
    camera.draw_frame(frame)
       
    # First, rotate to goal
    angle_tries = 0
    angle = angle_tolerance + 1
    front = (0,0)
    back = (0,0)
    #while angle>abs(angle_tolerance):
    while True:
        
        image,frame = camera.update_image(cap)
        front,back = camera.get_position(image,front=front,back=back)
        
        angle = angle_to_goal(back,front,goal)

        draw_circles["Front"] = front
        draw_circles["Back"] = back

        send_movement_command(get_rotation_to_goal(angle))
        
        
        #angle_tries += 1

        camera.display_text(frame,"Angle: {} degrees".format(math.degrees(angle)))
        camera.draw_frame(frame,draw_cirlces=draw_circles)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if angle_tries >= max_tries:
            print("FAILED TO GET ANGLE WITHIN TOLERANCE")
            break

    

    # Next, move to goal
    move_tries = 0
    distance = distance_tolerance + 1
    while True:
    #while distance > distance_tolerance:
        image,frame = camera.update_image(cap)
        front,back = camera.get_position(image,front=front,back=back)
        distance = get_distance_to_goal(back,front,goal)

        draw_circles["Front"] = front
        draw_circles["Back"] = back

        command = get_forward_motion_to_goal(distance)
        print("Hello")
        send_movement_command(command)

        #move_tries += 1

        camera.display_text(frame,"Distance: {}".format(distance))
        camera.draw_frame(frame,draw_cirlces=draw_circles)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if move_tries >= max_tries:
            print("FAILED TO GET DISTANCE WITHIN TOLERANCE")
            break

    # Finally, rotate to cardinal direction
    angle_tries = 0
    angle = angle_tolerance + 1
    new_goal = get_cardinal_goal(back,goal_direction)
    while True:
    #while  angle > angle_tolerance:
        image,frame = camera.update_image(cap)
        front,back = camera.get_position(image,front=front,back=back)
        angle = angle_to_goal(back,front,new_goal)

        draw_circles["Front"] = front
        draw_circles["Back"] = back

        send_movement_command(get_rotation_to_goal(angle))
        
        #angle_tries += 1

        camera.display_text(frame,"Angle: {} degrees".format(math.degrees(angle)))
        camera.draw_frame(frame,draw_cirlces=draw_circles)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if angle_tries >= max_tries:
            print("FAILED TO GET ANGLE WITHIN TOLERANCE")
            break

    

    
def run():
    client.create_connections(servers[0], ports[0], sockets[0])

    cap =  cap = cv2.VideoCapture(camera.CAMERA_NUMBER)

    goal = [400,350]
    goal_direction = "UP"

    draw_circles = {"Goal":goal}

    error_correct(cap,goal,goal_direction,draw_circles=draw_circles)

    while True:
        _,frame = camera.update_image(cap)
        camera.draw_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    client.close_connections(sockets[0])

if __name__ == "__main__":
    run()




    



