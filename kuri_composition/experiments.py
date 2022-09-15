import numpy as np
import math
import time
import cv2

import motion
import camera
import client

servers = ["192.168.1.2"]
ports = [12007]
sockets = [0]

from GridWorld import *
from library import *
import matplotlib.pyplot as plt

from sympy.logic import SOPform, boolalg
from sympy import Symbol, symbols as Symbols

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--exp',
    default="t&~l",
    help="Task expression"
)
args = parser.parse_args()

start_position = (8,2)
start_direction = Directions.up
exp = args.exp.replace("t", "(nw | ne)")
exp = exp.replace("l", "(nw | sw)")
env = GridWorld(exp = exp, start_position=start_position, start_direction=start_direction)
print("Expression: ", env.exp)
print('Goals: ',len(env.goals))

### Loading learned skills
print("Loading learned skills")
values = {}
max_evf = load(env, "models/max.npy")
min_evf = load(env, "models/min.npy")
values['t'] = load(env, "models/top.npy")
values['l']  = load(env, "models/left.npy")
values['n']  = load(env, "models/n.npy")
values['s']  = load(env, "models/s.npy")
values['e']  = load(env, "models/e.npy")
values['w']  = load(env, "models/w.npy")

### Zero-shot composition
print("Zero-shot composition")
exp = sympify(args.exp, evaluate=False)
exp = boolalg.simplify_logic(exp)
evf = exp_evf(values, max_evf, min_evf, exp)

def rl_correct_after_every_step(cap,out,points,max_episodes=1,max_steps=50,metrics={}):
    for episode in range(max_episodes):
        state = env.reset()
        evf.reset(state)

        if "num_actions" not in metrics.keys():
            metrics["num_actions"] = 0

        for step in range(max_steps):
            # print(evf.get_value(state))
            #print(step)
            #img = env.render(agent=True, mode = "rgb_array")
            #cv2.imshow("env_render",cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            #plt.pause(0.00001)
            action = evf.get_action(state)
            last_state = state
            state, reward, done, _ = env.step(action)
            #print(state)

            #print("STEP: {}".format(action))
            
            if action == env.actions.up:
                motion.send_movement_command(cap,[motion.FORWARD_MAGNITUDE*1000,0,0,0,0,0])
            elif action == env.actions.left:
                motion.send_movement_command(cap,[0,0,0,0,0,motion.TURN_MAGNITUDE*1000])
            elif action == env.actions.right:
                motion.send_movement_command(cap,[0,0,0,0,0,-motion.TURN_MAGNITUDE*1000])
            elif action == env.actions.done:
                break
            else:
                print("Command not recognised. Must be w, a, d or q")
                continue

            _,frame = camera.update_image(cap)
            camera.draw_frame(frame,out)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Correct to expected state
            if done:
                state = last_state
            
            goal,goal_direction = camera.get_goal_from_state(state)
            draw_circles = {"Goal":goal}
            metrics = motion.error_correct(cap,out,goal,goal_direction,draw_circles=draw_circles,metrics=metrics)


            metrics["num_actions"] += 1

            if done:
                break

    return metrics,last_state


def rl_no_error_correct(cap,out,points,max_episodes=1,max_steps=50,metrics={}):
    for episode in range(max_episodes):
        state = env.reset()
        evf.reset(state)

        if "num_actions" not in metrics.keys():
            metrics["num_actions"] = 0

        for step in range(max_steps):
            # print(evf.get_value(state))
            #print(step)
            #img = env.render(agent=True, mode = "rgb_array")
            #cv2.imshow("env_render",cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            #plt.pause(0.00001)
            action = evf.get_action(state)
            last_state = state
            state, reward, done, _ = env.step(action)
            #print(state)

            #print("STEP: {}".format(action))
            
            if action == env.actions.up:
                motion.send_movement_command(cap,[motion.FORWARD_MAGNITUDE*1000,0,0,0,0,0])
            elif action == env.actions.left:
                motion.send_movement_command(cap,[0,0,0,0,0,motion.TURN_MAGNITUDE*1000])
            elif action == env.actions.right:
                motion.send_movement_command(cap,[0,0,0,0,0,-motion.TURN_MAGNITUDE*1000])
            elif action == env.actions.done:
                break
            else:
                print("Command not recognised. Must be w, a, d or q")
                continue

            _,frame = camera.update_image(cap)
            camera.draw_frame(frame,out)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            metrics["num_actions"] += 1

            if done:
                break

    return metrics,last_state
    

def run():
    client.create_connections(servers[0], ports[0], sockets[0])
    cap = cv2.VideoCapture(camera.CAMERA_NUMBER)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    filename = camera.get_output_filename("frame")
    out = cv2.VideoWriter(camera.directory+filename+camera.format, fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))
    points = camera.get_room_states()

    print("==Moving to Start State==")
    start_state = [start_position,start_direction]
    goal,goal_direction = camera.get_goal_from_state(start_state)
    draw_circles = {"Goal":goal}
    motion.error_correct(cap,out,goal,goal_direction,distance_tolerance=5,draw_circles=draw_circles)

    total_time_start = time.time()
    print("==Running==")
    #metrics,state = rl_no_error_correct(cap,out,points)
    metrics,state = rl_correct_after_every_step(cap,out,points)

    total_time = time.time() - total_time_start
    metrics["total_time"] = total_time

    final_goal,_ = camera.get_goal_from_state(state)
    image,frame = camera.update_image(cap)
    front,back = camera.get_position(image)
    distance = motion.get_distance_to_goal(back,front,final_goal)

    metrics["final_distance"] = distance


    print(metrics)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    client.close_connections(sockets[0])

if __name__ == "__main__":
    run()

