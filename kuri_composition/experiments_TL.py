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

from kuri_composition_TL.env.GridWorld import *
from kuri_composition_TL.library import *
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from kuri_composition_TL.env.window import Window


start_position = (8,2)
start_direction = Directions.up
start_position = (0,9)
start_direction = Directions.up

### Deffine environment
gridworld_objects =  {
    '1room': roomA(),
    '2room': roomB(),
    '3room': roomC(),
    '4room': roomD(),
}
env = GridWorld(gridworld_objects=gridworld_objects)

env = Task(env, start_position=start_position, start_direction=start_direction)
print(len(env.possiblePositions), len(env.goals), len(env.possiblePositions)*len(env.goals))

### Loading learned EVFs
print("Loading learned EVFs")

R1, stats = np.load('models/r1.npy', allow_pickle=True)
R1 = EQ_load(R1)
R2, stats = np.load('models/r2.npy', allow_pickle=True)
R2 = EQ_load(R2)
R3, stats = np.load('models/r3.npy', allow_pickle=True)
R3 = EQ_load(R3)
R4, stats = np.load('models/r4.npy', allow_pickle=True)
R4 = EQ_load(R4)
DN, stats = np.load('models/dn.npy', allow_pickle=True)
DN = EQ_load(DN)
DE, stats = np.load('models/de.npy', allow_pickle=True)
DE = EQ_load(DE)
DS, stats = np.load('models/ds.npy', allow_pickle=True)
DS = EQ_load(DS)
DW, stats = np.load('models/dw.npy', allow_pickle=True)
DW = EQ_load(DW)

max_, stats = np.load('models/max.npy', allow_pickle=True)
max_ = EQ_load(max_)
min_, stats = np.load('models/min.npy', allow_pickle=True)
min_ = EQ_load(min_)
NEG = lambda EQ: NOT(EQ,EQ_max=max_,EQ_min=min_)

# ### Visualize values and policies
# print("Visualize values and policies")

# render_learned(env, P=EQ_P(R1), V = EQ_V(R1))
# plt.show()
# render_learned(env, P=EQ_P(DN), V = EQ_V(DN))
# plt.show()

### Skill machines
print("Skill machines")

skills = {
    "R1": R1,
    "R2": R2,
    "R3": R3,
    "R4": R4,
    "!R1": NEG(R1),
    "!R2": NEG(R2),
    "!R3": NEG(R3),
    "!R4": NEG(R4),
    "DN": DN,
    "!DN": NEG(DN),
    "DE": DE,
    "!DE": NEG(DE),
    "DW": DW,
    "!DW": NEG(DW),
    "DS": DS,
    "!DS": NEG(DS),
    "(R1.!DN)": AND(R1,NEG(DN)),
    "!(R1.!DN)": NEG(AND(R1,NEG(DN))),
    "MAX": max_,    
    "MIN": min_,   
}


class SM0(SM_base):
    name = 'patrol'
    terminal_states = set([1])
    skills = skills
    transitions = {
        0: {
            "!(R1.!DN)":[0, "(R1.!DN)"],
            "(R1.!DN)":[1, "MAX"],
        },
        1: {
            "MAX": [1, "MAX"],
        }
    }

skill_machine = SM0()

### Visualisation
window = Window('Office-World: ' + skill_machine.name)
def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        env.reset()
        return
window.reg_key_handler(key_handler)
# window.fig.set_size_inches(10, 6)
window.fig.set_size_inches(7, 4)

def rl_correct_after_every_step(cap,out,points,max_episodes=1,max_steps=50,metrics={}):
    for episode in range(max_episodes):
        state = env.reset()
        Q = skill_machine.reset()
        behaviour_policy =  epsilon_greedy_generalised_policy_improvement(env, Q)
        sm_state = skill_machine.state

        if "num_actions" not in metrics.keys():
            metrics["num_actions"] = 0

        for step in range(max_steps):
            goal = env.get_goal(state)     
            Q = skill_machine.step(state, goal)
            behaviour_policy =  epsilon_greedy_generalised_policy_improvement(env, Q)
            # window.set_caption("SM state: {}, SM skill: {}".format(skill_machine.state,skill_machine.skill))
            # window.show_img(env.render(env_map=True, mode='rgb_array'))
            # img = env.render(env_map=True, mode='rgb_array')
            # cv2.imshow("env_render",cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            probs = behaviour_policy(state, epsilon = 0)
            action = probs.argmax()             
            last_state = state
            state, reward, done, _ = env.step(action)

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
            
            if sm_state != skill_machine.state:
                sm_state = skill_machine.state
                
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

