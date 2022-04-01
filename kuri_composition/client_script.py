import time
import client

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

#servers = ["localhost"]
servers = ["192.168.1.2"]
ports = [12007]
sockets = [0]

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

cur_led = 9

start_position = (9,1)
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

max_episodes = 1
max_steps = 50

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

def run():
    for episode in range(max_episodes):
        state = env.reset()
        evf.reset(state)
        for step in range(max_steps):
            # print(evf.get_value(state))
            print(step)
            env.render(agent=True)
            plt.pause(0.00001)
            action = evf.get_action(state)
            state, reward, done, _ = env.step(action)
            
            if action == env.actions.up:
                limitSet = FORWARD.copy()
                limitSet.append(float(evf.get_value(state)))
                limitSet.append(HEAD_POSITION_STRAIGHT)
                limitSet.append(EYES_OPEN)
                cur_led = float(evf.get_value(state))
            elif action == env.actions.left:
                limitSet = TURN_LEFT.copy()
                limitSet.append(float(evf.get_value(state)))
                limitSet.append(HEAD_POSITION_LEFT)
                limitSet.append(EYES_OPEN)
                cur_led = float(evf.get_value(state))
            elif action == env.actions.right:
                limitSet = TURN_RIGHT.copy()
                limitSet.append(float(evf.get_value(state)))
                limitSet.append(HEAD_POSITION_RIGHT)
                limitSet.append(EYES_OPEN)
                cur_led = float(evf.get_value(state))
            elif action == env.actions.done:
                break
            else:
                print("Command not recognised. Must be w, a, d or q")
                continue

            print('sending: ', limitSet)
            client.send_to_server(limitSet, servers[0], sockets[0]) #Sending sets to servers
            results = client.get_replies(sockets[0])
            time.sleep(SLEEP_TIME)

            if done:
                break

if __name__ == "__main__":
    client.create_connections(servers[0], ports[0], sockets[0])
    reset_behaviour()
    startup_behaviour()
    run()
    end_behaviour()
    # closing connections to servers
    client.close_connections(sockets[0])
