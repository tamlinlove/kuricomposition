import time
import client

from env.GridWorld import *
from library import *
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from env.window import Window

#servers = ["localhost"]
servers = ["192.168.1.3"]
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
    TURN_MAGNITUDE = 3.2
    FORWARD_MAGNITUDE = 0.42
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
    SLEEP_TIME = 0.5

cur_led = 9

start_position = (9,1)
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

#######################################################################################

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
    max_episodes = 1
    max_steps = 50
    for episode in range(max_episodes):
        state = env.reset()
        Q = skill_machine.reset()
        behaviour_policy =  epsilon_greedy_generalised_policy_improvement(env, Q)
        for step in range(max_steps):
            goal = env.get_goal(state)     
            Q = skill_machine.step(state, goal)
            behaviour_policy =  epsilon_greedy_generalised_policy_improvement(env, Q)
            window.set_caption("SM state: {}, SM skill: {}".format(skill_machine.state,skill_machine.skill))
            window.show_img(env.render(env_map=True, mode='rgb_array'))
            
            probs = behaviour_policy(state, epsilon = 0)
            action = probs.argmax()             
            state_, reward, done, _ = env.step(action)                 
            state = state_            
            
            if action == env.actions.up:
                limitSet = FORWARD.copy()
                limitSet.append(float(Q[state][goal].max()))
                limitSet.append(HEAD_POSITION_STRAIGHT)
                limitSet.append(EYES_OPEN)
                cur_led = float(Q[state][goal].max())
            elif action == env.actions.left:
                limitSet = TURN_LEFT.copy()
                limitSet.append(float(Q[state][goal].max()))
                limitSet.append(HEAD_POSITION_LEFT)
                limitSet.append(EYES_OPEN)
                cur_led = float(Q[state][goal].max())
            elif action == env.actions.right:
                limitSet = TURN_RIGHT.copy()
                limitSet.append(float(Q[state][goal].max()))
                limitSet.append(HEAD_POSITION_RIGHT)
                limitSet.append(EYES_OPEN)
                cur_led = float(Q[state][goal].max())
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
