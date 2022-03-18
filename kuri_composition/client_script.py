import time
import client

from GridWorld import *
from library import *
import matplotlib.pyplot as plt

#servers = ["localhost"]
servers = ["192.168.1.2"]
ports = [12007]
sockets = [0]

FORWARD = [0.3,0,0,0,0,0]
TURN_LEFT = [0,0,0,0,0,1.55]
TURN_RIGHT = [0,0,0,0,0,-1.55]


MAP =   "LT T T T RT LT T T T RT\n" \
        "L 0 0 0 R L 0 0 0 R\n" \
        "L 0 0 0 0 0 0 0 0 R\n" \
        "L 0 0 0 R L 0 0 0 R\n" \
        "LD D 0 D RD LD D 0 D RD\n" \
        "LT T 0 T RT LT T 0 T RT\n" \
        "L 0 0 0 R L 0 0 0 R\n" \
        "L 0 0 0 0 0 0 0 0 R\n" \
        "L 0 0 0 R L 0 0 0 R\n" \
        "LD D D D RD LD D D D RD"

start_position = (8,1)
T_positions = [(2,2), (2,7), (7,7), (7,2)]
env = GridWorld(MAP=MAP, T_positions=T_positions, start_position=start_position)

MAX = load_EQ("models/max.npy")
MIN = load_EQ("models/min.npy")
A = load_EQ("models/top.npy")
B = load_EQ("models/left.npy")

### Zero-shot composition
NEG = lambda EQ: NOT(EQ, EQ_max=MAX, EQ_min=MIN)
XOR = lambda EQ1, EQ2: OR(AND(EQ1,NEG(EQ2)),AND(EQ2,NEG(EQ1)))

P=EQ_P(AND(A,NOT(B)))

max_episodes = 1
max_steps = 50

def run():
    for episode in range(max_episodes):
        state = env.reset()
        for step in range(max_steps):
            env.render(agent=True)
            plt.pause(0.00001)
            action = P[state]
            state, reward, done, _ = env.step(action)

            if action == env.actions.up:
                limitSet = FORWARD
            elif action == env.actions.left:
                limitSet = TURN_LEFT
            elif action == env.actions.right:
                limitSet = TURN_RIGHT
            elif action == env.actions.done:
                break
            else:
                print("Command not recognised. Must be w, a, d or q")
                continue

            client.send_to_server(limitSet, servers[0], sockets[0]) #Sending sets to servers
            results = client.get_replies(sockets[0])
            time.sleep(1)

            if done:
                break

if __name__ == "__main__":
    client.create_connections(servers[0], ports[0], sockets[0])

    run()
    # closing connections to servers
    client.close_connections(sockets[0])
