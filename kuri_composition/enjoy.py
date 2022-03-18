from GridWorld import *
from library import *
import matplotlib.pyplot as plt


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

start_position = (9,1)
T_positions = [(2,2), (2,7), (7,7), (7,2)]
env = GridWorld(MAP=MAP, T_positions=T_positions, start_position=start_position)
# env.render(agent=True)
# plt.show()

### Loading learned tasks

MAX = load_EQ("models/max.npy")
MIN = load_EQ("models/min.npy")
A = load_EQ("models/top.npy")
B = load_EQ("models/left.npy")

### Zero-shot composition
NEG = lambda EQ: NOT(EQ, EQ_max=MAX, EQ_min=MIN)
XOR = lambda EQ1, EQ2: OR(AND(EQ1,NEG(EQ2)),AND(EQ2,NEG(EQ1)))

P=EQ_P(AND(A,NOT(B)))

max_episodes = 100
max_steps = 50
for episode in range(max_episodes):
    state = env.reset()
    for step in range(max_steps):
        env.render(agent=True)
        plt.pause(0.00001)
        action = P[state]
        state, reward, done, _ = env.step(action)
        if done:
            break