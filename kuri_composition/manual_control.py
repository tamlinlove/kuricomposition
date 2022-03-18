#!/usr/bin/env python3

import time
import argparse
import numpy as np

from GridWorld import *
from gym_minigrid.window import Window
from matplotlib import pyplot as plt

def redraw(img):
    img = env.render(mode='rgb_array', agent=True)
    
    window.show_img(img)
    # window.fig.savefig("envs/fourrooms.pdf", bbox_inches='tight')

def reset():
    if args.seed:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f, action=%.2f' % (env.step_count, reward, action))

    if done:
        print('done!')
        reset()
    else:
        redraw(obs) 

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'enter':
        step(env.actions.done)
        return
    
    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.up)
        return
    
    if False: # env.use_cardinal_actions:
        if event.key == 'down':
            step(env.actions.down)
            return

parser = argparse.ArgumentParser()
parser.add_argument(
    '--env_key',
    default="MiniGrid-Task-FourRooms-v0",
    help="Environment"
)
parser.add_argument(
    '--exp',
    default=None,
    help="Task expression"
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=None
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

args = parser.parse_args()
# env = make_env(env_key=args.env_key, exp=args.exp, seed=args.seed)
# print("Expression: ", env.exp)
# print('Goals: ',len(env.goals))
MAP =   "LT T RT LT T RT\n" \
        "L 0 0 0 0 R\n" \
        "LD 0 RD LD 0 RD\n" \
        "LT 0 RT LT 0 RT\n" \
        "L 0 0 0 0 R\n" \
        "LD D RD LD D RD"

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

env = GridWorld(MAP=MAP)
T_positions = [(2,2), (2,7), (7,7), (7,2)]
goals=[(2,2),(2,7)]
env = GridWorld(MAP=MAP, goals=goals, T_positions=T_positions)

window = Window('gym_minigrid')
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
