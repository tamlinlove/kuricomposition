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
    '--exp',
    default=None,
    help="Task expression"
)

args = parser.parse_args()
env = GridWorld(exp=args.exp)
print("Expression: ", env.exp)
print('Goals: ',len(env.goals))

window = Window('Four rooms')
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
