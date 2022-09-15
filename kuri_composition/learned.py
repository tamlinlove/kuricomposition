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