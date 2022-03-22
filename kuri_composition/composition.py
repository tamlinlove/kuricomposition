from GridWorld import *
from library import *
import matplotlib.pyplot as plt

from sympy.logic import SOPform, boolalg
from sympy import Symbol, symbols as Symbols

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--exp',
    default="",
    help="Task expression"
)
args = parser.parse_args()

exp = args.exp.replace("t", "(nw | ne)")
exp = exp.replace("l", "(nw | sw)")
env = GridWorld(exp = exp)
print("Expression: ", env.exp)
print('Goals: ',len(env.goals))

### Loading learned skills
print("Loading learned skills")
values = {}
max_evf = load_EQ("models/max.npy")
min_evf = load_EQ("models/min.npy")
values['t'] = load_EQ("models/top.npy")
values['l']  = load_EQ("models/left.npy")
values['n']  = load_EQ("models/n.npy")
values['s']  = load_EQ("models/s.npy")
values['e']  = load_EQ("models/e.npy")
values['w']  = load_EQ("models/w.npy")

### Zero-shot composition
print("Zero-shot composition")
exp = sympify(args.exp, evaluate=False)
exp = boolalg.simplify_logic(exp)
EQ = exp_value(values, max_evf, min_evf, exp)
env.render( P=EQ_P(EQ), V = EQ_V(EQ))
plt.show()