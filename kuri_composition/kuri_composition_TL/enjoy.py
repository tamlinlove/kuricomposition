from env.GridWorld import *
from library import *
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from env.window import Window

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

#######################################################################################

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
            
            if done:
                break

if __name__ == "__main__":
    run()
