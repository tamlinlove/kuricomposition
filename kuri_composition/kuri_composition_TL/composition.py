from env.GridWorld import *
from library import *
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from env.window import Window


### Deffine environment
gridworld_objects =  {
    '1room': roomA(),
    '2room': roomB(),
    '3room': roomC(),
    '4room': roomD(),
}
env = GridWorld(gridworld_objects=gridworld_objects)

env = Task(env)
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

class SM1(SM_base):
    name = 'patrol'
    terminal_states = set([5])
    skills = skills
    transitions = {
        0: {
            "!R1":[0,"R1"],
            "R1":[1, "R2"],
        },
        1: {
            "!R2":[1,"R2"],
            "R2":[2, "R3"],
        },
        2: {
            "!R3":[2,"R3"],
            "R3":[3, "R4"],
        },
        3: {
            "!R4":[3,"R4"],
            "R4":[4, "R1"],
        },
        4: {
            "!R1":[4, "R1"],
            "R1":[5, "MAX"],
        },
        5: {
            "MAX": [5, "MAX"],
        }
    }


class SM2(SM_base):
    name = 'patrol'
    terminal_states = set([5])
    skills = skills
    transitions = {
        0: {
            "!DN":[0,"DN"],
            "DN":[1, "DE"],
        },
        1: {
            "!DE":[1,"DE"],
            "DE":[2, "DS"],
        },
        2: {
            "!DS":[2,"DS"],
            "DS":[3, "DW"],
        },
        3: {
            "!DW":[3,"DW"],
            "DW":[4, "DN"],
        },
        4: {
            "!DN":[4, "DN"],
            "DN":[5, "MAX"],
        },
        5: {
            "MAX": [5, "MAX"],
        }
    }


skill_machine = SM0() #SM_THEN([SM_UNTIL(SM2(), SM_not_tM()), SM_UNTIL(SM1(), SM_not_tO()), SM_UNTIL(SM0(), SM_tO())]) # SM_THEN([SM1(),SM2()]) #SM_THEN([SM1(),SM2()]*4) # [SM1, SM2, SM_THEN([SM1(),SM2()])] 

### Visualize
print("Visualize")

def visualize(skill_machine, save_trajectories = None):

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

    transitions = []
    trajectories = []
    start_position = env.start_position
    positions = [(0,9)]

    for position in positions:
        env.env.start_position = [position]
        
        state = env.reset()

        Q = skill_machine.reset()
        behaviour_policy =  epsilon_greedy_generalised_policy_improvement(env, Q)
        
        trajectory = [] 
        for t in range(300):
            if save_trajectories:
                fig = env.render(env_map=True)
                width, height = fig.get_size_inches() * fig.get_dpi()
                image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
                image = Image.fromarray(np.uint8(image))
                # image = image.resize((int(width)//2, int(height)//2))
                image = image.convert("P",palette=Image.ADAPTIVE)
                trajectory.append(image)
            else:
                image = env.render(env_map=True, mode='rgb_array')
                window.show_img(image)

            
            goal = env.get_goal(state)     
            print(state,goal) 
            Q = skill_machine.step(state, goal)
            behaviour_policy =  epsilon_greedy_generalised_policy_improvement(env, Q)
            window.set_caption(skill_machine.skill)
            print(skill_machine.state, skill_machine.skill)
            
            probs = behaviour_policy(state, epsilon = 0)
            action = probs.argmax()             
            state_, reward, done, _ = env.step(action)    
            transitions.append(action)                 
            state = state_            
            if done:
                break
            if window.closed:
                return
        
        if save_trajectories:
            trajectories += trajectory[:-1]
    env.start_position = start_position
    print(transitions)

    if save_trajectories:
        name = skill_machine.name
        trajectories[0].save(save_trajectories+name+'.gif',
                    save_all=True, append_images=trajectories[1:], optimize=False, duration=10, loop=0)

save_trajectories = None #'/home-mscluster/gnanguetasse/skill-machines/four_rooms/plots/trajectories/'
visualize(skill_machine, save_trajectories=save_trajectories)