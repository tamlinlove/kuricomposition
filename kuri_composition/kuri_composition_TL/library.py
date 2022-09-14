import numpy as np
import gym
from collections import defaultdict
from itertools import chain, combinations
from matplotlib import pyplot as plt
from copy import deepcopy

#########################################################################################
def Q_equal(Q1,Q2,epsilon=1e-8):    
    for state in Q1:
        for action in range(len(Q1[state])): 
            v1 = Q1[state][action]
            v2 = Q2[state][action]
            if abs(v1-v2)>epsilon:
                return False
    return True

def EQ_equal(EQ1,EQ2,epsilon=1e-8):    
    for state in EQ1:
        for goal in EQ1[state]:
            for action in range(len(EQ1[state][goal])): 
                v1 = EQ1[state][goal][action]
                v2 = EQ2[state][goal][action]
                if abs(v1-v2)>epsilon:
                    return False
    return True

def Q_copy(Q1,Q2):    
    for state in Q1:
        Q2[state] = Q1[state].copy()

def EQ_copy(EQ1,EQ2):    
    for state in EQ1:
        for goal in EQ1[state]:
            EQ2[state][goal] = EQ1[state][goal].copy()

#########################################################################################
def epsilon_greedy_policy_improvement(env, Q, epsilon=1):
    """
    Implements policy improvement by acting epsilon-greedily on Q

    Arguments:
    env -- environment with which agent interacts
    Q -- Action function for current policy
    epsilon -- probability

    Returns:
    policy_improved -- Improved policy
    """
        
    def policy_improved(state, epsilon = epsilon, Q=Q):
        probs = np.ones(env.action_space.n, dtype=float)*(epsilon/env.action_space.n)
        best_action = np.random.choice(np.flatnonzero(Q[state] == Q[state].max())) #np.argmax(Q[state]) #
        probs[best_action] += 1.0 - epsilon
        return probs

    return policy_improved

def epsilon_greedy_generalised_policy_improvement(env, Q, epsilon = 1):
    """
    Implements generalised policy improvement by acting epsilon-greedily on Q

    Arguments:
    env -- environment with which agent interacts
    Q -- Action function for current policy

    Returns:
    policy_improved -- Improved policy
    """
    
    def policy_improved(state, goal = None, epsilon = epsilon, Q=Q):
        probs = np.ones(env.action_space.n, dtype=float)*(epsilon/env.action_space.n)
        values = [Q[state][goal]] if goal else [Q[state][goal] for goal in Q[state].keys()]
        if len(values)==0:
            best_action = np.random.randint(env.action_space.n)
        else:
            values = np.max(values,axis=0)
            best_action = np.random.choice(np.flatnonzero(values == values.max()))
        probs[best_action] += 1.0 - epsilon
        return probs

    return policy_improved

def evaluate(env, Q, gamma=0.9, render='trajectory'):
    G=0
    behaviour_policy =  epsilon_greedy_policy_improvement(env, Q)
    start_position = env.start_position
    positions = [(3,3),(3,9),(9,3),(9,9)]
    for i in range(100):      
        # if not positions:
        #     break 
        # env.start_position = positions.pop()
        state = env.reset()
        for t in range(30):
            if render:
                if render:
                    fig = env.render()
                    plt.pause(0.00001)
            
            probs = behaviour_policy(state, epsilon = 0)
            action = np.random.choice(np.arange(len(probs)), p=probs)   
            print(state,Q[state]) 
            state_, reward, done, _ = env.step(action) 
            G += (gamma**t)*reward
            state = state_
            
            if done:
                break
    env.start_position = start_position
    return G/100

#########################################################################################
def Q_learning(env, Q_init=None, Q_optimal=None, gamma=1, epsilon=0.1, alpha=0.1, maxiter=100, maxstep=100):
    """
    Implements Q_learning

    Arguments:
    env -- environment with which agent interacts
    gamma -- discount factor
    alpha -- learning rate
    maxiter -- maximum number of episodes

    Returns:
    Q -- New estimate of Q function
    """
    Q = Q_init if Q_init else defaultdict(lambda: np.zeros(env.action_space.n)+env.rmax)
    Q_target = defaultdict(lambda: np.zeros(env.action_space.n)+env.rmax)
    behaviour_policy =  epsilon_greedy_policy_improvement(env, Q, epsilon = epsilon)
    
    stop_cond = lambda k: k > maxiter
    if Q_optimal:
        stop_cond = lambda k: k>1000 and Q_equal(Q_optimal,Q)
                
    stats = {"E":[], "R":[], "T":0}
    k=0
    T=0
    state = env.reset()
    stats["R"].append(0)
    while not stop_cond(k):
        probs = behaviour_policy(state, epsilon = epsilon)
        action = np.random.choice(np.arange(len(probs)), p=probs)    
        state_, reward, done, _ = env.step(action)
        
        stats["R"][k] += reward
        
        G = 0 if done else np.max(Q[state_])
        TD_target = reward + gamma*G
        TD_error = TD_target - Q[state][action]
        Q[state][action] = Q[state][action] + alpha*TD_error
        
        state = state_
        T+=1
        if done:            
            if k%100==0:
                print(k, len(Q), np.mean(stats["R"][-100:]))
                            
            state = env.reset()
            stats["R"].append(0)
            k+=1
    stats["T"] = T
    
    return Q, stats

def Goal_Oriented_Q_learning(env, T_states=None, Q_optimal=None, gamma=1, epsilon=0.1, alpha=0.1, maxiter=100, maxstep=100):
    """
    Implements Goal Oriented Q_learning

    Arguments:
    env -- environment with which agent interacts
    gamma -- discount factor
    alpha -- learning rate
    maxiter -- maximum number of episodes

    Returns:
    Q -- New estimate of Q function
    """
    N = min(env.rmin, (env.rmin-env.rmax)*env.diameter)
    Q = defaultdict(lambda: defaultdict(lambda: np.zeros(env.action_space.n)))
    Q_target = defaultdict(lambda: defaultdict(lambda: np.zeros(env.action_space.n)))
    behaviour_policy =  epsilon_greedy_generalised_policy_improvement(env, Q, epsilon = epsilon)
    
    goals={} # Goals memory
    if T_states:
        for state in T_states:
            goals[str(state)]=0
    
    stop_cond = lambda k: k > maxiter
    if Q_optimal:
        stop_cond = lambda k: k>1000 and EQ_equal(Q,Q_target)
                
    stats = {"E":[], "R":[], "T":0}
    k=0
    T=0
    state = env.reset()
    stats["R"].append(0)
    while not stop_cond(k):
        probs = behaviour_policy(state, epsilon = epsilon)
        action = np.random.choice(np.arange(len(probs)), p=probs)            
        state_, reward, done, _ = env.step(action)
        
        stats["R"][k] += (gamma**T)*reward
        
        if done:
            goals[state] = 0
        
        for goal in goals.keys():
            if state != goal and done:  
                reward_ = N
            else:
                reward_ = reward
            
            G = 0 if done else np.max(Q[state_][goal])
            TD_target = reward_ + gamma*G
            TD_error = TD_target - Q[state][goal][action]
            Q[state][goal][action] = Q[state][goal][action] + alpha*TD_error
                
        state = state_
        T+=1
        if done:  
            if k%100==0:
                print(k, len(Q), len(goals), np.mean(stats["R"][-100:]))
            
            EQ_copy(Q,Q_target)
            
            state = env.reset()
            stats["R"].append(0)
            stats["T"] += T
            k+=1
            T=0

    return Q, stats

def GOAL(env, Q_init=None, gamma=1, epsilon=0.1, alpha=0.1, maxiter=100, maxstep=100):
    """
    Implements Goal Oriented Q_learning V2

    Arguments:
    env -- environment with which agent interacts
    gamma -- discount factor
    alpha -- learning rate
    maxiter -- maximum number of episodes

    Returns:
    Q -- New estimate of Q function
    """
    N = min(env.rmin, (env.rmin-env.rmax)*abs(env.rmax))
    Q = defaultdict(lambda: defaultdict(lambda: np.zeros(env.action_space.n)))
    if Q_init:
        EQ_copy(Q_init, Q)
    behaviour_policy =  epsilon_greedy_generalised_policy_improvement(env, Q)
        
    stop_cond = lambda k: k > maxiter
    
    stats = {"E":[], "R":[], "T":0}
    k=0
    T=0
    state = env.reset()
    stats["R"].append(0)

    goals = set([0])
    goal = list(goals)[np.random.randint(len(list(goals)))]
    while not stop_cond(k):
        probs = behaviour_policy(state, goal=goal, epsilon = epsilon)
        action = np.random.choice(np.arange(len(probs)), p=probs)            
        state_, reward, done, _ = env.step(action)  
        
        stats["R"][k] += (gamma**T)*reward
                 
        for goal_ in goals:
            if state_ != goal_ and done:  
                reward_ = N
            else:
                reward_ = reward
            
            G = 0 if done else np.max(Q[state_][goal_])
            TD_target = reward_ + gamma*G
            TD_error = TD_target - Q[state][goal_][action]
            Q[state][goal_][action] = Q[state][goal_][action] + alpha*TD_error

        T+=1
        if done:
            goals.add(state_)
            if k%100==0:
                print(k, len(Q), len(goals), np.mean(stats["R"][-100:]) )
                
            state = env.reset()
            goal = list(goals)[np.random.randint(len(list(goals)))]
            
            stats["R"].append(0)
            stats["T"] += T
            k+=1
            T=0
        else:
            state = state_

    return Q, stats

#########################################################################################

class EVF():
    def __init__(self, action_space):
        self.values = defaultdict(lambda: defaultdict(lambda: np.zeros(action_space.n)))
    
    def __call__(self, obs, goal):
        return self.values[obs][goal]

    def reset(self, obs):
        self.goals = list(self.values[obs].keys())
        print(self.goals, obs, len(self.values))
        self.goal = self._get_goal(obs)

    def get_value(self, obs):
        return self(obs, self.goal).max()

    def get_action(self, obs):
        return self(obs, self.goal).argmax()

    def _get_goal(self, obs):
        values = [self(obs, goal).max() for goal in self.goals]
        values = np.array(values)
        idx = np.random.choice(np.flatnonzero(values == values.max()))
        return self.goals[idx]

def EQ_save(EQ):
    EQ_ = {}
    for state in EQ:
        EQ_[state] = {}
        for goal in EQ[state]:
                EQ_[state][goal] = EQ[state][goal]
    return EQ_
def EQ_load(EQ):
    s = list(EQ.keys())[0]
    g = list(EQ[s].keys())[0]
    actions_n = len(EQ[s][g])
    EQ_ = defaultdict(lambda: defaultdict(lambda: np.zeros(actions_n)))
    for state in EQ:
        for goal in EQ[state]:
            EQ_[state][goal] = EQ[state][goal]
    return EQ_
def EQ_NP(EQ):
    P = defaultdict(lambda: defaultdict(lambda: 0))
    for state in EQ:
        for goal in EQ[state]:
                P[state][goal] = np.argmax(EQ[state][goal])
    return P
def EQ_P(EQ, goal=None):
    P = defaultdict(lambda: 0)
    for state in EQ:
        if goal:
            P[state] = np.argmax(EQ[state][goal])
        else:
            Vs = [EQ[state][goal] for goal in EQ[state].keys()]
            P[state] = np.argmax(np.max(Vs,axis=0))
    return P
def Q_P(Q):
    P = defaultdict(lambda: 0)
    for state in Q:
        P[state] = np.argmax(Q[state])
    return P

def EQ_NV(EQ):
    V = defaultdict(lambda: defaultdict(lambda: 0))
    for state in EQ:
        for goal in EQ[state]:
                V[state][goal] = np.max(EQ[state][goal])
    return V
def EQ_V(EQ, goal=None):
    V = defaultdict(lambda: 0)
    for state in EQ:
        if goal:
            V[state] = np.max(EQ[state][goal])
        else:
            Vs = [EQ[state][goal] for goal in EQ[state].keys()]
            V[state] = np.max(np.max(Vs,axis=0))
    return V
def NV_V(NV, goal=None):
    V = defaultdict(lambda: 0)
    for state in NV:
        if goal:
            V[state] = NV[state][goal]
        else:
            Vs = [NV[state][goal] for goal in NV[state].keys()]
            V[state] = np.max(Vs)
    return V
def Q_V(Q):
    V = defaultdict(lambda: 0)
    for state in Q:
        V[state] = np.max(Q[state])
    return V

def EQ_Q(EQ, goal=None):
    Q = defaultdict(lambda: np.zeros(5))
    for state in EQ:
        if goal:
            Q[state] = EQ[state][goal]
        else:
            Vs = [EQ[state][goal] for goal in EQ[state].keys()]
            if Vs:
                Q[state] = np.max(Vs,axis=0)
    return Q

#########################################################################################
### Redering Value funtions and policies

def render_learned(env, agent=True, env_map=False, fig=None, mode='human', P=None, V = None, Q = None, v_min = 0, title=None, grid=False, cmap='YlOrRd'):
    # cmap = 'RdYlBu_r'
    fig = env.env.render(agent=agent, env_map=env_map, fig=fig, title=title, grid=grid)
    
    if Q: # For showing action-values
        ax = fig.gca()
        cmap_ = cm.get_cmap(cmap)
        v_max = env.rmax
        v_min = v_min if v_min else (v_max + env.diameter*env.step_reward)
        norm = colors.Normalize(v_min,v_max)
        direction = env.actions.up
        for position in env.possiblePositions:
            gridworld_object = []
            for i in env.gridworld_objects_keys:
                gridworld_object.append(env.gridworld_objects[i].state(position))
                env.gridworld_objects[i].reset()
            state = (position,direction,frozenset(),tuple(gridworld_object))
            q = Q[state]
            y, x = position
            for action in range(env.action_space.n):
                v = (q[action]-v_min)/(v_max-v_min)
                draw_action_values(env, ax, x, y, action, v, cmap_)
        m = cm.ScalarMappable(norm=norm, cmap=cmap_)
        m.set_array(ax.get_images()[0])
        fig.colorbar(m, ax=ax)
    
    if V: # For showing values
        ax = fig.gca()
        v = np.zeros((env.m,env.n))+float("-inf")
        direction = env.actions.up
        for position in env.possiblePositions:
            gridworld_object = []
            for i in env.gridworld_objects_keys:
                gridworld_object.append(env.gridworld_objects[i].state(position))
                env.gridworld_objects[i].reset()
            state = (position,direction,frozenset(),tuple(gridworld_object))

            y, x = position
            v[y,x] = V[state]  
        c = plt.imshow(v, origin="upper", cmap=cmap, extent=[0, env.n, env.m, 0])
        # fig.colorbar(c, ax=ax)
            
    if P:  # For drawing arrows of optimal policy
        ax = fig.gca()
        direction = env.actions.up
        for position in env.possiblePositions:
            gridworld_object = []
            for i in env.gridworld_objects_keys:
                gridworld_object.append(env.gridworld_objects[i].state(position))
                env.gridworld_objects[i].reset()
            state = (position,direction,frozenset(),tuple(gridworld_object))

            y, x = position
            action = P[state]
            draw_action(env, ax, x, y, action)
            
    if mode == 'rgb_array':
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(int(width), int(height), 3)
        print(img.shape)

        return img
    return fig
    
def draw_action(self, ax, x, y, action, color='black'):
    if action == self.actions.up:
        x += 0.5
        y += 1
        dx = 0
        dy = -0.4
    if hasattr(self.actions, 'down') and action == self.actions.down:
        x += 0.5
        dx = 0
        dy = 0.4
    if action == self.actions.right:
        y += 0.5
        dx = 0.4
        dy = 0
    if action == self.actions.left:
        x += 1
        y += 0.5
        dx = -0.4
        dy = 0
    if action == self.actions.done:
        x += 0.5
        y += 0.5
        dx = 0
        dy = 0
        
        # ax.add_patch(patches.Circle((x, y), radius=0.25, fc=color, transform=ax.transData))        
        ax.add_patch(plt.Circle((x, y), radius=0.25, fc=color, transform=ax.transData))
        return

    ax.add_patch(ax.arrow(x,  # x1
                    y,  # y1
                    dx,  # x2 - x1
                    dy,  # y2 - y1
                    facecolor=color,
                    edgecolor=color,
                    width=0.005,
                    head_width=0.4,
                    )
                )

def draw_action_values(env, ax, x, y, action, reward, cmap):
    x += 0.5
    y += 0.5
    triangle = np.zeros((3,2))
    triangle[0] = [x,y]
    
    if action == env.actions.up:
        triangle[1] = [x-0.5,y-0.5]
        triangle[2] = [x+0.5,y-0.5]
    if action == env.actions.down:
        triangle[1] = [x-0.5,y+0.5]
        triangle[2] = [x+0.5,y+0.5]
    if action == env.actions.right:
        triangle[1] = [x+0.5,y-0.5]
        triangle[2] = [x+0.5,y+0.5]
    if action == env.actions.left:
        triangle[1] = [x-0.5,y-0.5]
        triangle[2] = [x-0.5,y+0.5]
    if action == env.actions.done:            
        ax.add_patch(plt.Circle((x, y), radius=0.25, color=cmap(reward)))
        return

    ax.add_patch(plt.Polygon(triangle, color=cmap(reward)))

#########################################################################################
def MAX(Q1, Q2):
    Q = defaultdict(lambda: 0)
    for s in list(set(list(Q1.keys())) & set(list(Q2.keys()))):
        Q[s] = np.max([Q1[s],Q2[s]], axis=0)
    return Q

def MIN(Q1, Q2):
    Q = defaultdict(lambda: 0)
    for s in list(set(list(Q1.keys())) & set(list(Q2.keys()))):
        Q[s] = np.min([Q1[s],Q2[s]], axis=0)
    return Q

def AVG(Q1, Q2):
    Q = defaultdict(lambda: 0)
    for s in list(set(list(Q1.keys())) & set(list(Q2.keys()))):
        Q[s] = (Q1[s]+Q2[s])/2
    return Q

#########################################################################################
def EQMAX(EQ,rmax=2, nA = 4): #Estimating EQ_max
    rmax = rmax
    EQ_max = defaultdict(lambda: defaultdict(lambda: np.zeros(nA)))
    for s in list(EQ.keys()):
        for g in list(EQ[s].keys()):
            c = rmax-max(EQ[g][g])
            if s==g:
                EQ_max[s][g] = EQ[s][g]*0 + rmax
            else:      
                EQ_max[s][g] = EQ[s][g] + c   
    return EQ_max

def EQMIN(EQ,rmin=-0.1, nA = 4): #Estimating EQ_min
    rmin = rmin
    EQ_min = defaultdict(lambda: defaultdict(lambda: np.zeros(nA)))
    for s in list(EQ.keys()):
        for g in list(EQ[s].keys()):
            c = rmin-max(EQ[g][g])
            if s==g:
                EQ_min[s][g] = EQ[s][g]*0 + rmin
            else:      
                EQ_min[s][g] = EQ[s][g] + c  
    return EQ_min

def NOTD(EQ, EQ_max, nA = 4):
    EQ_not = defaultdict(lambda: defaultdict(lambda: np.zeros(nA)))
    for s in list(EQ_max.keys()):
        for g in list(EQ_max[s].keys()):
            EQ_not[s][g] = EQ_max[s][g] - EQ[s][g]    
    return EQ_not

def NOT(EQ, EQ_max=None, EQ_min=None, nA = 4):
    EQ_max = EQ_max if EQ_max else EQMAX(EQ)
    EQ_min = EQ_min if EQ_min else EQMIN(EQ)
    EQ_not = defaultdict(lambda: defaultdict(lambda: np.zeros(nA)))
    for s in list(EQ_max.keys()):
        for g in list(EQ_max[s].keys()):
            EQ_not[s][g] = (EQ_max[s][g]+EQ_min[s][g]) - EQ[s][g]    
    return EQ_not

def OR(EQ1, EQ2, nA = 4):
    EQ = defaultdict(lambda: defaultdict(lambda: np.zeros(nA)))
    for s in list(set(list(EQ1.keys())) | set(list(EQ2.keys()))):
        for g in list(set(list(EQ1[s].keys())) | set(list(EQ2[s].keys()))):
            EQ[s][g] = np.max([EQ1[s][g],EQ2[s][g]],axis=0)
    return EQ

def AND(EQ1, EQ2, nA = 4):
    EQ = defaultdict(lambda: defaultdict(lambda: np.zeros(nA)))
    for s in list(set(list(EQ1.keys())) | set(list(EQ2.keys()))):
        for g in list(set(list(EQ1[s].keys())) | set(list(EQ2[s].keys()))):
            EQ[s][g] = np.min([EQ1[s][g],EQ2[s][g]],axis=0)
    return EQ
#########################################################################################

class SM_base():
    name = ''
    terminal_states = []
    skills = {}
    transitions = {}
    state = None
    skill = None

    def reset(self):
        self.state = 0
        self.skill = 'MAX'
        Q = self.skills['MAX']
        return Q

    def step(self, env_state, goal):
        for primitive, (next_state, skill) in self.transitions[self.state].items():
            if self._verify(primitive, env_state, goal):
                self.state = next_state
                self.skill = skill
                Q = self.skills[skill]
                break
        return Q
    
    def _verify(self, primitive, env_state, goal):
        Q = self.skills[primitive]
        return abs(Q[env_state][goal].max() - self.skills['MAX'][env_state][goal].max()) < abs(Q[env_state][goal].max() - self.skills['MIN'][env_state][goal].max())
 
class SM_Wrapper(SM_base):
    def __init__(self, SM):
        self.SM = SM        

    @property
    def name(self):
        return self.SM.name
    
    @property
    def state(self):
        return self.SM.state
    
    @property
    def skills(self):
        return self.SM.skills

    @property
    def states(self):
        return self.SM.states
        
    @property
    def terminal_states(self):
        return self.SM.terminal_states
        
    @property
    def max_(self):
        return self.SM.max_
        
    @property
    def min_(self):
        return self.SM.min_
        
    @property
    def skills(self):
        return self.SM.skills
        
    @property
    def values(self):
        return self.SM.values
        
    @property
    def transitions(self):
        return self.SM.transitions
        
    def reset(self):
        return self.SM.reset()
        
    def step(self, env_state, goal):
        return self.SM.step(env_state, goal)

class SM_THEN(SM_Wrapper):
    def __init__(self, SMs = []):
        self.SMs = SMs
        self.SMi = 0
        super().__init__(self.SMs[self.SMi])
        
    def reset(self):
        self.SMi = 0
        super().__init__(self.SMs[self.SMi])
        return self.SM.reset()
    
    def step(self, env_state, goal):
        Q = self.SM.step(env_state, goal)
        if self.SMi<len(self.SMs)-1 and self.SM.state in self.SM.terminal_states:
            self.SMi +=1
            super().__init__(self.SMs[self.SMi])
            Q = self.SM.reset()
        return Q

class SM_UNTIL(SM_Wrapper):
    def __init__(self, SM1, SM2):
        self.SM1, self.SM2 = SM1, SM2
        self.SM2.reset()
        super().__init__(self.SM1)
    
    def step(self, env_state, goal):
        Q = self.SM.step(env_state, goal)
        if self.SM.state in self.SM.terminal_states:
            Q = self.SM.reset()
        
        Q2 = self.SM2.step(env_state, goal)
        if EQ_Q(Q)[env_state].argmax() == 4:
            self.SM.state = self.SM.terminal_states[0]
        # if self.SM2.state in self.SM2.terminal_states:
        #     Q = self.SM2.reset()
        return Q

class SM_NOT(SM_Wrapper):
    def __init__(self, SM):
        super().__init__(SM)
    
    def step(self, env_state, goal):
        Q = self.SM.step(env_state, goal)
        Q = NOT(Q,EQ_max=self.max_, EQ_min=self.min_)
        
        return Q

#########################################################################################
