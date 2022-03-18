import numpy as np
import hashlib
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt


#########################################################################################
def shortest(MAP):
    """
    Use Floyd-Warshall to compute shortest distances between all states
    """
    board = MAP.replace(" ","").split('\n')
    arr = np.array([list(row) for row in board])
    free_spaces = list(map(tuple, np.argwhere(arr != '1')))

    dist = {(x, y) : np.inf for x in free_spaces for y in free_spaces}

    for (u, v) in dist.keys():
        d = abs(u[0] - v[0]) + abs(u[1] - v[1])
        if d == 0:
            dist[(u, v)] = 0
        elif d == 1:
            dist[(u, v)] = 1

    for k in free_spaces:
        for i in free_spaces:
            for j in free_spaces:
                if dist[(i, j)] > dist[(i, k)] + dist[(k, j)]:
                    dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
    
    return dist, free_spaces

def get_V_optimal(env, dist, free_spaces, gamma = 1):
    V = defaultdict(lambda: 0)
    for state in free_spaces:
        values = []
        for goal in env.T_positions:
            d = dist[(state,goal)]
            C = d
            if gamma != 1:
                C = (1-(gamma**C))/(1-gamma)
            v = env._get_reward(goal, 0) + C * env._get_reward(state, 0)
            values.append(v)
        V[state] = np.max(values)
    return V

def get_EV_optimal(env, dist, free_spaces, rmin_ = -100, gamma = 1):
    EV = defaultdict(lambda: defaultdict(lambda: 0))
    for state in free_spaces:
        for goal in env.T_positions:
            d = dist[(state,goal)]
            C = d
            if gamma != 1:
                C = (1-(gamma**C))/(1-gamma)
            EV[state][goal] = env._get_reward(goal, 0) + C * env._get_reward(state, 0)
    return EV

def V_equal(V1,V2,epsilon=1e-2):    
    for state in V1:
        if abs(V1[state]-V2[state])>epsilon:
            return False
    return True

def EV_equal(EV1,EV2,epsilon=1e-2):    
    for state in EV1:
        for goal in EV1[state]:
            if abs(EV1[state][goal]-EV2[state][goal])>epsilon:
                return False
    return True

#########################################################################################
def to_hash(x, b=False):
    # hash_x = x['mission'] + ' | '+ hashlib.md5(x['image'].tostring()).hexdigest()
    hash_x = hashlib.md5(x.tostring()).hexdigest() if b else x
    return hash_x

def epsilon_greedy_policy_improvement(env, Q, epsilon = 1):
    """
    Implements policy improvement by acting epsilon-greedily on Q

    Arguments:
    env -- environment with which agent interacts
    Q -- Action function for current policy

    Returns:
    policy_improved -- Improved policy
    """

    def policy_improved(state, goal = None, epsilon = epsilon):
        probs = np.ones(env.action_space.n, dtype=float)*(epsilon/env.action_space.n)
        best_action = np.random.choice(np.flatnonzero(Q[state][goal] == Q[state][goal].max())) #np.argmax(Q[state][goal]) #
        probs[best_action] += 1.0 - epsilon
        return probs
    return policy_improved

def goal_policy(Q, state, goals, epsilon = 0):
    """
    Implements generalised policy improvement
    """
    goal = None
    if goals:
        values = [Q[state][goal].max() for goal in goals]
        values = np.array(values)
        best_goal = np.random.choice(np.flatnonzero(values == values.max()))
        if np.random.random()>epsilon:
            goal = goals[best_goal]
        else:
            goal = goals[np.random.randint(len(goals))]

    return goal

def Q_learning(env, V_optimal=None, gamma=1, epsilon=1, alpha=0.1, maxiter=100, mean_episodes=100, p=False):
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
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    behaviour_policy =  epsilon_greedy_policy_improvement(env, Q, epsilon = epsilon)
    
    stats = {"R":[], "T":0}
    stats["R"].append(0)
    k=0
    T=0
    t=0    
    state = env.reset()
    state = to_hash(state)

    stop_cond = lambda k: k < maxiter
    if V_optimal:
        stop_cond = lambda k: True if k%mean_episodes != 0 else not V_equal(V_optimal,Q_V(Q))

    while stop_cond(k):
        probs = behaviour_policy(state, epsilon = epsilon)
        action = np.random.choice(np.arange(len(probs)), p=probs)            
        state_, reward, done, _ = env.step(action)
        state_ = to_hash(state_)
        
        stats["R"][k] += (gamma**t)*reward
        
        G = 0 if done else np.max(Q[state_])
        TD_target = reward + gamma*G
        TD_error = TD_target - Q[state][action]
        Q[state][action] = Q[state][action] + alpha*TD_error
        
        state = state_
        t+=1
        if done:            
            state = env.reset()
            state = to_hash(state)

            stats["R"].append(0)
            stats["T"] += t
            t=0
            k+=1
            if p and k%mean_episodes == 0:
                mean_return = np.mean(stats["R"][-mean_episodes-1:-1])
                print('Episode: ', k, ' | Mean return: ', mean_return)
                    
    return Q, stats
    
def Goal_Oriented_Q_learning(env, EV_init=None, EV_optimal=None, T_positions=None, gamma=1, rmin_ = -100, epsilon=0.1, alpha=1, maxiter=100, mean_episodes=100, p=True):
    """
    Implements Goal Oriented Q_learning

    Arguments:
    env -- environment with which agent interacts
    T_positions -- Absorbing set
    gamma -- discount factor
    alpha -- learning rate
    maxiter -- maximum number of episodes

    Returns:
    Q -- New estimate of Q function
    """
    
    Q = defaultdict(lambda: defaultdict(lambda: np.zeros(env.action_space.n)))
    if EV_init!=None:
        for s in EV_init:
            for g,v in EV_init[s].items():
                Q[s][g] += v
    behaviour_policy =  epsilon_greedy_policy_improvement(env, Q)

    sMem={} # Goals memory
    if T_positions:
        for state in T_positions:
            state = to_hash(state)
            sMem[state]=0
    goals = list(sMem.keys())

    stats = {"R":[], "T":0}
    stats["R"].append(0)
    k=0
    T=0
    t=0    
    state = env.reset()
    state = to_hash(state)
    goal = goal_policy(Q, state, goals)

    stop_cond = lambda k: k < maxiter
    if EV_optimal:
        stop_cond = lambda k: True if k%mean_episodes != 0 else not EV_equal(EV_optimal,EQ_EV(Q))

    while stop_cond(k):
        if goal:
            probs = behaviour_policy(state, goal = goal, epsilon = epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)
        else:
            action = env.action_space.sample()
        state_, reward, done, _ = env.step(action)
        state_ = to_hash(state_)

        stats["R"][k] += (gamma**t)*reward
        if done:
            sMem[state] = 0
            goals = list(sMem.keys())

        for goal_ in goals:
            if state != goal_ and done:
                reward_ = rmin_
            else:
                reward_ = reward

            G = 0 if done else np.max(Q[state_][goal_])
            TD_target = reward_ + gamma*G
            TD_error = TD_target - Q[state][goal_][action]
            Q[state][goal_][action] = Q[state][goal_][action] + alpha*TD_error

        state = state_
        t+=1
        if done or t>100:
            state = env.reset()
            state = to_hash(state)
            goal = goal_policy(Q, state, goals, epsilon = 1)

            stats["R"].append(0)
            stats["T"] += t
            t=0
            k+=1
            if p and k%mean_episodes == 0:
                mean_return = np.mean(stats["R"][-mean_episodes-1:-1])                
                print('Episode: ', k, ' | Mean return: ', mean_return,
                      ' | States: ', len(list(Q.keys())), ' | Goals: ', len(goals))

    return Q, stats

#########################################################################################
def EQ_EP(EQ):
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

def EQ_EV(EQ):
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
def EV_V(NV, goal=None):
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

def EQ_Q(EQ, goal=None, actions = 5):
    Q = defaultdict(lambda: np.zeros(actions))
    for state in EQ:
        if goal:
            Q[state] = EQ[state][goal]
        else:
            Vs = [EQ[state][goal] for goal in EQ[state].keys()]
            Q[state] = np.max(Vs,axis=0)
    return Q


def EQ_R(EQ, rmin=-0.1, actions = 5):
    R = {}
    for state in EQ:
        R[state] = np.zeros(actions) + rmin
        if state in EQ[state]:
            R[state][actions-1] = EQ[state][state][actions-1] - rmin
    return R

def EQ_T(EQ_, state, action, goals=None, rmin=-0.1, gamma=1, amax=False):
    T = {}
    R = EQ_R(EQ_)

    states = list(EQ_.keys()) 
    goals = goals if goals else states
    EQ = np.array([EQ_[state][goal][action] for goal in goals])
    R = np.array([R[state][action] for goal in goals])
    EV = np.array([[np.max(EQ_[state][goal]) for goal in goals] for state in goals])

    EVi = np.linalg.pinv(EV)
    P = (1/gamma)*np.matmul((EQ-R),EVi)
    P = P==P.max() if amax else P
    T = {state:0 for state in states}
    T = {goals[i]:P[i] for i in range(len(goals))}
    return T

def EQ_Ta(EQ, state, goals=None, gamma=1, actions = 5):
    Ta = defaultdict(lambda: np.zeros(actions))
    for action in range(actions):
        probs = EQ_T(EQ, state, action, goals=goals, gamma=gamma, amax=True)
        for s,prob in probs.items():
            Ta[s][action] = prob
    return Ta

#########################################################################################

def save_EQ(EQ, path):
    data = [[s,[[g,EQ[s][g]] for g in EQ[s]]] for s in EQ]
    np.save(path,data, allow_pickle=True)

def load_EQ(path, actions = 5):
    data = np.load(path, allow_pickle=True)
    EQ = {s: defaultdict(lambda: np.zeros(actions), {g:v for (g,v) in gv}) for (s,gv) in data}
    EQ = defaultdict(lambda: defaultdict(lambda: np.zeros(actions)), EQ)
    return EQ

#########################################################################################
def EQMAX(EQ, rmax=2, actions = 5): #Estimating EQ_max
    EQ_max = defaultdict(lambda: defaultdict(lambda: np.zeros(actions)))
    for s in list(EQ.keys()):
        for g in list(EQ[s].keys()):
            c = rmax-max(EQ[g][g])
            if s==g:
                EQ_max[s][g] = EQ[s][g]*0 + rmax
            else:      
                EQ_max[s][g] = EQ[s][g] + c   
    return EQ_max

def EQMIN(EQ, rmin=-0.1, actions = 5): #Estimating EQ_min
    EQ_min = defaultdict(lambda: defaultdict(lambda: np.zeros(actions)))
    for s in list(EQ.keys()):
        for g in list(EQ[s].keys()):
            c = rmin-max(EQ[g][g])
            if s==g:
                EQ_min[s][g] = EQ[s][g]*0 + rmin
            else:      
                EQ_min[s][g] = EQ[s][g] + c  
    return EQ_min

def NOT(EQ, EQ_max=None, EQ_min=None, actions = 5):
    EQ_max = EQ_max if EQ_max else EQMAX(EQ)
    EQ_min = EQ_min if EQ_min else EQMIN(EQ)
    EQ_not = defaultdict(lambda: defaultdict(lambda: np.zeros(actions)))
    for s in list(EQ.keys()):
        for g in list(EQ[s].keys()):
            EQ_not[s][g] = (EQ_max[s][g]+EQ_min[s][g]) - EQ[s][g]    
    return EQ_not

def NEG(EQ, EQ_max=None, EQ_min=None, actions = 5):
    EQ_max = EQ_max if EQ_max else EQ
    EQ_min = EQ_min if EQ_min else defaultdict(lambda: defaultdict(lambda: np.zeros(actions)))
    EQ_not = defaultdict(lambda: defaultdict(lambda: np.zeros(actions)))
    for s in list(EQ_max.keys()):
        for g in list(EQ_max[s].keys()):
            EQ_not[s][g] = EQ_max[s][g] - EQ[s][g]
    return EQ_not

def OR(EQ1, EQ2, actions = 5):
    EQ = defaultdict(lambda: defaultdict(lambda: np.zeros(actions)))
    for s in list(set(list(EQ1.keys())) | set(list(EQ2.keys()))):
        for g in list(set(list(EQ1[s].keys())) | set(list(EQ2[s].keys()))):
            EQ[s][g] = np.max([EQ1[s][g],EQ2[s][g]],axis=0)
    return EQ

def AND(EQ1, EQ2, actions = 5):
    EQ = defaultdict(lambda: defaultdict(lambda: np.zeros(actions)))
    for s in list(set(list(EQ1.keys())) | set(list(EQ2.keys()))):
        for g in list(set(list(EQ1[s].keys())) | set(list(EQ2[s].keys()))):
            EQ[s][g] = np.min([EQ1[s][g],EQ2[s][g]],axis=0)
    return EQ

#########################################################################################

def render_evf(env, evf):
    def get_grid_evfs(env, evf):
        evf_ = np.ones([env.n**2,env.n**2])
        grid = np.zeros([env.n**2,env.n**2,4])
        
        for x in range(env.n):
            for y in range(env.m):
                if (x,y) not in env.walls:
                    img = np.zeros([env.n, env.m, 4])
                    for (i,j) in env.walls:
                        img[i,j,-1] = 1.0
                    grid[x*env.n:x*env.n+env.n,y*env.m:y*env.m+env.m] = img

                    img = np.zeros([env.n, env.m])+float("-inf")
                    for (i,j) in env.possiblePositions:
                        state = ((i,j),env.directions.up)
                        goal = ((x,y),env.directions.up)
                        img[i,j] = evf[state][goal].max()
                    evf_[x*env.n:x*env.n+env.n,y*env.m:y*env.m+env.m] = img
                else:
                    img = np.ones([env.n, env.m, 4])
                    grid[x*env.n:x*env.n+env.n,y*env.m:y*env.m+env.m] = img
        return grid, evf_

    #####################################################

    grid, evf = get_grid_evfs(env, evf)
    # evf = evf[env.n:-env.n,env.m:-env.m]
    # grid = grid[env.n:-env.n,env.m:-env.m]

    fig = plt.figure(1, figsize=(20, 20), dpi=60, facecolor='w', edgecolor='k')
    plt.clf()
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.grid(False)
    ax = fig.gca()

    cmap = 'RdBu_r'#'YlOrRd' if False else 'RdYlBu_r'
    ax.imshow(evf, origin="upper", cmap=cmap, extent=[0, env.n, env.m, 0])
    ax.imshow(grid, origin="upper", extent=[0, env.n, env.m, 0])
    return fig