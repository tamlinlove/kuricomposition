from env.GridWorld import *
from library import *
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw



env = Task(GridWorld())
print(len(env.possiblePositions), len(env.goals), len(env.possiblePositions)*len(env.goals))
# fig = render_learned(env, agent=False,env_map=True)
# # fig.savefig("env.pdf", bbox_inches='tight')
# plt.show()

learner = GOAL
rmax=10
rmin=-0.1
gamma=1
slip_prob=0
maxiter=50000
epsilon=0.1
alpha=0.5
start_position = None


### Initialize models to continue training

max_, min_, C, O, M, D, R1, R2, R3, R4 = None, None, None, None, None, None, None, None, None, None

# max_, stats = np.load('models/EQ_max.npy', allow_pickle=True)
# max_ = EQ_load(max_)
# min_, stats = np.load('models/EQ_min.npy', allow_pickle=True)
# min_ = EQ_load(min_)

# C, stats = np.load('models/EQ_A.npy', allow_pickle=True)
# C = EQ_load(C)
# O, stats = np.load('models/EQ_C.npy', allow_pickle=True)
# O = EQ_load(O)
# M, stats = np.load('models/EQ_D.npy', allow_pickle=True)
# M = EQ_load(M)
# D, stats = np.load('models/EQ_B.npy', allow_pickle=True)
# D = EQ_load(D)
# R1, stats = np.load('models/EQ_1.npy', allow_pickle=True)
# R1 = EQ_load(R1)
# R2, stats = np.load('models/EQ_2.npy', allow_pickle=True)
# R2 = EQ_load(R2)
# R3, stats = np.load('models/EQ_3.npy', allow_pickle=True)
# R3 = EQ_load(R3)
# R4, stats = np.load('models/EQ_4.npy', allow_pickle=True)
# R4 = EQ_load(R4)

### Test training

# task_goals = []
# for goal in range(env.goal_space.n):
#     if ('t_coffee' in env.goals[goal]):
#         task_goals.append(goal)
# env = Task(GridWorld(slip_prob=slip_prob), start_position=start_position, task_goals=task_goals, rmax=rmax, rmin=rmin)
# C = CE#learner(env, Q_init = max_, gamma=gamma, epsilon=epsilon, alpha=alpha, maxiter=maxiter)
# # np.save("models/EQ_E",(EQ_save(C), stats))
# env.gridworld_objects['coffee']._count = lambda: float('inf')
# env.gridworld_objects['mail']._count = lambda: float('inf')
# env.gridworld_objects['office']._count = lambda: float('inf')
# env.reset()
# render_learned(env,  P=EQ_P(C), V = EQ_V(C))
# plt.show()
# g = env.predicates_goal(set(('t_coffee',)))
# render_learned(env,  P=EQ_P(C, goal=g), V = EQ_V(C, goal=g))
# plt.show()
# g = env.predicates_goal(set(('coffee','t_coffee',)))
# render_learned(env,  P=EQ_P(C, goal=g), V = EQ_V(C, goal=g))
# plt.show()
# NEG = lambda EQ: NOT(EQ,EQ_max=max_, EQ_min=min_)
# render_learned(env,  P=EQ_P(NEG(C)), V = EQ_V((NEG(C))))
# plt.show()
# g = env.predicates_goal(set(('t_coffee',)))
# render_learned(env,  P=EQ_P(NEG(C), goal=g), V = EQ_V(NEG(C), goal=g))
# plt.show()
# g = env.predicates_goal(set(('coffee',)))
# render_learned(env,  P=EQ_P(NEG(C), goal=g), V = EQ_V(NEG(C), goal=g))
# plt.show()
# env.gridworld_objects['coffee']._count = lambda: 0
# env.reset()
# render_learned(env,  P=EQ_P(NEG(C)), V = EQ_V((NEG(C))))
# plt.show()
# g = env.predicates_goal(set(('t_coffee',)))
# render_learned(env,  P=EQ_P(NEG(C), goal=g), V = EQ_V(NEG(C), goal=g))
# plt.show()
# g = env.predicates_goal(set(('coffee',)))
# render_learned(env,  P=EQ_P(NEG(C), goal=g), V = EQ_V(NEG(C), goal=g))
# plt.show()


### Training

task_goals = list(range(env.goal_space.n))
env = Task(GridWorld(slip_prob=slip_prob), start_position=start_position, task_goals=task_goals, rmax=rmax, rmin=rmin)
max_,stats = learner(env, Q_init = max_, gamma=gamma, epsilon=epsilon, alpha=alpha, maxiter=maxiter)
np.save("models/max",(EQ_save(max_), stats))
# render_learned(env,  P=EQ_P(max_), V = EQ_V(max_))
# plt.show()

task_goals = []
env = Task(GridWorld(slip_prob=slip_prob), start_position=start_position, task_goals=task_goals, rmax=rmax, rmin=rmin)
min_,stats = learner(env, Q_init = min_, gamma=gamma, epsilon=epsilon, alpha=alpha, maxiter=maxiter)
np.save("models/min",(EQ_save(min_), stats))
# render_learned(env,  P=EQ_P(min_), V = EQ_V(min_))
# plt.show()


task_goals = []
for goal in range(env.goal_space.n):
    if ('1room' in env.goals[goal]):
        task_goals.append(goal)
env = Task(GridWorld(slip_prob=slip_prob), start_position=start_position, task_goals=task_goals, rmax=rmax, rmin=rmin)
R1,stats = learner(env, Q_init = R1, gamma=gamma, epsilon=epsilon, alpha=alpha, maxiter=maxiter)
np.save("models/r1",(EQ_save(R1), stats))
# render_learned(env,  P=EQ_P(R1), V = EQ_V(R1))
# plt.show()

task_goals = []
for goal in range(env.goal_space.n):
    if ('2room' in env.goals[goal]):
        task_goals.append(goal)
env = Task(GridWorld(slip_prob=slip_prob), start_position=start_position, task_goals=task_goals, rmax=rmax, rmin=rmin)
R2,stats = learner(env, Q_init = R2, gamma=gamma, epsilon=epsilon, alpha=alpha, maxiter=maxiter)
np.save("models/r2",(EQ_save(R2), stats))
# render_learned(env,  P=EQ_P(R2), V = EQ_V(R2))
# plt.show()

task_goals = []
for goal in range(env.goal_space.n):
    if ('3room' in env.goals[goal]):
        task_goals.append(goal)
env = Task(GridWorld(slip_prob=slip_prob), start_position=start_position, task_goals=task_goals, rmax=rmax, rmin=rmin)
R3,stats = learner(env, Q_init = R3, gamma=gamma, epsilon=epsilon, alpha=alpha, maxiter=maxiter)
np.save("models/r3",(EQ_save(R3), stats))
# render_learned(env,  P=EQ_P(R3), V = EQ_V(R3))
# plt.show()

task_goals = []
for goal in range(env.goal_space.n):
    if ('4room' in env.goals[goal]):
        task_goals.append(goal)
env = Task(GridWorld(slip_prob=slip_prob), start_position=start_position, task_goals=task_goals, rmax=rmax, rmin=rmin)
R4,stats = learner(env, Q_init = R4, gamma=gamma, epsilon=epsilon, alpha=alpha, maxiter=maxiter)
np.save("models/r4",(EQ_save(R4), stats))
# render_learned(env,  P=EQ_P(R4), V = EQ_V(R4))
# plt.show()

task_goals = []
for goal in range(env.goal_space.n):
    if ('ndoor' in env.goals[goal]):
        task_goals.append(goal)
env = Task(GridWorld(slip_prob=slip_prob), start_position=start_position, task_goals=task_goals, rmax=rmax, rmin=rmin)
EQ,stats = learner(env, gamma=gamma, epsilon=epsilon, alpha=alpha, maxiter=maxiter)
np.save("models/dn",(EQ_save(EQ), stats))
# render_learned(env,  P=EQ_P(EQ), V = EQ_V(EQ))
# plt.show()

task_goals = []
for goal in range(env.goal_space.n):
    if ('edoor' in env.goals[goal]):
        task_goals.append(goal)
env = Task(GridWorld(slip_prob=slip_prob), start_position=start_position, task_goals=task_goals, rmax=rmax, rmin=rmin)
EQ,stats = learner(env, gamma=gamma, epsilon=epsilon, alpha=alpha, maxiter=maxiter)
np.save("models/de",(EQ_save(EQ), stats))
# render_learned(env,  P=EQ_P(EQ), V = EQ_V(EQ))
# plt.show()

task_goals = []
for goal in range(env.goal_space.n):
    if ('sdoor' in env.goals[goal]):
        task_goals.append(goal)
env = Task(GridWorld(slip_prob=slip_prob), start_position=start_position, task_goals=task_goals, rmax=rmax, rmin=rmin)
EQ,stats = learner(env, gamma=gamma, epsilon=epsilon, alpha=alpha, maxiter=maxiter)
np.save("models/ds",(EQ_save(EQ), stats))
# render_learned(env,  P=EQ_P(EQ), V = EQ_V(EQ))
# plt.show()

task_goals = []
for goal in range(env.goal_space.n):
    if ('wdoor' in env.goals[goal]):
        task_goals.append(goal)
env = Task(GridWorld(slip_prob=slip_prob), start_position=start_position, task_goals=task_goals, rmax=rmax, rmin=rmin)
EQ,stats = learner(env, gamma=gamma, epsilon=epsilon, alpha=alpha, maxiter=maxiter)
np.save("models/dw",(EQ_save(EQ), stats))
# render_learned(env,  P=EQ_P(EQ), V = EQ_V(EQ))
# plt.show()