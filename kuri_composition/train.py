from GridWorld import *
from library import *
import matplotlib.pyplot as plt

env = GridWorld()
### Learning base tasks

maxiter = 50000
gamma = 1
rmin = -0.1
rmax = 10
rmin_ = -100


env = GridWorld(goals = env.all_goals, goal_reward=rmax, step_reward=rmin)
A,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter, gamma=gamma, rmin_=rmin_)
save_EQ(A,"models/max")
fig=env.render( P=EQ_P(A), V = EQ_V(A))
fig.savefig("images/max.pdf", bbox_inches='tight')
# plt.show()
fig=render_evf(env,A)
fig.savefig("images/max_evf.pdf", bbox_inches='tight')
# plt.show()

env = GridWorld(goals = [], goal_reward=rmax, step_reward=rmin)
A,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter, gamma=gamma, rmin_=rmin_)
save_EQ(A,"models/min")
fig=env.render( P=EQ_P(A), V = EQ_V(A))
fig.savefig("images/min.pdf", bbox_inches='tight')
# plt.show()
fig=render_evf(env,A)
fig.savefig("images/min_evf.pdf", bbox_inches='tight')
# plt.show()

env = GridWorld(exp = "nw | ne", goal_reward=rmax, step_reward=rmin)
A,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter, gamma=gamma, rmin_=rmin_)
save_EQ(A,"models/top")
fig=env.render( P=EQ_P(A), V = EQ_V(A))
fig.savefig("images/top.pdf", bbox_inches='tight')
# plt.show()
fig=render_evf(env,A)
fig.savefig("images/top_evf.pdf", bbox_inches='tight')
# plt.show()

env = GridWorld(exp = "nw | sw", goal_reward=rmax, step_reward=rmin)
A,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter, gamma=gamma, rmin_=rmin_)
save_EQ(A,"models/left")
fig=env.render( P=EQ_P(A), V = EQ_V(A))
fig.savefig("images/left.pdf", bbox_inches='tight')
# plt.show()
fig=render_evf(env,A)
fig.savefig("images/left_evf.pdf", bbox_inches='tight')
# plt.show()

env = GridWorld(exp = "n", goal_reward=rmax, step_reward=rmin)
A,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter, gamma=gamma, rmin_=rmin_)
save_EQ(A,"models/n")
fig=env.render( P=EQ_P(A), V = EQ_V(A))
fig.savefig("images/n.pdf", bbox_inches='tight')
# plt.show()
fig=render_evf(env,A)
fig.savefig("images/n_evf.pdf", bbox_inches='tight')
# plt.show()

env = GridWorld(exp = "s", goal_reward=rmax, step_reward=rmin)
A,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter, gamma=gamma, rmin_=rmin_)
save_EQ(A,"models/s")
fig=env.render( P=EQ_P(A), V = EQ_V(A))
fig.savefig("images/s.pdf", bbox_inches='tight')
# plt.show()
fig=render_evf(env,A)
fig.savefig("images/s_evf.pdf", bbox_inches='tight')
# plt.show()

env = GridWorld(exp = "w", goal_reward=rmax, step_reward=rmin)
A,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter, gamma=gamma, rmin_=rmin_)
save_EQ(A,"models/w")
fig=env.render( P=EQ_P(A), V = EQ_V(A))
fig.savefig("images/w.pdf", bbox_inches='tight')
# plt.show()
fig=render_evf(env,A)
fig.savefig("images/w_evf.pdf", bbox_inches='tight')
# plt.show()

env = GridWorld(exp = "e", goal_reward=rmax, step_reward=rmin)
A,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter, gamma=gamma, rmin_=rmin_)
save_EQ(A,"models/e")
fig=env.render( P=EQ_P(A), V = EQ_V(A))
fig.savefig("images/e.pdf", bbox_inches='tight')
# plt.show()
fig=render_evf(env,A)
fig.savefig("images/e_evf.pdf", bbox_inches='tight')
# plt.show()