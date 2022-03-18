from GridWorld import *
from library import *
import matplotlib.pyplot as plt

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

### Learning base tasks

maxiter=5000

goals=T_positions
env = GridWorld(MAP=MAP, goals=goals, T_positions=T_positions)
A,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter)
save_EQ(A,"models/max")
fig=env.render( P=EQ_P(A), V = EQ_V(A))
fig.savefig("images/max.pdf", bbox_inches='tight')
plt.show()
fig=render_evf(env,A)
fig.savefig("images/max_evf.pdf", bbox_inches='tight')
plt.show()

goals=[]
env = GridWorld(MAP=MAP, goals=goals, T_positions=T_positions)
A,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter)
save_EQ(A,"models/min")
fig=env.render( P=EQ_P(A), V = EQ_V(A))
fig.savefig("images/min.pdf", bbox_inches='tight')
plt.show()
fig=render_evf(env,A)
fig.savefig("images/min_evf.pdf", bbox_inches='tight')
plt.show()

goals=[(2,2),(2,7)]
env = GridWorld(MAP=MAP, goals=goals, T_positions=T_positions)
A,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter)
save_EQ(A,"models/top")
fig=env.render( P=EQ_P(A), V = EQ_V(A))
fig.savefig("images/top.pdf", bbox_inches='tight')
plt.show()
fig=render_evf(env,A)
fig.savefig("images/top_evf.pdf", bbox_inches='tight')
plt.show()

goals=[(2,2),(7,2)]
env = GridWorld(MAP=MAP, goals=goals, T_positions=T_positions)
A,stats1 = Goal_Oriented_Q_learning(env, maxiter=maxiter)
save_EQ(A,"models/left")
fig=env.render( P=EQ_P(A), V = EQ_V(A))
fig.savefig("images/left.pdf", bbox_inches='tight')
plt.show()
fig=render_evf(env,A)
fig.savefig("images/left_evf.pdf", bbox_inches='tight')
plt.show()
