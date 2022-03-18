from GridWorld import *
from library import *
import matplotlib.pyplot as plt

MAP =   "1 1 1 1 1 1 1 1 1\n" \
        "1 0 0 0 1 0 0 0 1\n" \
        "1 0 0 0 0 0 0 0 1\n" \
        "1 0 0 0 1 0 0 0 1\n" \
        "1 1 0 1 1 1 0 1 1\n" \
        "1 0 0 0 1 0 0 0 1\n" \
        "1 0 0 0 0 0 0 0 1\n" \
        "1 0 0 0 1 0 0 0 1\n" \
        "1 1 1 1 1 1 1 1 1"

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

T_positions = [(2,2), (2,7), (7,7), (7,2)]
env = GridWorld(MAP=MAP, T_positions=T_positions)
# env.render(agent=True)
# plt.show()

### Loading learned tasks

MAX = load_EQ("models/max.npy")
MIN = load_EQ("models/min.npy")
A = load_EQ("models/top.npy")
B = load_EQ("models/left.npy")

### Zero-shot composition
NEG = lambda EQ: NOT(EQ, EQ_max=MAX, EQ_min=MIN)
XOR = lambda EQ1, EQ2: OR(AND(EQ1,NEG(EQ2)),AND(EQ2,NEG(EQ1)))

env.render( P=EQ_P(A), V = EQ_V(A))
plt.show()
env.render( P=EQ_P(B), V = EQ_V(B))
plt.show()

env.render( P=EQ_P(OR(A,B)), V = EQ_V(OR(A,B)))
plt.show()
env.render( P=EQ_P(AND(A,B)), V = EQ_V(AND(A,B)))
plt.show()
env.render( P=EQ_P(XOR(A,B)), V = EQ_V(XOR(A,B)))
plt.show()
env.render( P=EQ_P(NEG(OR(A,B))), V = EQ_V(NEG(OR(A,B))))
plt.show()