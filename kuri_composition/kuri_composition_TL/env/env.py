import gym
from GridWorld import GridWorld, Predicate


# MAP1 =  "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n" \
#         "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
#         "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
#         "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1\n" \
#         "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
#         "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
#         "1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1\n" \
#         "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
#         "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
#         "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
#         "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
#         "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
#         "1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1\n" \
#         "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
#         "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
#         "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1\n" \
#         "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
#         "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
#         "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"

# class decor(Predicate):
#     def __init__(self, count=float("inf")):
#         positions = [(9,3),(9,21),(3,9),(3,15),(15,9),(15,15),  (3,3), (3,21), (15,21), (15,3)]
#         super().__init__(positions, count)

# class office(Predicate):
#     def __init__(self, count=float("inf")):
#         positions = [(9,9)]
#         super().__init__(positions, count)

# class mail(Predicate):
#     def __init__(self, count=0):
#         positions = [(9,15)]
#         super().__init__(positions, count)

# class coffee(Predicate):
#     def __init__(self, count=0):
#         positions = [(5,7),(13,17)]
#         super().__init__(positions, count)


MAP =   "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n" \
        "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n" \
        "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1\n" \
        "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n" \
        "1 1 0 1 1 1 0 1 1 1 0 1 1 1 0 1 1\n" \
        "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n" \
        "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n" \
        "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n" \
        "1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1\n" \
        "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n" \
        "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1\n" \
        "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n" \
        "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"

class decor(Predicate):
    def __init__(self, count=float("inf")):
        positions = [(6,2),(6,14),(2,6),(2,10),(10,6),(10,10),  (2,2), (2,14), (10,14), (10,2)]
        super().__init__(positions, count)

class office(Predicate):
    def __init__(self, count=float("inf")):
        positions = [(6,6)]
        super().__init__(positions, count)

class mail(Predicate):
    def __init__(self, count=0):
        positions = [(6,10)]
        super().__init__(positions, count)

class coffee(Predicate):
    def __init__(self, count=0):
        positions = [(3,5),(9,11)]
        super().__init__(positions, count)

predicates =  {
    'decor': decor(),
    'coffee': coffee(),
    'office': office(),
    'mail': mail(),
}  

def make_env(MAP=MAP,  predicates = predicates, goals=goals, goal_reward=goal_reward, step_reward=step_reward, slip_prob=slip_prob):
    if 'Custom' in env_key:
        env = gym.make(env_key, obj_type=obj_type, obj_color=obj_color, 
                        dist_type=dist_type, dist_color=dist_color, num_dists=num_dists)
    else:
        env = gym.make(env_key)
    env.seed(seed)
    env = FixEnv(env)
    if rgbimgobs:
        env = RGBImgPartialObsWrapper(env, tile_size=tile_size) if rgbimgobs=='partial' else FixRGBImgObsWrapper(env, tile_size=tile_size)
    else:        
        env = FixFullyObsWrapper(env)
    
    return env
