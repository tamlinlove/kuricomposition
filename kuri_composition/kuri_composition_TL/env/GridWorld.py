from ast import literal_eval
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

import matplotlib.pyplot as plt
import matplotlib.cm     as cm
import matplotlib.colors as colors
from collections import defaultdict
from itertools import chain, combinations
from copy import deepcopy

from enum import IntEnum
from sympy.logic import boolalg
from sympy import sympify, Symbol

COLOURS = {0: [1, 1, 1], 1: [0.0, 0.0, 0.0], 3: [0.9,0.9,0.9], 10: [0, 0, 1], 20:[1, 1, 0.0], 21:[0.8, 0.8, 0.8]}


### Predicates base class

class GridWorld_Object():
    def __init__(self, positions=[], count=float('inf'), track=False):
        self.track = track
        self.positions = positions
        self._count = lambda: count if count else np.random.randint(3)
        
        self.achieved = False
        self.count = {position: self._count() for position in self.positions}

    def reset(self):
        self.count = {position: self._count() for position in self.positions}
        self.achieved = False
        return None
    
    def state(self, position):
        achieved = False
        if (position in self.count) and self.count[position]:
            self.count[position] -= 1
            achieved = True
        
        achieved = self.achieved or achieved
        if self.track:
            self.achieved = achieved
        
        state = []
        for position in self.positions:
            state.append(self.count[position]>0)
        
        return achieved, tuple(state)


### Office world objects

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

class roomA(GridWorld_Object):
    def __init__(self):
        positions = [(2,2)]
        super().__init__(positions)
        
class roomB(GridWorld_Object):
    def __init__(self):
        positions = [(2,7)]
        super().__init__(positions)
        
class roomC(GridWorld_Object):
    def __init__(self):
        positions = [(7,7)]
        super().__init__(positions)
        
class roomD(GridWorld_Object):
    def __init__(self):
        positions = [(7,2)]
        super().__init__(positions)

# class door1(GridWorld_Object):
#     def __init__(self):
#         positions = [(7,2)]
#         super().__init__(positions)

# class coffee(GridWorld_Object):
#     def __init__(self):
#         positions = [(3,5),(9,11)]
#         super().__init__(positions)

# class mail(GridWorld_Object):
#     def __init__(self, count=0):
#         positions = [(6,10)]
#         super().__init__(positions, count)

# class office(GridWorld_Object):
#     def __init__(self, count=0):
#         positions = [(6,6)]
#         super().__init__(positions, count)

# class decor(GridWorld_Object):
#     def __init__(self, track=True):
#         # positions = [(6,2),(6,14),(2,6),(2,10),(10,6),(10,10),(2,2),(2,14),(10,14),(10,2)] # If room predicates are not used
#         positions = [(6,2),(6,14),(2,6),(2,10),(10,6),(10,10)]
#         super().__init__(positions, track=track)

gridworld_objects =  {
    '1room': roomA(),
    '2room': roomB(),
    '3room': roomC(),
    '4room': roomD(),
    # '10door': door1(),
    # '20door': door2(),
    # '30door': door3(),
    # '40door': door4(),
    # 'decor': decor(),
    # 'coffee': coffee(),
    # 'mail': mail(),
    # 'office': office(),
}

# Defining actions and directions
class Directions(IntEnum):
    # Move up, move right, move down, move left , done
    up = 0
    right = 1 
    down = 2
    left = 3
class PolarActions(IntEnum):
    # Move up, rotate right, rotate left, done
    up = 0
    right = 1
    left = 2
    done = 3 

### GridWorld environment

class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, MAP=MAP, gridworld_objects=gridworld_objects, goal_reward=10, step_reward=-0.1, start_position=None, start_direction=None, has_doors=True, slip_prob=0):

        self.n = None
        self.m = None

        self.grid = None
        self.hallwayStates = None
        self.possiblePositions = []
        self.walls = []
        
        self.MAP = MAP
        self._map_init()
        self.diameter = (self.n+self.m)-4
        self.directions = Directions
        self.actions = PolarActions

        self.done = False
        
        self.slip_prob = slip_prob
        
        self.gridworld_objects = gridworld_objects
        self.gridworld_objects_keys =tuple(sorted(list(self.gridworld_objects.keys())))

        self.start_position = start_position
        self.start_direction = start_direction
        self.position = self.start_position if start_position else self.possiblePositions[0]
        self.direction = self.start_direction if start_direction else self.directions.up
        self.step_count = 0
        
        object_states = []
        for i in self.gridworld_objects_keys:
            object_states.append(self.gridworld_objects[i].state(self.position))
        self.state = self.position,tuple(object_states)

        # Rewards
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.rmax = goal_reward
        self.rmin = step_reward
        
        # Gym spaces for observation and action space
        self.observation_space = spaces.Discrete(len(self.possiblePositions))
        self.action_space = spaces.Discrete(len(self.actions))

        ##################### Goals
        self.GoalLocations = {}
        self.Doors = {}
        self.closed_doors = []
        
        self.has_doors = has_doors
        if self.has_doors:
            self.Doors[(2,4)] = "n"
            self.Doors[(2,5)] = "n"
            self.Doors[(7,4)] = "s"
            self.Doors[(7,5)] = "s"
            self.Doors[(4,2)] = "w"
            self.Doors[(5,2)] = "w"
            self.Doors[(4,7)] = "e"
            self.Doors[(5,7)] = "e"
        self.doors = set(list(self.Doors.values()))
        self.closed_doors = self.doors.copy()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.rand(seed)
        return [seed]
    
    def pertube_action(self,action): 
        a = 1-self.slip_prob
        b = self.slip_prob/(self.action_space.n-2)
        if action == self.actions['UP']:
            probs = [a,b,b,b]
        elif action == self.actions['RIGHT']:
            probs = [b,a,b,b]
        elif action == self.actions['DOWN']:
            probs = [b,b,a,b]
        elif action == self.actions['LEFT']:
            probs = [b,b,b,a]
        action = np.random.choice(np.arange(len(probs)), p=probs)       
        return action

    def step(self, action):
        assert self.action_space.contains(action)

        action = action #self.pertube_action(action)
        reward = self._get_reward(self.state, action)
        
        x, y = self.position     
        cell = str(self._get_grid_value(self.position))
        if action == self.actions.up:
            wall = (self.direction==self.directions.up and 'T' in cell) or (self.direction==self.directions.right and 'R' in cell) or (self.direction==self.directions.down and 'D' in cell) or (self.direction==self.directions.left and 'L' in cell)
            if not wall:
                dirs = [0,-1],[1,0],[0,1],[-1,0]
                x = x + dirs[self.direction][1]
                y = y + dirs[self.direction][0]
        elif action == self.actions.right:
            self.direction = (self.direction+1)%len(self.directions)
        elif action == self.actions.left:
            self.direction = (self.direction+len(self.directions)-1)%len(self.directions)

        if self.position in self.Doors and (x,y) in self.Doors and self.position!=(x,y) and self.Doors[self.position] in self.closed_doors:
            self.closed_doors.remove(self.Doors[self.position])
        
        self.position = (x, y)
                
        object_states = []
        for i in self.gridworld_objects_keys:
            object_states.append(self.gridworld_objects[i].state(self.position))
            
        if self._get_grid_value(self.position) == 1:  # new position in walls list
            # stay at old state if new coord is wall
            self.position = self.state[0]
        else:
            self.state = self.position, self.direction, frozenset(self.doors-self.closed_doors), tuple(object_states)
                             
        return self.state, reward, self.done, None

    def _get_reward(self, state, action):      
        return self.step_reward 

    def reset(self):
        self.done = False
        self.closed_doors = self.doors.copy()
        
        if not self.start_position:
            idx = np.random.randint(len(self.possiblePositions))
            self.position = self.possiblePositions[idx]  # self.start_state_coord
        else:
            self.position = self.start_position[np.random.randint(len(self.start_position))]
        
        self.direction = self.start_direction if self.start_direction != None else np.random.choice(self.directions)
        
        for p,f in self.gridworld_objects.items():
            f.reset()
        
        object_states = []
        for i in self.gridworld_objects_keys:
            object_states.append(self.gridworld_objects[i].state(self.position))
        self.state = (self.position,self.direction, frozenset(self.doors-self.closed_doors), tuple(object_states))
        return self.state

    def render(self, agent=True, env_map=False, goal=None, fig=None, mode='human', title=None, grid=False):        
        img = self._gridmap_to_img(goal=goal)        
        if not fig:
            fig = plt.figure(1, figsize=(12, 8), dpi=60, facecolor='w', edgecolor='k')
        
        params = {'font.size': 20}
        plt.rcParams.update(params)
        plt.clf()
        plt.xticks([])
        plt.yticks([])
        plt.grid(grid)
        if title:
            plt.title(title, fontsize=20)

        plt.imshow(img, origin="upper", extent=[0, self.n, self.m, 0])
        fig.canvas.draw()

        if env_map:
            ax = fig.gca()            
            for position in self.possiblePositions:
                y,x = position
                # Grid walls
                if (y,x) in self.Doors and self.Doors[(y,x)] in self.closed_doors:
                    if (y,x+1) in self.Doors:
                        self._draw_cell(ax, x, y, "R", color="#c2c2c2")
                    if (y,x-1) in self.Doors:
                        self._draw_cell(ax, x, y, "L", color="#c2c2c2")
                    if (y+1,x) in self.Doors:
                        self._draw_cell(ax, x, y, "D", color="#c2c2c2")
                    if (y-1,x) in self.Doors:
                        self._draw_cell(ax, x, y, "T", color="#c2c2c2")
                    continue
                cell = self.grid[y][x]
                if cell == 0 or cell == 1:
                    continue
                self._draw_cell(ax, x, y, cell)
                
                # Grid objects
                for gridworld_object, function in self.gridworld_objects.items():
                    if position in function.positions and function.count[position]>0:
                        p = gridworld_object[0].upper()
                        c = function.count[position]
                        c = '' if c == float('inf') else str(c)
                        label = "{}{}".format(c,p)
                        
                        ax.text(x+0.25, y+0.65, label, style='oblique', size=fig.get_figheight()*2)
                        break

            if agent:
                for (x,y) in self.GoalLocations.keys():
                    self._draw_action(ax, x, y, self.actions.done, color="#c2c2c2")
                y, x = self.position
                self._draw_agent(ax, x, y, self.direction)
        
        if mode == 'rgb_array':
            width, height = fig.get_size_inches() * fig.get_dpi()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(int(width), int(height), 3)
            return img
        
        return fig

    def _map_init(self):
        self.grid = []
        lines = self.MAP.split('\n')

        for i, row in enumerate(lines):
            row = row.split(' ')
            if self.n is not None and len(row) != self.n:
                raise ValueError(
                    "Map's rows are not of the same dimension...")
            self.n = len(row)
            rowArray = []
            for j, col in enumerate(row):
                if col in "01":
                    rowArray.append(int(col))
                else:
                    rowArray.append(col)
                if col == "1":
                    self.walls.append((i, j))
                # possible positions
                else:
                    self.possiblePositions.append((i, j))
            self.grid.append(rowArray)
        self.m = i + 1

        self._find_hallWays()

    def _find_hallWays(self):
        self.hallwayStates = []
        for x, y in self.possiblePositions:
            if ((self.grid[x - 1][y] == 1) and (self.grid[x + 1][y] == 1)) or \
                    ((self.grid[x][y - 1] == 1) and (self.grid[x][y + 1] == 1)):
                self.hallwayStates.append((x, y))

    def _get_grid_value(self, position):
        return self.grid[position[0]][position[1]]

    # specific for self.MAP
    def _getRoomNumber(self, state=None):
        if state == None:
            state = self.state
        # if state isn't at hall way point
        xCount = self._greaterThanCounter(state, 0)
        yCount = self._greaterThanCounter(state, 1)
        room = 0
        if yCount >= 2:
            if xCount >= 2:
                room = 2
            else:
                room = 1
        else:
            if xCount >= 2:
                room = 3
            else:
                room = 0

        return room

    def _greaterThanCounter(self, state, index):
        count = 0
        for h in self.hallwayStates:
            if state[index] > h[index]:
                count = count + 1
        return count

    def _draw_agent(self, ax, x, y, dir, color='black'):
        triangle = np.zeros((3,2))
        
        if dir == self.directions.up:
            triangle[0] = [x+0.5,y+0.2]
            triangle[1] = [x+0.25,y+0.8]
            triangle[2] = [x+0.75,y+0.8]
        if dir == self.directions.down:
            triangle[0] = [x+0.5,y+0.8]
            triangle[1] = [x+0.25,y+0.2]
            triangle[2] = [x+0.75,y+0.2]
        if dir == self.directions.right:
            triangle[0] = [x+0.8,y+0.5]
            triangle[1] = [x+0.2,y+0.25]
            triangle[2] = [x+0.2,y+0.75]
        if dir == self.directions.left:
            triangle[0] = [x+0.2,y+0.5]
            triangle[1] = [x+0.8,y+0.25]
            triangle[2] = [x+0.8,y+0.75]

        ax.add_patch(plt.Polygon(triangle, color=color))
        
    def _draw_cell(self, ax, x, y, cell, color='black'):
        pos = x, y
        for wall in cell:
            x, y = pos
            if wall == "T":
                dx = 1
                dy = 0
            if wall == "R":
                x += 1
                dx = 0
                dy = 1
            if wall == "D":
                y += 1
                dx = 1
                dy = 0
            if wall == "L":
                dx = 0
                dy = 1

            ax.add_patch(ax.arrow(x,  # x1
                        y,  # y1
                        dx,  # x2 - x1
                        dy,  # y2 - y1
                        facecolor=color,
                        edgecolor=color,
                        width=0.1,
                        head_width=0.0,
                        )
                        )

    def _draw_action(self, ax, x, y, action, color='black'):
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
            
            ax.add_patch(patches.Circle((x, y), radius=0.25, fc=color, transform=ax.transData))
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

    def _draw_reward(self, ax, x, y, action, reward, cmap):
        x += 0.5
        y += 0.5
        triangle = np.zeros((3,2))
        triangle[0] = [x,y]
        
        if action == self.actions.up:
            triangle[1] = [x-0.5,y-0.5]
            triangle[2] = [x+0.5,y-0.5]
        if hasattr(self.actions, 'down') and action == self.actions.down:
            triangle[1] = [x-0.5,y+0.5]
            triangle[2] = [x+0.5,y+0.5]
        if action == self.actions.right:
            triangle[1] = [x+0.5,y-0.5]
            triangle[2] = [x+0.5,y+0.5]
        if action == self.actions.left:
            triangle[1] = [x-0.5,y-0.5]
            triangle[2] = [x-0.5,y+0.5]
        if action == self.actions.done:            
            ax.add_patch(plt.Circle((x, y), radius=0.25, color=cmap(reward)))
            return

        ax.add_patch(plt.Polygon(triangle, color=cmap(reward)))


    def _gridmap_to_img(self, goal=None):
        row_size = len(self.grid)
        col_size = len(self.grid[0])

        obs_shape = [row_size, col_size, 3]

        img = np.zeros(obs_shape)

        gs0 = int(img.shape[0] / row_size)
        gs1 = int(img.shape[1] / col_size)
        for i in range(row_size):
            for j in range(col_size):
                for k in range(3):
                    if False and (i, j) == self.position:#start_position:
                        this_value = COLOURS[10][k]
                    else:
                        cell = self.grid[i][j]
                        if cell == 0 or cell == 1:
                            colour_number = int(cell)
                        else:
                            colour_number = 0
                        this_value = COLOURS[colour_number][k]
                    img[i * gs0:(i + 1) * gs0, j * gs1:(j + 1)
                                                       * gs1, k] = this_value
        return img


### Defining tasks over the environment

predicates =  {
    '1room': lambda state: state[3][0][0],
    '2room': lambda state: state[3][1][0],
    '3room': lambda state: state[3][2][0],
    '4room': lambda state: state[3][3][0],
    'sdoor': lambda state: "s" in state[2],
    'ndoor': lambda state: "n" in state[2],
    'edoor': lambda state: "e" in state[2],
    'wdoor': lambda state: "w" in state[2],
}

class Task(gym.core.Wrapper):
    def __init__(self, env, predicates=predicates, task_goals=[], rmax=10, rmin=-0.1, start_position=None, start_direction=None):
        super().__init__(env)
        
        self.start_position = start_position
        self.start_direction = start_direction
        self.task_goals = task_goals
        self.rmax = rmax
        self.rmin = rmin
                  
        # self.env.actions['DONE'] = len(self.env.actions)    
        # self.env.action_space = spaces.Discrete(env.action_space.n+1)

        self.predicates = predicates
        self.predicate_keys =tuple(sorted(list(self.predicates.keys())))
        self.goals = [self.goal_predicates(i) for i in range(2**len(self.predicate_keys))]
        self.goal_space = spaces.Discrete(len(self.goals))

        self.state = None
    
    def reset(self):
        self.state = self.env.reset()   
        return self.state
    
    def step(self, action):
        if action == self.actions.done:
            state = self.get_goal(self.state)
            reward = self._get_reward(state)
            done = True
            info = {}
        else:
            state, reward, done, info = self.env.step(action)
            self.state = state
        
        return state, reward, done, info
    
    def get_goal(self, state): # Labelling function
        goal = ''
        for predicate in self.predicate_keys:
            goal += str(0+self.predicates[predicate](state))

        return int(goal, 2)
    
    def predicates_goal(self, predicates):
        goal = ''
        for predicate in self.predicate_keys:
            goal += '1' if predicate in predicates else '0'

        return int(goal, 2)
        
    def goal_predicates(self, goal):
        goal = bin(goal)[2:]
        goal = '0'*(len(self.predicate_keys)-len(goal)) + goal
        predicates = set()
        for i in range(len(self.predicate_keys)-1,-1,-1):
            if goal[i] == '1':
                predicates.add(self.predicate_keys[i])

        return predicates
    
    def _get_reward(self, goal):
        return self.rmax if (goal in self.task_goals) else self.rmin   

    def render(self, **kwargs):
        return self.env.render(**kwargs) 
