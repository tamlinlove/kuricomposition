import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

import matplotlib.pyplot as plt
import matplotlib.cm     as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
from collections import defaultdict

from enum import IntEnum


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
class CardinalActions(IntEnum):
    # Move up, move right, move down, move left , done
    up = 0
    right = 1
    down = 2
    left = 3 
    done = 5

# Define colors
COLOURS = {0: [1, 1, 1], 1: [0.0, 0.0, 0.0], 3: [0, 0.5, 0], 10: [0, 0, 1], 20:[1, 1, 0.0], 21:[0.8, 0.8, 0.8]}

class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}
    MAP = "1 1 1 1 1 1 1 1 1 1 1 1 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 1 0 1 1 1 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 1 1 1 0 1 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
          "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
          "1 1 1 1 1 1 1 1 1 1 1 1 1"

    def __init__(self, MAP=MAP, dense_goal_rewards = False, dense_rewards = False, rmin_=-100, goal_reward=2, step_reward=-0.1, goals=None, T_positions=None, start_position=None):

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
        
        self.start_position = start_position
        self.position = self.start_position if start_position else (1, 1)
        self.direction = None
        self.state = None
        self.w = None #virtual state 
        self.step_count = 0
        
        if goals:
            self.goals = goals
        else:
            self.goals = (11, 11) 
            
        if T_positions:
            self.T_positions = T_positions
        else:
            self.T_positions = self.goals

        # Rewards
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.rmax = goal_reward
        self.rmin = step_reward
        
        self.dense_goal_rewards = dense_goal_rewards
        self.dense_rewards = dense_rewards
        if self.dense_rewards:
            self.rmin = self.rmin*10

        # Gym spaces for observation and action space
        self.observation_space = spaces.Discrete(len(self.possiblePositions))
        self.action_space = spaces.Discrete(len(self.actions))
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.rand(seed)
        return [seed]
    
    def pertube_action(self,action):        
        a = 0.8
        b = (1-a)/2
        if action == self.actions.up:
            probs = [a,b,0,b,0]
            action = np.random.choice(np.arange(len(probs)), p=probs)
        elif hasattr(self.actions, 'down') and action == self.actions.down:
            probs = [0,b,a,b,0]
            action = np.random.choice(np.arange(len(probs)), p=probs)
        elif action == self.actions.right:
            probs = [b,a,b,0,0]
            action = np.random.choice(np.arange(len(probs)), p=probs)
        elif action == self.actions.left:
            probs = [b,0,b,a,0]
            action = np.random.choice(np.arange(len(probs)), p=probs)       
        return action
    
    def step(self, action):
        assert self.action_space.contains(action)
        self.step_count += 1
        
        if action == self.actions.done and (self.state[0] in self.T_positions):
            return (self.w), self._get_reward(self.state, action), True, None
        else:
            action_ = action #self.pertube_action(action)
            x, y = self.position     
            cell = str(self._get_grid_value(self.position))
            if hasattr(self.actions, 'down'):
                if action_ == self.actions.up and 'T' not in cell:
                    x = x - 1
                elif action_ == self.actions.down and 'D' not in cell:
                    x = x + 1
                elif action_ == self.actions.right and 'R' not in cell:
                    y = y + 1
                elif action_ == self.actions.left and 'L' not in cell:
                    y = y - 1
            else:
                if action_ == self.actions.up:
                    wall = (self.direction==self.directions.up and 'T' in cell) or (self.direction==self.directions.right and 'R' in cell) or (self.direction==self.directions.down and 'D' in cell) or (self.direction==self.directions.left and 'L' in cell)
                    if not wall:
                        dirs = [0,-1],[1,0],[0,1],[-1,0]
                        x = x + dirs[self.direction][1]
                        y = y + dirs[self.direction][0]
                elif action_ == self.actions.right:
                    self.direction = (self.direction+1)%len(self.directions)
                elif action_ == self.actions.left:
                    self.direction = (self.direction+len(self.directions)-1)%len(self.directions)

            self.position = (x, y)
        
        reward = self._get_reward(self.state, action)
        
        if self._get_grid_value(self.position) == 1:  # new position in walls list
            # stay at old state if new coord is wall
            self.position = self.state[0]
        else:
            self.state = self.position, self.direction
        
        return self.state, reward, self.done, None
    
    def get_neightbours(self, position):
        directions = [(1,0),(1,1),(1,-1),(0,1),(-1,1),(-1,0),(0,-1)]
        positions = [position]
        for dx,dy in directions:
            pos = position[0]+dx, position[1]+dy
            if self._get_grid_value(pos) != 1:
                positions.append(pos)
        return positions

    def _get_dense_reward(self, position, action):
        g = np.array([g for g in self.goals])
        s = np.array([position]*len(g))
        reward = 0.1*np.mean(np.exp(-0.25*np.linalg.norm(s-g, axis=1)**2))
        return reward

    def _get_reward(self, state, action):      
        reward = 0
        position = state[0]
        if self.dense_rewards:
            reward += self._get_dense_reward(position,action)

        if position in self.T_positions and action == self.actions.done:
            if self.dense_goal_rewards:
                g = np.array([g for g in self.goals])
                s = np.array([position]*len(g))
                # steps = np.abs(s-g).max()
                steps = np.abs(s-g).sum(axis=1).min()
                rewards = np.linspace(self.goal_reward,self.step_reward,self.n-2)
                reward += rewards[steps]
            else:
                reward += self.goal_reward if position in self.goals else self.step_reward
        else:
            reward += self.step_reward
        
        return reward
        
    def get_rewards(self):
        R = defaultdict(lambda: np.zeros(self.action_space.n))
        for position in self.possiblePositions: 
            state = position, 0
            for action in range(self.action_space.n):
                R[state][action] = self._get_reward(state,action)
        return R

    def reset(self):
        self.step_count = 0
        self.done = False
        if not self.start_position:
            idx = np.random.randint(len(self.possiblePositions))
            self.position = self.possiblePositions[idx]  # self.start_state_coord
        else:
            self.position = self.start_position
        self.direction = self.directions.up # np.random.choice(self.directions)
        self.state = self.position, self.direction
        return self.state
        
    def render(self, fig=None, ax=None, goal=None, mode='human', agent=False,
                P=None, V = None, Q = None, R = None, T = None, Ta = None,
                Ta_true = None, title=None, grid=False, cmap=None, show_color_bar=False):

        if not cmap:
            cmap = 'YlOrRd' if R else 'RdYlBu_r'

        img = self._gridmap_to_img(goal=goal)        
        if not fig:
            fig = plt.figure(1, figsize=(20, 20), dpi=60, facecolor='w', edgecolor='k')
            params = {'font.size': 40}
            plt.rcParams.update(params)
            plt.clf()
            plt.xticks([])
            plt.yticks([])
            plt.grid(grid)
            if title:
                plt.title(title, fontsize=20)    
            if mode == 'human':
                fig.canvas.draw()
        if not ax:
            ax = fig.gca()
        
        vmax = 2 #1.8#
        vmin = -2 #0#
        ax.imshow(img, origin="upper", extent=[0, self.n, self.m, 0])
        
        for x in range(self.n): # For showing rewards
            for y in range(self.m):
                cell = self.grid[y][x]
                if cell == 0 or cell == 1:
                    continue
                self._draw_cell(ax, x, y, cell)
        if agent:
            for (x,y) in self.T_positions:
                self._draw_action(ax, x, y, self.actions.done, color="#d3d3d3")
            y, x = self.position
            self._draw_agent(ax, x, y, self.direction)
        
        if Q: # For showing action_value
            cmap_ = cm.get_cmap(cmap)
            norm = colors.Normalize(vmin,vmax)
            for state, q in Q.items():
                if state[1] == self.actions.up:
                    y, x = state[0]
                    for action in range(self.action_space.n):
                        v = (q[action]-vmin)/(vmax-vmin) # 
                        self._draw_reward(ax, x, y, action, v, cmap_)
            if show_color_bar:
                m = cm.ScalarMappable(norm=norm, cmap=cmap_)
                m.set_array(ax.get_images()[0])
                fig.colorbar(m, ax=ax)
                    
        if V: # For showing values
            v = np.zeros((self.m,self.n))+float("-inf")
            for state, val in V.items():
                if state[1] == self.actions.up:
                    y, x = state[0]
                    v[y,x] = val  
            c = ax.imshow(v, origin="upper", cmap=cmap, extent=[0, self.n, self.m, 0])
            if show_color_bar:
                fig.colorbar(c, ax=ax)
                
        if P:  # For drawing arrows of policy
            for state, action in P.items():
                if state[1] == self.actions.up:
                    y, x = state[0]
                    self._draw_action(ax, x, y, action)
        
        if R: # For showing rewards
            cmap_ = cm.get_cmap(cmap)
            norm = colors.Normalize(vmin=self.rmin, vmax=self.rmax)
            for state, reward in R.items():
                if state[1] == self.actions.up:
                    y, x = state[0]
                    for action in range(self.action_space.n):
                        r = (reward[self.actions.done]-self.rmin)/(self.rmax-self.rmin)
                        self._draw_reward(ax, x, y, action, r, cmap_)
            # if show_color_bar:
            #     m = cm.ScalarMappable(norm=norm, cmap=cmap_)
            #     m.set_array(ax.get_images()[0])
            #     fig.colorbar(m, ax=ax)
        
        if T:  # For showing transition probabilities of single action
            vprob = np.zeros((self.m,self.n))+float("-inf")
            for state, prob in T.items():
                if state[1] == self.actions.up:
                    y, x = state[0]
                    vprob[y,x] = prob  
            c = plt.imshow(vprob, origin="upper", cmap=cmap, extent=[0, self.n, self.m, 0])
            if show_color_bar:
                fig.colorbar(c, ax=ax)
            
        if Ta:  # For showing transition probabilities of all actions
            for state, probs in Ta.items():
                if state[1] == self.actions.up:
                    y, x = state[0]
                    for action in range(len(probs)):
                        if probs[action]:
                            if Ta_true and not Ta_true[state][action]:
                                self._draw_action(ax, x, y, action, color='red')
                            else:
                                self._draw_action(ax, x, y, action)

        if mode == "rgb_array":
            return self.fig2data(fig)

        return fig
        
    def fig2data ( self, fig ):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw ( )
    
        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()
        buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
        buf.shape = ( w, h,4 )
    
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll ( buf, 3, axis = 2 )
        return buf
        
    def fig_image(fig):
        fig.tight_layout(pad=0)
        fig.gca().margins(0)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image

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
                    elif goal and (i, j) == (goal)[1]:
                        this_value = COLOURS[20][k]
                    elif [(i, j),(i, j)] in self.goals:
                        this_value = COLOURS[3][k]
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