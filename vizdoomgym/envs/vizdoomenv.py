import gym
from gym import spaces
from vizdoom import *
import numpy as np
import os
from gym.envs.classic_control import rendering
import math

import numpy as np
from gym.spaces import MultiDiscrete, Box
from scipy.interpolate import interp1d
import pandas as pd
import sys

CONFIGS = [['basic.cfg', 3],                # 0
           ['deadly_corridor.cfg', 7],      # 1
           ['defend_the_center.cfg', 3],    # 2
           ['defend_the_line.cfg', 3],      # 3
           ['health_gathering.cfg', 3],     # 4
           ['my_way_home.cfg', 5],          # 5
           ['predict_position.cfg', 3],     # 6
           ['take_cover.cfg', 2],           # 7
           ['deathmatch.cfg', 20],          # 8
           ['health_gathering_supreme.cfg', 3],  # 9,
           ['multi_duel.cfg', 3],  # 10
           ['cig.cfg', 9]]  # 11

class VizdoomEnv(gym.Env):
    
    # Each agent is spawned as a separate environment
    def setup_multiplayer(self, game, agent_id):
        print("Starting agent: ", agent_id)

        if agent_id == 0:
            game.add_game_args("-host 2 -deathmatch +timelimit 1 +sv_spawnfarthest 1")
            game.add_game_args(f"+name Player{agent_id} +colorset {agent_id}")
        else:
            game.add_game_args("-join 127.0.0.1")
            game.add_game_args(f"+name Player{agent_id} +colorset {agent_id}")

        return game

    def __init__(self, level, agent_id=None):
        print("------")
        print(agent_id)
        print("------")

        self.agent_id = agent_id
        self.started = False

        # init game
        self.level = level
        self.game = DoomGame()
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')

        if not(agent_id is None):
            print("Setup multiplayer")
            self.game = self.setup_multiplayer(self.game, agent_id)

        self.game.load_config(os.path.join(scenarios_dir, CONFIGS[level][0]))
        self.game.set_window_visible(False)

        if agent_id is None:
            print("Setup singleplayer")
            self.game.init()
            self.started = True

        self.state = None

        # self.action_space = MultiDiscreteWithBoundsInfoFloatized([CONFIGS[level][1]])
        self.action_space = spaces.Discrete(CONFIGS[level][1])

        self.observation_space = spaces.Box(0, 255, (self.game.get_screen_height(),
                                                     self.game.get_screen_width(),
                                                     self.game.get_screen_channels()),
                                            dtype=np.uint8)
        self.viewer = None

    def close(self):
        print("Closing")
        self.game.close()

    def step(self, action):
        if action is None:
            act = []
        else:
            act = np.zeros(CONFIGS[self.level][1])
            act[action] = 1
            act = np.uint8(act)
            act = act.tolist()

        reward = self.game.make_action(act)
        state = self.game.get_state()
        done = self.game.is_episode_finished()

        if not done:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
        else:
            observation = np.uint8(np.zeros(self.observation_space.shape))

        info = {'dummy': 0}
        
        if self.agent_id != None:
            print("---")
            print("Reward", reward)
            print("Done: ", done)
            print(self.game.get_game_variable(GameVariable.HEALTH))
            print(self.game.get_game_variable(GameVariable.FRAGCOUNT))
            print("Agent: ", self.agent_id)
            print("---")

        return observation, reward, done, info

    def reset(self):
        if self.started == False:
            self.game.init()

        self.game.new_episode()
        self.state = self.game.get_state()
        img = self.state.screen_buffer

        return np.transpose(img, (1, 2, 0))

    def render(self, mode='human'):
        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])

            # if self.viewer is None:
            #     self.viewer = rendering.SimpleImageViewer()
            # self.viewer.imshow(img)
        except AttributeError:
            pass

    @staticmethod
    def get_keys_to_action():
        # you can press only one key at a time!
        keys = {(): 2,
                (ord('a'),): 0,
                (ord('d'),): 1,
                (ord('w'),): 3,
                (ord('s'),): 4,
                (ord('q'),): 5,
                (ord('e'),): 6}
        return keys
