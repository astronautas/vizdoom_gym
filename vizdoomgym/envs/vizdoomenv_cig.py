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
import time
from threading import Thread

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
           ['cig.cfg', 7]]  # 11 disabled deltas, should be 9

class VizdoomEnvCig(gym.Env):
    @property
    def imitation(self):
        return self._imitation
    
    @imitation.setter
    def imitation(self, value):
        self._imitation = value

    @property
    def action(self):
        return self._last_action

    @property
    def started(self):
        return self._started

    def __init__(self, level, port, agent_id=0, agents_total=1):
        self.step_tickrate = 0.0
        self.last_step_time = time.time()
        self.steps = 0
        self.start = time.time()

        self.game = DoomGame()
        self._imitation = False

        self.level = level
        scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')
        self.game.load_config(os.path.join(scenarios_dir, CONFIGS[self.level][0]))

        self.game.set_doom_map("map01")  # Limited deathmatch.
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.port = port

        self.cig_args = "-deathmatch +timelimit 60 +sv_forcerespawn 1 -netmode 0 +sv_spawnfarthest 1 +sv_nocrouch 1 +viz_respawn_delay 3 +viz_nocheat 1 +sv_noautoaim 1 "

        print(f"CREATING ENV with 127.0.0.1:{port}")

        self.agents_total = int(agents_total)
        self.agent_id = int(agent_id)
        self._started = False
        
        # In-game variables
        self.frag_count = 0
        self.health = 100
        self.last_step_timestep = int(round(time.time() * 1000))
        self.bot_count = 8 - self.agents_total

        # init game
        self.state = None

        self.action_space = spaces.Discrete(CONFIGS[level][1]) # disable deltas
        self.observation_space = spaces.Box(0, 255, (self.game.get_screen_height(),
                                                     self.game.get_screen_width(),
                                                     self.game.get_screen_channels()),
                                            dtype=np.uint8)

        self.viewer = None

    def _start_game(self):        
        # Setup either bot vs. player or multiagent game
        self.game.add_game_args(self.cig_args)

        if int(self.agents_total) > 1:
            if int(self.agent_id) == 0:
                self.game.add_game_args(f"-host {self.agents_total} -port {self.port} +name AI#{self.agent_id} +colorset {self.agent_id}")

                print("----")
                print("making master")
                print("----")
            else:
                print("----")
                print("making slave")
                print("----")
                
                self.game.add_game_args(f"-join 127.0.0.1:{self.port} +name Player{self.agent_id} +colorset {self.agent_id}")
        else:
            print("making single player")
            self.game.add_game_args(f"-host {self.agents_total} -port {self.port} +name Player{self.agent_id}")
        

        if int(self.agents_total) > 1:
            self.game.set_window_visible(False)

        if int(self.agents_total) > 1:
            print("ASYNC")
            self.game.set_mode(Mode.ASYNC_PLAYER)
        else:
            print("SYNC")
            self.game.set_mode(Mode.ASYNC_PLAYER)

        if self.imitation:
            self.game.set_mode(Mode.SPECTATOR)

        if self._imitation:
            self.game.set_ticrate(40)
        else:
            self.game.set_ticrate(20)

        self.game.init()

        # Add bots
        self.game.send_game_command("removebots")

        for i in range(self.bot_count):
            self.game.send_game_command("addbot")
        
        self._started = True

    def close(self):
        print(f"Closing: {self.port}")
        self.game.close()
        self._started = False

    def step(self, action):
        self.steps += 1

        if self.steps % 200 == 0:
            self.fps = int(200 / (time.time() - self.start))
            print("FPS: ", self.fps)
            self.start = time.time()

        # # print("Step seconds: ", time.time() - self.last_step_time)
        # self.step_tickrate = self.step_tickrate * 0.75 + (1.0 / (time.time() - self.last_step_time)) * 0.25
        # # print("Stepping tickrate: ", self.step_tickrate)

        if self.started != True:
            return None, None, None, None

        if action is None:
            act = []
        else:
            act = np.zeros(CONFIGS[self.level][1])
            act[action] = 1
            act = np.uint8(act)
            act = act.tolist()

        if self.imitation:
            self.game.advance_action()
        else:
            self.game.make_action(act)

        self._last_action = self.game.get_last_action()

        state = self.game.get_state()
        done = self.game.is_episode_finished()

        if not done:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
        else:
            observation = np.uint8(np.zeros(self.observation_space.shape))

        info = {'dummy': 0}
        
        reward = self._reward()

        if reward != 0:
            print(f"Agent {self.agent_id}, reward: {reward}")

        if self.game.is_episode_finished():
            print("FINISHED")

        return observation, reward, done, info

    def _reward(self):
        total_reward = 0

        # Frag reward (+1 if it kills another agent)
        total_reward += max(self.game.get_game_variable(GameVariable.FRAGCOUNT) - self.frag_count, 0) 
        self.frag_count = self.game.get_game_variable(GameVariable.FRAGCOUNT)

        # Death discount (as bots do not reduce frags)
        if self.game.get_game_variable(GameVariable.HEALTH) <= 0 and self.game.get_game_variable(GameVariable.HEALTH) < self.health:
            total_reward -= 1.0

        self.health = self.game.get_game_variable(GameVariable.HEALTH)

        return total_reward

    def reset(self):
        
        if self._started == False:
            self._start_game()

        # Reset only when the episode (10 mins) finishes to add exploration    
        if self.game.is_episode_finished():
            self.game.new_episode()
            self.frag_count = self.game.get_game_variable(GameVariable.FRAGCOUNT)

        self.state = self.game.get_state()
        img = self.state.screen_buffer

        return np.transpose(img, (1, 2, 0))

    def render(self, mode='human'):
        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
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
