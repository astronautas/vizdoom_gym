import gym
from gym import spaces
from vizdoom import *
import numpy as np
import os
#from gym.envs.classic_control import rendering
import math

import numpy as np
from gym.spaces import MultiDiscrete, Box
from scipy.interpolate import interp1d
import pandas as pd
import sys
import time
from threading import Thread
from multiprocessing import Process, Value

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
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.port = port

        self.cig_args = "-deathmatch +sv_forcerespawn 1 -netmode 0 +sv_spawnfarthest 1 +sv_nocrouch 1 +viz_respawn_delay 3 +viz_nocheat 1 +sv_noautoaim 1 "

        print(f"CREATING ENV with 127.0.0.1:{port}")

        self.agents_total = int(agents_total)
        self.agent_id = int(agent_id)
        self._started = False
        self._inited = False
        
        # In-game variables
        self.frag_count = 0
        self.health = 100
        self.last_step_timestep = int(round(time.time() * 1000))
        self.bot_count = 5

        # init game
        self.state = None

        self.action_space = spaces.Discrete(CONFIGS[level][1]) # disable deltas
        self.observation_space = spaces.Box(0, 255, (self.game.get_screen_height(),
                                                     self.game.get_screen_width(),
                                                     self.game.get_screen_channels()),
                                            dtype=np.uint8)
        
        self.last_obs = lambda: self.observation_space.sample()

        self.viewer = None

        self.game.set_doom_skill(5)

        # Thread(target=self.frozen_instance_detector).start()

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
        

        # if int(self.agents_total) > 1:
        self.game.set_window_visible(True)

        if int(self.agents_total) > 1:
            self.game.set_mode(Mode.PLAYER)
        else:
            print("SYNC")
            self.game.set_mode(Mode.PLAYER)

        if self.imitation:
            print("making a spectactor")
            self.game.set_mode(Mode.SPECTATOR)
            self.game.set_window_visible(True)
            self.game.set_screen_resolution(ScreenResolution.RES_640X480)

        self.game.init()
        

        # Add bots
        self.reset_bots(self.bot_count)
        
        self._started = True
        self._inited = True

    def reset_bots(self, count):
        if self.agent_id != 0:
            return

        print("Resetting bots")
        self.game.send_game_command("removebots")

        for i in range(count):
            print(f"Spawning bot: {i} out of {count}")
            self.game.send_game_command("addbot")

        print(self.game.send_game_command("listbots"))

    def frozen_instance_detector(self):
        game = self.game

        print("Starting frozen instance detector")
        last_game_time = 0
        last_time = time.time()
        pinged = 0

        while True:
            # print(f"Pinging game {pinged}")
            # print(f"Game time at agent: {self.agent_id} {last_game_time}")
            pinged += 1
            time.sleep(1)
            
            if self.started == False:
                print("Game not started yet")
                last_time = time.time()
            else:
                if game.get_episode_time() - last_game_time > 0:
                    print(f"Game seems alive at {last_game_time} tick")
                    last_game_time = game.get_episode_time()
                    last_time = time.time()
                else:
                    print("Frozen check")

                    if (time.time() - last_time) > 4.0:
                        print(f"Frozen agent: {self.agent_id}")
                        self.close()
                                    
    def close(self):
        if self._started:      
            print(f"Closing: {self.agent_id}")
            # print(f"Closing env for agent {self.agent_id}")
            self._started = False
            self.game.new_episode()
            print(f"Closed: {self.agent_id}")
            # # # self.game.close()

    def step(self, action):
        self.steps += 1

        if self._started != True:
            print("not started mate")
            return self.last_obs(), 0.0, False, {'dummy': 0}
        
        if not(self.imitation):
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
        done = False # deathmatch never finishes :)
        # done = self.game.is_episode_finished()

        if state:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
        else:
            observation = np.uint8(np.zeros(self.observation_space.shape))

        info = {'dummy': 0}
        
        reward = self._reward()

        if reward != 0:
            print(f"Agent received a reward! {self.agent_id}, reward: {reward}")

        return observation, reward, done, info

    def _reward(self):
        total_reward = 0

        # Frag reward (+-1 if it kills another agent, suicides itself CIG rules)
        total_reward = self.game.get_game_variable(GameVariable.FRAGCOUNT) - self.frag_count 
        self.frag_count = self.game.get_game_variable(GameVariable.FRAGCOUNT)

        # Death discount (as bots do not reduce frags)
        #if self.game.get_game_variable(GameVariable.HEALTH) <= 0 and self.game.get_game_variable(GameVariable.HEALTH) < self.health:
          #  total_reward -= 1.0

        #self.health = self.game.get_game_variable(GameVariable.HEALTH)

        return total_reward

    def reset(self):
        if not(self._inited):
            self._start_game()

        self._started = True
        self.frag_count = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        
        # Reset only when the episode (10 mins) finishes to add exploration    
        if self.game.is_episode_finished():
            self.game.new_episode()
            self.frag_count = self.game.get_game_variable(GameVariable.FRAGCOUNT)

        self.state = self.game.get_state()
        img = self.state.screen_buffer

        print(f"Resetting env for agent {self.agent_id} out of total {self.agents_total}")
        self.reset_bots(self.bot_count)
        
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