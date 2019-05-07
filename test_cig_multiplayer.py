# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym_vizdoom import (LIST_OF_ENVS, EXPLORATION_GOAL_FRAME, GOAL_REACHING_REWARD)
import vizdoomgym
from multiprocessing import Process
from threading import Thread
import time

def run_agent(a_id):
    print(f"making {a_id}")
    env = gym.make("VizdoomCig-v0", agent_id=a_id, agents_total=3, port=5039)

    for i in range(0, 20):
        policy = lambda env, obs: env.action_space.sample()
        done = False
        obs = env.reset()

        while True:
            action = policy(env, obs)

            if int(a_id) == 0:
                obs, reward, done, info = env.step(action)
            else:
                obs, reward, done, info = env.step(action)
            
            if reward != 0:
                print(reward)

            if done:
                return

            time.sleep(0.2)

agents = []

host = Process(target=run_agent, args=(str(0)))
host.start()

player2 = Process(target=run_agent, args=(str(1)))
player2.start()

player3 = Process(target=run_agent, args=(str(2)))
player3.start()

# run_agent(0)
input()