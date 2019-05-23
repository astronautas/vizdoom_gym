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

env = gym.make("VizdoomCig-v0", port=5030, agent_id=0, agents_total=2)
env.imitation = True

for i in range(0, 20):
    policy = lambda env, obs: env.action_space.sample()
    done = False
    obs = env.reset()

    while True:
        action = policy(env, obs)
        obs, reward, done, info = env.step(action)
        
        if done:
            env.reset()