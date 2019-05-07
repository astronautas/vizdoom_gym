from vizdoomgym.envs.vizdoomenv import VizdoomEnv
from multiprocessing import Process, Lock

class VizdoomMultiDuel(VizdoomEnv):
    def __init__(self, agent_id=None):
        super(VizdoomMultiDuel, self).__init__(10, agent_id=agent_id)
