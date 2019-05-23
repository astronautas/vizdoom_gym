from vizdoomgym.envs.vizdoomenv import VizdoomEnv


class VizdoomDeathmatch(VizdoomEnv):

    def __init__(self, agent_id):
        super(VizdoomDeathmatch, self).__init__(8, agent_id)
