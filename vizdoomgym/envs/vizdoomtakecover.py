from vizdoomgym.envs.vizdoomenv import VizdoomEnv


class VizdoomTakeCover(VizdoomEnv):

    def __init__(self, agent_id):
        super(VizdoomTakeCover, self).__init__(level=7, agent_id=agent_id)