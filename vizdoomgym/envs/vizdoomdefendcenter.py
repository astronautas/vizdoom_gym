from vizdoomgym.envs.vizdoomenv import VizdoomEnv


class VizdoomDefendCenter(VizdoomEnv):

    def __init__(self, agent_id):
        super(VizdoomDefendCenter, self).__init__(2, agent_id=agent_id)
