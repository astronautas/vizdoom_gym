from vizdoomgym.envs.vizdoomenv_cig import VizdoomEnvCig

class VizdoomCig(VizdoomEnvCig):

    def __init__(self, port=None, agent_id=None, agents_total=1):
        super(VizdoomCig, self).__init__(11, port, agent_id=agent_id, agents_total=agents_total)