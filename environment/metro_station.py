from collections import defaultdict


class MetroStation:
    def __init__(self, parameters, env, id, name, dis2nextstation):
        self.parameters = parameters
        self.env = env
        self.id = id
        self.name = name
        self.dis2nextstation = dis2nextstation
        self.info = defaultdict(dict)

    def reset(self):
        self.info = defaultdict(dict)

