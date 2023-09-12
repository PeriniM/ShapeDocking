import numpy as np

class Vertex:
    def __init__(self, x, y, internal_angle, id):
        self.x = x
        self.y = y
        self.id = id
        self.internal_angle = internal_angle

    def getVertex(self):
        return np.array([self.x, self.y])
    