import numpy as np

class Edge:
    def __init__(self, vertex1, vertex2, id):
        self.vertices = [vertex1, vertex2]
        self.id = id
        self.length = None
        self.prev_angle = self.vertices[0].internal_angle
        self.next_angle = self.vertices[1].internal_angle
        self.normal_orientation = None
        self.neighbor_edges = []

    def getLength(self):
        if self.length is None:
            self.length = np.linalg.norm(self.vertices[1].getVertex() - self.vertices[0].getVertex())
        return self.length