from ShapeDocking.Vertex import Vertex
from ShapeDocking.Edge import Edge

import numpy as np

class Polygon:
    def __init__(self, points, name):
        self.points = points
        self.name = name
        self.vertices = []
        self.edges = []

        self.computeVertices()
        self.computeEdges()
        self.computeNeighbors()

    def computeVertices(self):
        for i in range(len(self.points)):
            if i == 0:
                prev_point = self.points[-1]
            else:
                prev_point = self.points[i-1]
            if i == len(self.points)-1:
                next_point = self.points[0]
            else:
                next_point = self.points[i+1]
            
            self.vertices.append(Vertex(self.points[i], i))
            
        
        