import numpy as np 
import numpy.linalg as la
import scipy.sparse as sps

class Network: 
    def __init__(self, graph, names):
        self.graph = graph
        self.names = names

    def normalize(self):
        data = 1.0 / np.sqrt(self.graph.sum(1))
        d = sps.dia_matrix( (data.T,[0]), (len(data),len(data)) )
        self.graph = d * self.graph * d

        return self

    def smooth(self, y, alpha=0.6, eps=1e-5, max_iter=1000):
        f = np.copy(y)
        for i in range(max_iter):
            fold, f = f, alpha * self.graph * f + (1-alpha) * y
            if la.norm(f-fold) < eps:
                break

        return f
