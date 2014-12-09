from gurobipy import *
import numpy as np 

def find_connected_ilp(network, scores, k=10, beta=None):
    if beta == None:
        beta = np.mean(scores)

    # we want to solve the following ILP program
    #
    # x_i = a 0/1 variable for each node. 
    #       1 iff node in the solution 
    #
    # y_ij = a 0/1 variable for each edge 
    #       1 iff edge is in the solution 
    #
    # the goal is to select a subset of k nodes 
    # such that the sum of nodes score
    # minus beta times the number of induce non-edges 
    # is maximized.
    #
    # Formally:
    #
    # max sum x * s^T - beta * (k * (k-1) / 2 - sum y_ij)
    #
    # s.t. 
    #      y_ij <= x_i, x_j
    #      y_ij >= x_i + x_j - 1
    #
    #      sum x_i == k
    #

    
    try:
        i, j = network.graph.nonzero()
        mask = i<j
        i = i[mask]
        j = j[mask]
        
        m = Model("component")

        # add variables
        x = [m.addVar(vtype=GRB.BINARY, name=n) for n in network.names.index]
        y = [[m.addVar(vtype=GRB.BINARY, name=("y_%d_%d" % (r, c))) for c in j] for r in i]

        m.update()

        # add constraints
        for u,v in zip(i,j):
            m.addConstr(y[u][v] <= x[u])
            m.addConstr(y[u][v] <= x[v])
            m.addConstr(y[u][v] >= x[u] + x[v] - 1)

        m.addConstr(quicksum(x) == k)

        m.setObjective(quicksum(x[i]*scores[i] for i in range(len(x))), GRB.MAXIMIZE)

        m.update()
        m.optimize()

        vals = m.getAttr("x", x)
        cluster =  [i for i in range(len(x)) if vals[i] > 0.5]

        return cluster
    
    except GurobiError:
        print "ERROR"


        


def find_connected(network, scores, beta = None):
    """ search for connected highscoring nodes """
    
    if beta == None:
        beta = np.mean(scores)
    
    # get the indices for all the edges
    i, j = network.graph.nonzero()
    mask = i<j
    i = i[mask]
    j = j[mask]
    
    # calculate the scores for all the edges (seeds)
    s = (scores[i] + scores[j]).flatten()
    
    o = np.argsort(s)[::-1]
    
    clusters = []
    assigned = np.zeros(scores.shape[0]) - 1
    
    # process seeds (from most likely to least likely)
    for idx in o:
        r, c = i[idx], j[idx]
        
        if np.all(assigned[[r,c]] >= 0):
            continue 

        nodes = [r,c]
        
        # construct neighbors list
        neighbors = np.unique(np.append(network.graph[r,:].nonzero()[1],network.graph[c,:].nonzero()[1]))
        neighbors = np.setdiff1d(neighbors, nodes, assume_unique=True)
        
        while True:
            in_edges = network.graph[neighbors,:][:,nodes]
            mask = (in_edges>0).toarray()
            neighbor_score = np.sum(mask*scores[neighbors] - ~mask * beta, axis=1)
            found = False

            while not found:
                neighbor_idx = np.argmax(neighbor_score)
                
                to_add = neighbors[neighbor_idx]
                if assigned[to_add]>=0:
                    neighbor_score[neighbor_idx] = -beta * neighbor_score.shape[0]
                else:
                    found = True

            if neighbor_score[neighbor_idx] <= 0:
                break
                

            idx = np.searchsorted(nodes, to_add)
            nodes.insert(idx, to_add)

            new_neighbors = network.graph[to_add,:].nonzero()[1]
            neighbors = np.union1d(neighbors, new_neighbors)
            neighbors = np.setdiff1d(neighbors, nodes, assume_unique=True)

        if len(nodes) == 2:
            continue
            
        assigned[nodes] = len(clusters)
        clusters.append(nodes)

    return clusters
