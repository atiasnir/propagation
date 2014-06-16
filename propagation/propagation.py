#! /usr/bin/env python

import numpy as np 
import numpy.linalg as la
import pandas as pd
import scipy.sparse as sps
import os.path

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

def _build_index(dataset, *columns):
    mapset = pd.DataFrame()
    for name_col, idx_col in columns:
        subset = dataset[[name_col, idx_col]]
        subset.columns = 'name', 'index'
        mapset = mapset.append(subset, ignore_index=True)
    
    mapset.drop_duplicates(inplace=True)
    return mapset

def read_ppi(filename):
    """ reads a ppi network from file  """
        
    column_names = ('from', 'to', 'confidence', 'flag')
    dataset = pd.read_table(filename, header=None, names=column_names)

    all_genes = pd.concat((dataset['from'], dataset['to'])).unique()
    n = len(all_genes)

    names = pd.DataFrame(data={'name': all_genes, 'index': range(n)})
    
    d1 = dataset.merge(names, left_on='from', right_on='name')
    d1.rename(columns={'index': 'from_index'}, inplace=True)
    d1.drop('name', axis=1, inplace=True)
    d2 = d1.merge(names, left_on='to', right_on='name')
    d2.rename(columns={'index': 'to_index'}, inplace=True)
    d2.drop('name', axis=1, inplace=True)

    d2.confidence.fillna(1, inplace=True)
    d = sps.coo_matrix((d2['confidence'], (d2['from_index'], d2['to_index'])), shape=(n,n))
    sym = (d + d.T)
    
    names = _build_index(d2, ('from', 'from_index'), ('to', 'to_index'))
    names.sort(columns=('index'), inplace=True)

    return Network(sym, pd.Series(index=names['name'].values, data=names['index'].values))

def read_prior(filename, index):
    prior = np.zeros((len(index),1))
    with open(filename, 'r') as f:
        for line in f:
            name, score = line.strip().split()
            prior[index[name]] = float(score)

    return prior


def parse_commandline(args):
    usage = """USAGE: propagation.py [options] network prior 
   -n, --network [filename]   network filename (tsv: from/to/weight)
   -p, --prior [filename]     prior (tsv: from/value)
   -i, --iterations [int]     maximum number of iterations (default: 1000)
   -e, --epsilon [double]     epsilon for convergence (default: 1e-5)
   -a, --alpha   [double]     relative weight for the network (default: 0.6)

   -h, --help                 print this help 
"""
    states = {'-n': 0, '--network': 0, 
              '-p': 1, '--prior': 1, 
              '-i': 2, '--iterations': 2, 
              '-e': 3, '--epsilon': 3, 
              '-a': 4, '--alpha': 4,
              '-v': -1, '--verbose': -1,
              '-h': -1, '--help': -1}

    opts = {'alpha': 0.6, 'epsilon': 1e-5, 'iterations': 100, 'verbose': False}
    state = 0;

    for i in args:
        if i.startswith('-'):
            if i == '-h' or i == '--help':
                print usage
                sys.exit(-1)
            if i == '-v' or i == '--verbose':
                opts['verbose'] = True
                continue
            if i not in states:
                print "Unknown option: ", i
                print usage
                sys.exit(-2)

            state = states[i]
            continue
        
        if state == 0:
            opts['network'] = i
            state = 1
            continue

        if state == 1:
            opts['prior'] = i
            if 'network' in opts:
                break
            state = 0
            continue

        if state == 2:
            opts['iterations'] = float(i)
        elif state == 3:
            opts['epsilon'] = float(i)
        elif state == 4:
            opts['alpha'] = float(i)

        if 'network' not in opts:
            state = 0 
        elif 'prior' not in opts:
            state = 1
        else:
            state = -1

    return opts

if __name__ == '__main__':
    import sys
    
    opts = parse_commandline(sys.argv[1:])
    if opts['verbose']:
        sys.stderr.write(str(opts) + "\n")
    
    fail = False
    if 'network' not in opts or not os.path.exists(opts['network']):
        print "Invalid network filename"
        fail = True

    if 'prior' in opts and not os.path.exists(opts['prior']):
        print "Invalid prior filename"
        fail = True

    if opts['epsilon'] > 0.5:
        print "Epsilon is too large"
        fail = True
    
    if not 0 < opts['alpha'] < 1:
        print "alpha should be in (0,1)"
        fail = True

    if not fail:
        network = read_ppi(opts['network'])
        if 'prior' in opts:
            prior = read_prior(opts['prior'], network.names)
            if np.any(prior > 1.0):
                sys.stderr.write("Scaling prior, values outside [0,1]\n")
                prior /= np.max(prior)
        else:
            prior = np.ones((network.graph.shape[0],1))
          
        result = network.normalize().smooth(prior,
                                            alpha=opts['alpha'],
                                            eps=opts['epsilon'],
                                            max_iter=opts['iterations'])

        for item in sorted(zip(network.names.index, result.flat), key=lambda x: x[1]):
            print "\t".join(map(str, item))
