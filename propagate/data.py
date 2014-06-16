import numpy as np
import pandas as pd 
import scipy.sparse as sps
from network import Network

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


