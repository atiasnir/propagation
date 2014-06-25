import numpy as np
import pandas as pd 
import scipy.sparse as sps
from network import Network

def _build_index(dataset, *columns):
    mapset = pd.DataFrame(columns=('name', 'index'), dtype=(str,int))
    for name_col, idx_col in columns:
        subset = dataset[[name_col, idx_col]]
        subset.columns = 'name', 'index'
        mapset = mapset.append(subset, ignore_index=True)
    
    mapset.drop_duplicates(inplace=True)
    #mapset['name'] = mapset.name.astype(str)
    return mapset

def read_ppi_from_dataframe(dataset, from_column='from', to_column='to', confidence_column='confidence'):
    all_genes = pd.concat((dataset[from_column], dataset[to_column])).unique()
    n = len(all_genes)

    names = pd.DataFrame(data={'name': all_genes, 'index': range(n)})
    
    d1 = dataset.merge(names, left_on=from_column, right_on='name')
    d1.rename(columns={'index': 'from_index'}, inplace=True)
    d1.drop('name', axis=1, inplace=True)
    d2 = d1.merge(names, left_on=to_column, right_on='name')
    d2.rename(columns={'index': 'to_index'}, inplace=True)
    d2.drop('name', axis=1, inplace=True)

    d2[confidence_column].fillna(1, inplace=True)
    d = sps.coo_matrix((d2[confidence_column], (d2['from_index'], d2['to_index'])), shape=(n,n))
    sym = (d + d.T)
    
    names = _build_index(d2, (from_column, 'from_index'), (to_column, 'to_index'))
    names.sort(columns=('index'), inplace=True)

    # for some strange reason the following:
    # > names_series = pd.Series(index=names['name'].values, data=names['index'].values)
    # will convert the index to int, while the current version won't.
    # it does not reproduce in a 'simple' case.
    names_series = pd.Series(index=names.name.values, data=names['index'].values)
    return Network(sym, names_series)

def read_ppi(filename):
    """ reads a ppi network from file  """
        
    column_names = ('from', 'to', 'confidence', 'flag')
    dataset = pd.read_table(filename, header=None, names=column_names)

    return read_ppi_from_dataframe(dataset)

def create_prior(index, names, scores):
    prior = np.zeros(shape=(index.shape[0], 1))
    mapped_names = index[names]
    na_mask = (~np.isnan(mapped_names)).values
    if hasattr(scores, '__getitem__'):
        prior[mapped_names[na_mask].values.astype(np.int)] = scores[na_mask].reshape((np.count_nonzero(na_mask),1))
    else:
        prior[mapped_names[na_mask].values.astype(np.int)] = scores

    return prior

def read_prior(filename, index):
    prior = np.zeros((len(index),1))
    with open(filename, 'r') as f:
        for line in f:
            name, score = line.strip().split()
            prior[index[name]] = float(score)

    return prior


