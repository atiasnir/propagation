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
    n = 1
    if isinstance(names, list) or isinstance(names, tuple):
        n = len(names)
        mapped_names = [index[names[x]] for x in range(n)]
    else:
        mapped_names = [index[names]]

    prior = np.zeros(shape=(index.shape[0], n))
    #mapped_names = [index[names[x]] for x in range(n)]
    #na_mask = (~np.isnan(mapped_names)).values
    #if hasattr(scores, '__getitem__'):
    #    prior[mapped_names[na_mask].values.astype(np.int)] = scores[na_mask].reshape((np.count_nonzero(na_mask),1))
    #else:

    for i in range(n):
        prior[mapped_names[i].values.astype(np.int),i] = scores

    return np.vstack(prior)

def read_prior(filenames, index):
    if not isinstance(filenames, list) and not isinstance(filenames, tuple):
        filenames = [filenames]

    prior = np.zeros((len(index),len(filenames)))
    for i, filename in enumerate(filenames):
        with open(filename, 'r') as f:
            for line in f:
                arr = line.strip().split()
                if len(arr)>1:
                    name, score = arr
                    prior[index[name],i] = float(score)
                else:
                    prior[index[arr[0]],i] = 1.0

    return prior
