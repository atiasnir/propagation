import numpy as np 
from sklearn.metrics import roc_curve, auc

import command_line
import data

def parse_commandline(args):
    parser = command_line.create_parser()

    parser.add_argument('-s,--seed', dest="seed", type=int, default=None, help="a seed for the random number generator (default: use internal logic)")
    parser.add_argument('-f,--folds', dest="folds", type=int, default=5, help="the number of cross-validation folds")
    parser.add_argument('--validation', dest="validation", type=str, default=None, help="A validation dataset, that is use 'prior' +'network' to predict 'validation'")
    parser.add_argument('-l,--leave-one-out', dest="loo", action='store_true', default=False, help="Perform leave-one-out cross validation")

    return parser.parse_args(args)

def leave_one_out(network, prior, cv_func=None, **kwargs):
    prior_ind = np.nonzero(prior)[0]
    result = np.zeros(shape=(len(prior_ind),len(prior)))

    prior_cpy = np.copy(prior)

    for i in range(len(prior_ind)):
        prior_cpy[prior_ind[i]] = 0
        if i > 0:
            prior_cpy[prior_ind[i-1]] = prior[prior_ind[i-1]]

        if cv_func is None:
            result[i,:] = network.smooth(prior_cpy, **kwargs).flat
        else:
            result[i,:] = cv_func(network, prior_cpy, **kwargs).flat
        result[i,np.where((prior_cpy>0).flat)] = None

    return result

def kfold_m(network, priors, folds, seed=None, **kwargs):
    np.random.seed(seed)
    nzind, pind = np.nonzero(priors)
    
    nz_per_prior = np.bincount(pind)
    if np.any(folds > nz_per_prior):
        raise ValueError("Folds should be less than the number of nonzero elements in prior")

    #for i in range(priors.shape[1]):
    perm = np.random.permutation(nzind.shape[0])
    nzind = nzind[perm]
    pind = pind[perm]

    results = np.zeros(shape=(folds, priors.shape[0], priors.shape[1]))
    cv_chunks = nz_per_prior // folds

    for i in range(folds):
        prior_cpy = np.copy(priors)
        if i<folds-1:
            for p in range(priors.shape[1]):
                pmask = (pind == p)
                prior_cpy[nzind[pmask][i*cv_chunks[p]:(i+1)*cv_chunks[p]],p] = 0
        else:
            for p in range(priors.shape[1]):
                pmask = (pind == p)
                prior_cpy[nzind[pmask][i*cv_chunks[p]:],p] = 0

        results[i,:,:] = network.smooth(prior_cpy, **kwargs)
        results[i][np.where(prior_cpy>0)] = None

    return results

def kfold(network, prior, folds, seed=None, cv_func=None, **kwargs):
    if prior.shape[1] > 1:
        return kfold_m(network, prior, folds, seed, **kwargs)

    # shuffle the non zero indices so that each cv iteration 
    # is contiguous in prior_ind
    np.random.seed(seed)
    prior_ind = np.copy(np.nonzero(prior)[0])
    np.random.shuffle(prior_ind)
    
    if folds > len(prior_ind):
        raise ValueError("Folds should be less than the number of elements in the prior")
    
    result = np.zeros(shape=(folds, len(prior)))
    cv_chunk = len(prior_ind) // folds

    for i in range(folds):
        prior_cpy = np.copy(prior)
        if i < folds-1:
            prior_cpy[prior_ind[(i*cv_chunk):((i+1)*cv_chunk)]] = 0 # remove associations
        else:
            prior_cpy[prior_ind[(i*cv_chunk):]] = 0 # remove associations
        if cv_func is None:
            result[i,:] = network.smooth(prior_cpy, **kwargs).flat
        else:
            result[i,:] = cv_func(network, prior_cpy, **kwargs).flat
        result[i,np.where((prior_cpy>0).flat)] = None

    return result

def _extract_scores(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    return {'tpr': tpr, 'fpr': fpr, 'auc': auc(fpr,tpr)}

def roc(result, prior, aggregate=np.nanmean):
    scores = aggregate(result, axis=0)
    if scores.ndim > 1 and scores.shape[1] > 1:
        if prior.shape == scores.shape:
            return [_extract_scores(prior[:,i], scores[:,i]) for i in range(scores.shape[1])]
        
        return [_extract_scores(prior, scores[:,i]) for i in range(scores.shape[1])]

    return [_extract_scores(prior, scores)]


if __name__ == '__main__':
    import sys
    
    opts = parse_commandline(sys.argv[1:])
    succeeded = command_line.is_valid_args(opts)
    
    if opts.prior is None:
        print "Prior filename must be specfied for cross validation"
        succeeded = False

    if opts.folds <= 1:
        print "Folds should be at least 2"
        succeeded = False

    if opts.validation is not None and not os.path.exists(opts.validation):
        succeeded = False
        print "Invalid validation filename"
 
    if succeeded:
        network = data.read_ppi(opts.network)
        prior = data.read_prior(opts.prior, network.names)

        if np.any(np.abs(prior) > 1.0):
            sys.stderr.write("Scaling prior, values outside [-1,1]\n")
            prior /= np.max(np.abs(prior), axis=0)

        network = network.normalize()

        if opts.loo:
            result = leave_one_out(network, prior, alpha=opts.alpha, eps=opts.epsilon, max_iter=opts.iterations)
        else:
            result = kfold(network, prior, opts.folds, opts.seed, alpha=opts.alpha, eps=opts.epsilon, max_iter=opts.iterations)

        # handle the results array 
        roc_results = roc(result, prior != 0)

        for i,res in enumerate(roc_results):
            print opts.prior[i]
            print "tpr:", res['tpr']
            print "fpr:", res['fpr']
            print "auc:", res['auc']
            print
