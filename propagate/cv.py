import numpy as np 
from sklearn.metrics import roc_curve, auc

import command_line
import data

def parse_commandline(args):
    parser = command_line.create_parser()

    parser.add_argument('-s,--seed', dest="seed", type=int, default=None, help="a seed for the random number generator (default: use internal logic)")
    parser.add_argument('-f,--folds', dest="folds", type=int, default=5, help="the number of cross-validation folds")
    parser.add_argument('-l,--leave-one-out', dest="loo", action='store_true', default=False, help="Perform leave-one-out cross validation")

    return parser.parse_args(args)

def leave_one_out(network, prior, **kwargs):
    prior_ind = np.nonzero(prior)[0]
    result = np.zeros(shape=(len(prior_ind),len(prior)))

    prior_cpy = np.copy(prior)

    for i in range(len(prior_ind)):
        prior_cpy[prior_ind[i]] = 0
        if i > 0:
            prior_cpy[prior_ind[i-1]] = prior[prior_ind[i-1]]

        result[i,:] = network.smooth(prior_cpy, **kwargs).flat
        result[i,np.where((prior_cpy>0).flat)] = None

    return result

def kfold(network, prior, folds, seed=None, **kwargs):
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
        result[i,:] = network.smooth(prior_cpy, **kwargs).flat
        result[i,np.where((prior_cpy>0).flat)] = None

    return result

def roc(result, prior, aggregate=np.nanmean):
    scores = aggregate(result, axis=0)
    fpr, tpr, _ = roc_curve(prior>0, scores) 
    return tpr, fpr, auc(fpr, tpr)

if __name__ == '__main__':
    import sys
    
    opts = parse_commandline(sys.argv[1:])
    succeeded = command_line.is_valid_args(opts)
    
    if opts.prior is None:
        print "Prior filename must be specfied for cross validation"
        succeeded = False

    if opts.folds <= 1:
        print "Folds should be greater than 2"
        succeeded = False

    if succeeded:
        network = data.read_ppi(opts.network)
        prior = data.read_prior(opts.prior, network.names)

        if np.any(prior > 1.0):
            sys.stderr.write("Scaling prior, values outside [0,1]\n")
            prior /= np.max(prior)

        network = network.normalize()

        if opts.loo:
            result = leave_one_out(network, prior, alpha=opts.alpha, eps=opts.epsilon, max_iter=opts.iterations)
        else:
            result = kfold(network, prior, opts.folds, opts.seed, alpha=opts.alpha, eps=opts.epsilon, max_iter=opts.iterations)

        # handle the results array 
        tpr, fpr, auc_score = roc(result, prior)

        print tpr
        print fpr
        print auc_score
