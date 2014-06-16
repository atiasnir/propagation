import argparse
import os.path

import numpy as np

import data

def parse_commandline(args):
    parser = argparse.ArgumentParser(description="Compute a smoothed score over network nodes")

    # positional
    parser.add_argument('network', type=str, help="network filename (tsv format: from/to/weight)")
    parser.add_argument('prior', type=str, default=None, help="prior filename (tsv format: from/value)")
    # named
    parser.add_argument('-a,--alpha', dest="alpha", type=float, default=0.6, help="relative weight for the network")
    parser.add_argument('-i,--iterations', dest="iterations", type=int, default=1000, help="maximum number of iterations to execute")
    parser.add_argument('-e,--epsilon', dest="epsilon", type=float, default=1e-5, help="convergence threshold")
    parser.add_argument('-v,--verbose', dest="verbose", action='store_true', default=False, help="verbose output")

    return parser.parse_args(args)

if __name__ == '__main__':
    import sys
    
    opts = parse_commandline(sys.argv[1:])
    if opts.verbose:
        sys.stderr.write(str(opts) + "\n")
    
    failed = False
    if not os.path.exists(opts.network):
        print "Invalid network filename"
        failed = True

    if not os.path.exists(opts.prior):
        print "Invalid prior filename"
        failed = True

    if opts.epsilon > 0.5:
        print "Epsilon is too large"
        failed = True
    
    if not 0 < opts.alpha < 1:
        print "alpha should be in (0,1)"
        failed = True

    if not failed:
        network = data.read_ppi(opts.network)
        if 'prior' in opts:
            prior = data.read_prior(opts.prior, network.names)
            if np.any(prior > 1.0):
                sys.stderr.write("Scaling prior, values outside [0,1]\n")
                prior /= np.max(prior)
        else:
            prior = np.ones((network.graph.shape[0],1))
          
        result = network.normalize().smooth(prior,
                                            alpha=opts.alpha,
                                            eps=opts.epsilon,
                                            max_iter=opts.iterations)

        for item in sorted(zip(network.names.index, result.flat), key=lambda x: x[1]):
            print "\t".join(map(str, item))
