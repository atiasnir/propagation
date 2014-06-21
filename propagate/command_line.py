import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Compute a smoothed score over network nodes")

    # positional
    parser.add_argument('network', type=str, help="network filename (tsv format: from/to/weight)")
    parser.add_argument('prior', type=str, default=None, help="prior filename (tsv format: from/value)")
    # named
    parser.add_argument('-a,--alpha', dest="alpha", type=float, default=0.6, help="relative weight for the network")
    parser.add_argument('-i,--iterations', dest="iterations", type=int, default=1000, help="maximum number of iterations to execute")
    parser.add_argument('-e,--epsilon', dest="epsilon", type=float, default=1e-5, help="convergence threshold")
    parser.add_argument('-v,--verbose', dest="verbose", action='store_true', default=False, help="verbose output")

    return parser

def is_valid_args(opts):
    import sys
    import os.path
    
    if opts.verbose:
        sys.stderr.write(str(opts) + "\n")
    
    failed = False
    if opts.network is None or not os.path.exists(opts.network):
        print "Invalid network filename"
        failed = True

    if opts.prior is not None and not os.path.exists(opts.prior):
        print "Invalid prior filename"
        failed = True

    if opts.epsilon > 0.5:
        print "Epsilon is too large"
        failed = True
    
    if not 0 < opts.alpha < 1:
        print "alpha should be in (0,1)"
        failed = True

    return not failed


