import numpy as np

import data
import command_line

def parse_commandline(args):
    parser = command_line.create_parser()

    return parser.parse_args(args)

if __name__ == '__main__':
    import sys
    
    opts = parse_commandline(sys.argv[1:])
    succeeded = command_line.is_valid_args(opts)
    
    if succeeded:
        network = data.read_ppi(opts.network)
        if opts.prior is None:
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
