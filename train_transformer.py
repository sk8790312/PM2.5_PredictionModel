"""Train one causal Transformer configuration; evaluate validation only."""
from sequence_training import main, run as run_sequence


def run(args):
    return run_sequence(args, 'transformer')


if __name__ == '__main__':
    main('transformer')
