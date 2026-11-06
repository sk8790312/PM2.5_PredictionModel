"""Train one LSTM configuration using the shared chronological protocol."""
from sequence_training import main, run as run_sequence


def run(args):
    return run_sequence(args, 'lstm')


if __name__ == '__main__':
    main('lstm')
