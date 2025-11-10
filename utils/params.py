import argparse

def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--nworkers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='base learning rate')
    parser.add_argument('--gpus', type=int, default=0,
                        help='train on n gpus')
    parser.add_argument('--max_epochs', type=int, default=-1,
                        help='max number of epochs')
    parser.add_argument('--overfit_batches', type=float, default=0.0,
                        help='overfit batches')
    parser.add_argument('--monitor', type=str, default='val/loss',
                        help='metric to monitor for callbacks')
    parser.add_argument('--mode', type=str, default='min',
                        help='min or max')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience for callbacks')
    
    parser.add_argument('--text_encoding', type=str, default='onehot',
                        help='the type of text encoder to use')
    parser.add_argument('--unit_type', type=str, default='word',
                        help='the type of text unit to use')

    return parser.parse_args()