import argparse

def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, default='test',
                        help='name of this training run')
    
    return parser.parse_args()