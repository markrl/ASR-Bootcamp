import argparse

def get_params():
    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument('--run_name', type=str, default='test',
                        help='name of this run')
    parser.add_argument('--seed', type=int, default=18792,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='general debugging flag')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='base learning rate')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--gpus', type=int, default=1,
                        help='train on n gpus')
    parser.add_argument('--max_epochs', type=int, default=-1,
                        help='max number of epochs')
    parser.add_argument('--min_delta', type=float, default=1e-3,
                        help='minimum relative change for patience')
    parser.add_argument('--overfit_batches', type=float, default=0.0,
                        help='overfit batches')
    parser.add_argument('--monitor', type=str, default='val/loss',
                        help='metric to monitor for callbacks')
    parser.add_argument('--mode', type=str, default='min',
                        help='min or max')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience for callbacks')
    parser.add_argument('--train_wer', default=False, action='store_true',
                        help='turn on tracking for wer during training')
    parser.add_argument('--val_wer', default=False, action='store_true',
                        help='turn on tracking for wer during validation')
    parser.add_argument('--finetune_epoch', type=int, default=50,
                        help='epoch to start fine tuning end-to-end')
    parser.add_argument('--finetune_lr_mult', type=float, default=0.01,
                        help='factor to multiply learning rate by when starting fine tuning')                    
    
    # Data arguments
    parser.add_argument('--text_encoding', type=str, default='onehot',
                        help='the type of text encoder to use')
    parser.add_argument('--unit_type', type=str, default='word',
                        help='the type of text unit to use')
    parser.add_argument('--data_root', type=str, default='/data/cv-corpus-23.0-2025-09-05/en',
                        help='root directory of the dataset')
    parser.add_argument('--dictionary_dir', type=str, default=None,
                        help='directory containing the dictionary files')
    parser.add_argument('--min_occurrences', type=int, default=5,
                        help='the minimum number of occurrences required for a token to be in the dictionary')
    parser.add_argument('--vocab_size', type=int, default=0,
                        help='size of the vocabulary; default `0` automatically sets to the size of dictionary')
    parser.add_argument('--dataset_fs', type=float, default=16000,
                        help='sampling rate of the dataset')
    parser.add_argument('--keep_punctuation', default=False, action='store_true',
                        help='keep punctuation marks as tokens')
    parser.add_argument('--limit_train_batches', type=float, default=0.01,
                        help='percentage of the training data to use per epoch')
    parser.add_argument('--limit_val_batches', type=float, default=0.5,
                        help='percentage of the validation data to use per epoch')

    # Audio processing arguments
    parser.add_argument('--normalization_fn', type=str, default=None,
                        help='signal normalization function')
    parser.add_argument('--preemphasis_alpha', type=float, default=None,
                        help='preemphasis coefficient')
    parser.add_argument('--input_feature_type', type=str, default='waveform',
                        help='feature type to input to the network')
    parser.add_argument('--augment', default=False, action='store_true',
                        help='apply spectral augmentation')

    return parser.parse_args()