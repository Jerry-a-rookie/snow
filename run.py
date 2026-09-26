import argparse
import multiprocessing as mp
import random
import warnings

import numpy as np
import torch

warnings.filterwarnings(
    'ignore',
    message='Pandas requires version .*',
    category=UserWarning,
)

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

DEFAULT_SEED = 2021


def set_seed(seed):
    """Set all process-local RNGs used by training and data loading."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(DEFAULT_SEED)
torch.set_num_threads(6)
parser = argparse.ArgumentParser(description='SnowNet experiments')
# basic config
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument(
    '--model',
    type=str,
    required=True,
    default='SnowNet',
    help='registered baseline, fixed *_Snow integration, replacement variant, or SnowNet',
)
parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                    help='random seed used for Python, NumPy, PyTorch, and CUDA')
# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--d_core', type=int, default=512, help='dimension of core')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--attention_type', type=str, default="full", help='the attention type of transformer')
parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
parser.add_argument('--individual', type=int, default=1, help='use channel-independent parameters where supported')
parser.add_argument('--patch_len', type=int, default=16, help='patch length for PatchTST')
parser.add_argument('--stride', type=int, default=8, help='patch stride for PatchTST')
parser.add_argument('--n_heads', type=int, default=4, help='attention heads for PatchTST')
parser.add_argument('--seg_len', type=int, default=8, help='segment length for SegRNN')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden size for Amplifier')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension for GRU and TSMixer')
parser.add_argument('--num_layers', type=int, default=1, help='number of recurrent layers')
parser.add_argument('--kernel_size', type=int, default=3, help='kernel size for TCN')
parser.add_argument('--eps', type=float, default=1e-5, help='epsilon for reversible normalization')
parser.add_argument('--cycle', type=int, default=24, help='cycle length for CycleNet')
parser.add_argument('--model_type', type=str, default='linear', choices=['linear', 'mlp'], help='CycleNet predictor type')
parser.add_argument('--use_revin', type=int, default=1, help='use RevIN in CycleNet')
parser.add_argument('--snow_clusters', type=int, default=8, help='number of SnowNet local branches')
parser.add_argument('--snow_temperature', type=float, default=1.0, help='signed-softmax routing temperature for SnowNet')
parser.add_argument('--snow_residual_scale', type=float, default=0.1,
                    help='initial residual scale when SnowNet is used as a plugin')
parser.add_argument('--snow_learnable_scale', type=int, default=1,
                    help='learn the residual scale for internal Snow replacement')
parser.add_argument('--result_path', type=str, default='./results/snow_results.csv',
                    help='csv file used to append test metrics')
parser.add_argument('--result_data', type=str, default='',
                    help='dataset name written to the result csv; defaults to --data')
parser.add_argument('--train_noise_type', type=str, default='none',
                    choices=['none', 'gaussian'],
                    help='noise applied to training inputs only')
parser.add_argument('--train_noise_channel_ratio', type=float, default=0.0,
                    help='fraction of channels corrupted in the training inputs')
parser.add_argument('--train_noise_std', type=float, default=0.3,
                    help='Gaussian noise standard deviation as a multiple of each channel training standard deviation')
parser.add_argument('--train_noise_seed', type=int, default=None,
                    help='seed for fixed noisy channels and values; defaults to --seed')
parser.add_argument('--channel_shuffle_ratio', type=float, default=0.0,
                    help='fraction of channels included in one fixed permutation for all splits')
parser.add_argument('--channel_shuffle_seed', type=int, default=None,
                    help='seed for the fixed channel permutation; defaults to --seed')
parser.add_argument('--eval_channel_shuffle_ratio', type=float, default=None,
                    help='test-only channel permutation ratio; training and validation remain unchanged')
parser.add_argument('--eval_channel_shuffle_ratios', type=str, default='',
                    help='comma-separated test-only ratios evaluated after one normal training run')
parser.add_argument('--eval_channel_shuffle_seed', type=int, default=None,
                    help='seed for test-only channel permutations; defaults to --channel_shuffle_seed')
parser.add_argument('--profile_batches', type=int, default=1,
                    help='number of measured batches for inference efficiency profiling; 0 disables profiling')
parser.add_argument('--profile_warmup', type=int, default=1,
                    help='number of warmup batches for inference efficiency profiling')

# optimization
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

parser.add_argument('--save_model', action='store_true')

Exp = Exp_Long_Term_Forecast


def build_setting(args):
    base = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
            args.distil,
            args.des)
    if 'Snow' in args.model:
        base = '{}_dc{}_sk{}_st{}_ss{}_sl{}'.format(
            base,
            args.d_core,
            args.snow_clusters,
            args.snow_temperature,
            args.snow_residual_scale,
            args.snow_learnable_scale,
        )
    if args.channel_shuffle_ratio > 0.0:
        base = '{}_chshuffle{}_chseed{}'.format(
            base,
            format(args.channel_shuffle_ratio, 'g'),
            args.channel_shuffle_seed,
        )
    if args.train_noise_type != 'none' and args.train_noise_channel_ratio > 0.0:
        base = '{}_noise{}_cr{:g}_ns{:g}_nseed{}'.format(
            base,
            args.train_noise_type,
            args.train_noise_channel_ratio,
            args.train_noise_std,
            args.train_noise_seed,
        )
    return base


def parse_eval_channel_shuffle_ratios(value):
    ratios = []
    for item in (value or '').split(','):
        item = item.strip()
        if not item:
            continue
        ratio = float(item)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError('evaluation channel shuffle ratios must be in [0, 1]')
        if ratio not in ratios:
            ratios.append(ratio)
    return ratios


def build_eval_setting(training_setting, ratio, seed):
    return '{}_evalchshuffle{}_chseed{}'.format(
        training_setting,
        format(ratio, 'g'),
        seed,
    )


def train(args):
    setting = build_setting(args)
    exp = Exp(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    ratios = getattr(args, '_eval_channel_shuffle_ratios', [])
    if ratios:
        original_ratio = args.eval_channel_shuffle_ratio
        try:
            for ratio in ratios:
                args.eval_channel_shuffle_ratio = ratio
                eval_setting = build_eval_setting(
                    setting,
                    ratio,
                    args.eval_channel_shuffle_seed,
                )
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(eval_setting))
                exp.test(eval_setting)
        finally:
            args.eval_channel_shuffle_ratio = original_ratio
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
    torch.cuda.empty_cache()


def main():
    args = parser.parse_args()
    if not 0.0 <= args.train_noise_channel_ratio <= 1.0:
        parser.error('--train_noise_channel_ratio must be in [0, 1]')
    if args.train_noise_std < 0.0:
        parser.error('--train_noise_std must be non-negative')
    if args.train_noise_type == 'none' and args.train_noise_channel_ratio != 0.0:
        parser.error('--train_noise_channel_ratio must be 0 when --train_noise_type=none')
    if args.train_noise_type != 'none' and args.train_noise_channel_ratio == 0.0:
        parser.error('--train_noise_channel_ratio must be positive when training noise is enabled')
    if not 0.0 <= args.channel_shuffle_ratio <= 1.0:
        parser.error('--channel_shuffle_ratio must be in [0, 1]')
    if args.eval_channel_shuffle_ratio is not None and not 0.0 <= args.eval_channel_shuffle_ratio <= 1.0:
        parser.error('--eval_channel_shuffle_ratio must be in [0, 1]')
    try:
        args._eval_channel_shuffle_ratios = parse_eval_channel_shuffle_ratios(
            args.eval_channel_shuffle_ratios
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.eval_channel_shuffle_ratio is not None and args._eval_channel_shuffle_ratios:
        parser.error('use either --eval_channel_shuffle_ratio or --eval_channel_shuffle_ratios, not both')
    if not args.is_training and args._eval_channel_shuffle_ratios:
        parser.error('--eval_channel_shuffle_ratios requires --is_training 1')
    if args.train_noise_seed is None:
        args.train_noise_seed = args.seed
    if args.channel_shuffle_seed is None:
        args.channel_shuffle_seed = args.seed
    if args.eval_channel_shuffle_seed is None:
        args.eval_channel_shuffle_seed = args.channel_shuffle_seed
    args.train_noise_channel_count = 0
    args.train_noise_channel_hash = ''
    args.channel_shuffle_order = ''
    args.channel_shuffle_hash = ''
    args.active_channel_shuffle_ratio = 0.0
    args.active_channel_shuffle_seed = args.eval_channel_shuffle_seed
    args.active_channel_shuffle_count = 0
    args.active_channel_shuffle_hash = ''
    set_seed(args.seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    print('Args in experiment:')
    print(args)

    if args.is_training:
        train(args)
    else:
        setting = build_setting(args)
        exp = Exp(args)  # set experiments
        eval_setting = setting
        if args.eval_channel_shuffle_ratio is not None:
            eval_setting = build_eval_setting(
                setting,
                args.eval_channel_shuffle_ratio,
                args.eval_channel_shuffle_seed,
            )
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(eval_setting))
        exp.test(eval_setting, test=1, checkpoint_setting=setting)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    mp.freeze_support()
    main()
