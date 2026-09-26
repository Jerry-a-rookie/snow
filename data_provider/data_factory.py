import hashlib
import math
import hashlib

import numpy as np
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_LongForecast, \
    Dataset_Solar, Dataset_PEMS, Dataset_PatchSTG, Dataset_Pred, Dataset_Random
from torch.utils.data import DataLoader, Dataset

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'LongForecast': Dataset_LongForecast,
    'Crop': Dataset_Custom,
    'crop': Dataset_Custom,
    'random': Dataset_Random,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'PatchSTG': Dataset_PatchSTG,
}


class TrainChannelNoiseDataset(Dataset):
    """Expose clean targets while reading inputs from one fixed noisy timeline."""

    def __init__(self, dataset, channel_ratio, noise_std, noise_seed):
        if not 0.0 < channel_ratio <= 1.0:
            raise ValueError('train_noise_channel_ratio must be in (0, 1]')
        if noise_std < 0.0:
            raise ValueError('train_noise_std must be non-negative')
        if not hasattr(dataset, 'data_x') or not hasattr(dataset, 'seq_len'):
            raise TypeError('channel noise requires a window dataset with data_x and seq_len')

        clean_data_x = np.asarray(dataset.data_x)
        if clean_data_x.ndim != 2:
            raise ValueError(
                f'channel noise expects data_x shaped [time, channel], got {clean_data_x.shape}'
            )

        self.dataset = dataset
        self.noise_seed = int(noise_seed)
        self.channel_ratio = float(channel_ratio)
        self.noise_std = float(noise_std)
        self.channel_count = clean_data_x.shape[1]
        self.noisy_channel_count = max(
            1,
            min(
                self.channel_count,
                int(math.floor(self.channel_count * self.channel_ratio + 0.5)),
            ),
        )

        channel_rng = np.random.default_rng(
            np.random.SeedSequence([self.noise_seed, 0])
        )
        channel_order = channel_rng.permutation(self.channel_count)
        self.noisy_channels = np.sort(channel_order[:self.noisy_channel_count])
        self.noisy_channel_hash = hashlib.sha256(
            self.noisy_channels.astype('<i8', copy=False).tobytes()
        ).hexdigest()[:16]

        work_dtype = np.result_type(clean_data_x.dtype, np.float32)
        self.noisy_data_x = np.array(clean_data_x, dtype=work_dtype, copy=True)
        channel_scale = np.nanstd(clean_data_x, axis=0, ddof=0)
        time_steps = clean_data_x.shape[0]

        # A per-channel RNG keeps shared channel noise identical for nested ratios.
        for channel in self.noisy_channels:
            value_rng = np.random.default_rng(
                np.random.SeedSequence([self.noise_seed, 1, int(channel)])
            )
            noise = value_rng.standard_normal(time_steps).astype(work_dtype, copy=False)
            self.noisy_data_x[:, channel] += (
                noise * self.noise_std * channel_scale[channel]
            )

    def __getitem__(self, index):
        sample = list(self.dataset[index])
        sample[0] = self.noisy_data_x[index:index + self.dataset.seq_len]
        return tuple(sample)

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class ChannelPermutationDataset(Dataset):
    """Apply one deterministic channel permutation to all data splits."""

    def __init__(self, dataset, shuffle_ratio, shuffle_seed):
        if not 0.0 < shuffle_ratio <= 1.0:
            raise ValueError('channel_shuffle_ratio must be in (0, 1]')
        if not hasattr(dataset, 'data_x') or not hasattr(dataset, 'data_y'):
            raise TypeError(
                'channel permutation requires a window dataset with data_x and data_y'
            )

        self.dataset = dataset
        self.channel_shuffle_ratio = float(shuffle_ratio)
        self.channel_shuffle_seed = int(shuffle_seed)
        self.channel_count = int(np.asarray(dataset.data_x).shape[1])

        count = int(round(self.channel_count * self.channel_shuffle_ratio))
        count = min(self.channel_count, max(2, count))
        self.channel_shuffle_count = count
        rng = np.random.default_rng(self.channel_shuffle_seed)
        order = np.arange(self.channel_count, dtype=np.int64)
        selected = rng.choice(self.channel_count, size=count, replace=False)
        shuffled = selected.copy()
        while count > 1:
            rng.shuffle(shuffled)
            if not np.array_equal(selected, shuffled):
                break
        order[selected] = shuffled
        self.channel_order = order
        self.data_x = np.asarray(dataset.data_x)[:, self.channel_order]
        self.data_y = np.asarray(dataset.data_y)[:, self.channel_order]
        self.channel_shuffle_hash = hashlib.sha256(
            self.channel_order.astype('<i8', copy=False).tobytes()
        ).hexdigest()[:16]

    def __getitem__(self, index):
        sample = list(self.dataset[index])
        sample[0] = sample[0][..., self.channel_order]
        sample[1] = sample[1][..., self.channel_order]
        return tuple(sample)

    def __len__(self):
        return len(self.dataset)

    def inverse_transform(self, data):
        scaler = getattr(self.dataset, 'scaler', None)
        if scaler is None or not hasattr(scaler, 'mean_'):
            return data
        mean = np.asarray(scaler.mean_)[self.channel_order]
        scale = np.asarray(scaler.scale_)[self.channel_order]
        if hasattr(data, 'detach'):
            import torch
            mean = torch.as_tensor(mean, device=data.device, dtype=data.dtype)
            scale = torch.as_tensor(scale, device=data.device, dtype=data.dtype)
            return data * scale + mean
        return data * scale + mean

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def resolve_channel_shuffle_config(args, flag):
    """Return the effective permutation ratio and seed for one split.

    ``channel_shuffle_ratio`` keeps its historical all-split behavior.
    ``eval_channel_shuffle_ratio`` overrides it only for the test split so a
    normally trained checkpoint can be evaluated under multiple orders.
    """
    ratio = float(getattr(args, 'channel_shuffle_ratio', 0.0))
    seed = getattr(args, 'channel_shuffle_seed', None)
    eval_ratio = getattr(args, 'eval_channel_shuffle_ratio', None)
    if flag == 'test' and eval_ratio is not None:
        ratio = float(eval_ratio)
        seed = getattr(args, 'eval_channel_shuffle_seed', None)
    if seed is None:
        seed = getattr(args, 'seed', 2021)
    return ratio, int(seed)


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.enc_in,
            args.dec_in,
            args.c_out,
        ],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )
    shuffle_ratio, shuffle_seed = resolve_channel_shuffle_config(args, flag)
    args.active_channel_shuffle_ratio = shuffle_ratio
    args.active_channel_shuffle_seed = shuffle_seed
    args.active_channel_shuffle_count = 0
    args.active_channel_shuffle_hash = ''
    args.channel_shuffle_order = ''
    args.channel_shuffle_hash = ''
    if flag in ('train', 'val', 'test') and shuffle_ratio > 0.0:
        data_set = ChannelPermutationDataset(
            data_set,
            shuffle_ratio=shuffle_ratio,
            shuffle_seed=shuffle_seed,
        )
        args.active_channel_shuffle_count = data_set.channel_shuffle_count
        args.active_channel_shuffle_hash = data_set.channel_shuffle_hash
        args.channel_shuffle_order = ','.join(
            map(str, data_set.channel_order.tolist())
        )
        args.channel_shuffle_hash = data_set.channel_shuffle_hash
        print(
            f'{flag} channel permutation: '
            f'ratio={shuffle_ratio:g}, channels={data_set.channel_count}, '
            f'seed={shuffle_seed}, order_hash={data_set.channel_shuffle_hash}'
        )
    noise_type = getattr(args, 'train_noise_type', 'none')
    noise_ratio = float(getattr(args, 'train_noise_channel_ratio', 0.0))
    if flag == 'train' and noise_type != 'none' and noise_ratio > 0.0:
        if getattr(data_set, 'is_patchstg', False):
            raise ValueError(
                'training channel noise is disabled for memory-mapped PatchSTG datasets'
            )
        if noise_type != 'gaussian':
            raise ValueError(f'unsupported train noise type: {noise_type}')
        noise_seed = getattr(args, 'train_noise_seed', None)
        if noise_seed is None:
            noise_seed = getattr(args, 'seed', 2021)
        data_set = TrainChannelNoiseDataset(
            data_set,
            channel_ratio=noise_ratio,
            noise_std=float(getattr(args, 'train_noise_std', 0.3)),
            noise_seed=noise_seed,
        )
        args.train_noise_channel_count = data_set.noisy_channel_count
        args.train_noise_channel_hash = data_set.noisy_channel_hash
        print(
            'train channel noise: '
            f'type={noise_type}, ratio={noise_ratio:g}, '
            f'channels={data_set.noisy_channel_count}/{data_set.channel_count}, '
            f'std={data_set.noise_std:g}, seed={data_set.noise_seed}, '
            f'channel_hash={data_set.noisy_channel_hash}'
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
