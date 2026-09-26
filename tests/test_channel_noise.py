import numpy as np
from types import SimpleNamespace

from data_provider.data_factory import (
    ChannelPermutationDataset,
    TrainChannelNoiseDataset,
    resolve_channel_shuffle_config,
)


class DummyWindowDataset:
    def __init__(self):
        time = np.arange(24, dtype=np.float32)[:, None]
        channel = np.arange(10, dtype=np.float32)[None, :]
        self.data_x = time * (channel + 1.0) + channel
        self.data_y = self.data_x.copy()
        self.seq_len = 6
        self.label_len = 2
        self.pred_len = 3

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (
            self.data_x[index:s_end],
            self.data_y[r_begin:r_end],
            np.zeros((self.seq_len, 1), dtype=np.float32),
            np.zeros((self.label_len + self.pred_len, 1), dtype=np.float32),
        )


def test_fixed_channel_noise_is_nested_and_keeps_targets_clean():
    clean = DummyWindowDataset()
    noisy_30 = TrainChannelNoiseDataset(clean, 0.3, 0.3, 2021)
    noisy_50 = TrainChannelNoiseDataset(clean, 0.5, 0.3, 2021)

    assert noisy_30.noisy_channel_count == 3
    assert noisy_50.noisy_channel_count == 5
    assert set(noisy_30.noisy_channels).issubset(set(noisy_50.noisy_channels))

    shared = noisy_30.noisy_channels
    np.testing.assert_allclose(
        noisy_30.noisy_data_x[:, shared],
        noisy_50.noisy_data_x[:, shared],
    )

    untouched = np.setdiff1d(np.arange(10), noisy_30.noisy_channels)
    np.testing.assert_allclose(
        noisy_30.noisy_data_x[:, untouched],
        clean.data_x[:, untouched],
    )

    noisy_sample = noisy_30[2]
    clean_sample = clean[2]
    assert not np.array_equal(noisy_sample[0], clean_sample[0])
    np.testing.assert_array_equal(noisy_sample[1], clean_sample[1])
    np.testing.assert_array_equal(noisy_sample[2], clean_sample[2])
    np.testing.assert_array_equal(noisy_sample[3], clean_sample[3])


def test_fixed_channel_noise_is_reproducible():
    clean = DummyWindowDataset()
    first = TrainChannelNoiseDataset(clean, 0.7, 0.3, 2021)
    second = TrainChannelNoiseDataset(clean, 0.7, 0.3, 2021)

    np.testing.assert_array_equal(first.noisy_channels, second.noisy_channels)
    np.testing.assert_allclose(first.noisy_data_x, second.noisy_data_x)
    assert first.noisy_channel_hash == second.noisy_channel_hash


def test_channel_permutation_is_reproducible_and_permutes_targets():
    clean = DummyWindowDataset()
    first = ChannelPermutationDataset(clean, 1.0, 2021)
    second = ChannelPermutationDataset(clean, 1.0, 2021)

    assert first.channel_shuffle_count == clean.data_x.shape[1]
    assert not np.array_equal(
        first.channel_order, np.arange(clean.data_x.shape[1])
    )
    np.testing.assert_array_equal(first.channel_order, second.channel_order)
    assert first.channel_shuffle_hash == second.channel_shuffle_hash

    clean_sample = clean[2]
    permuted_sample = first[2]
    np.testing.assert_array_equal(
        permuted_sample[0], clean_sample[0][..., first.channel_order]
    )
    np.testing.assert_array_equal(
        permuted_sample[1], clean_sample[1][..., first.channel_order]
    )


def test_eval_channel_permutation_overrides_only_test_split():
    args = SimpleNamespace(
        seed=2021,
        channel_shuffle_ratio=0.0,
        channel_shuffle_seed=11,
        eval_channel_shuffle_ratio=1.0,
        eval_channel_shuffle_seed=22,
    )

    assert resolve_channel_shuffle_config(args, 'train') == (0.0, 11)
    assert resolve_channel_shuffle_config(args, 'val') == (0.0, 11)
    assert resolve_channel_shuffle_config(args, 'test') == (1.0, 22)

    args.eval_channel_shuffle_ratio = None
    args.channel_shuffle_ratio = 0.5
    assert resolve_channel_shuffle_config(args, 'train') == (0.5, 11)
    assert resolve_channel_shuffle_config(args, 'test') == (0.5, 11)
