#!/usr/bin/env python3
"""Run a lightweight forward smoke test for the high-dimensional traffic datasets."""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow direct execution from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.data_loader import Dataset_HighDimTraffic


DATASET_CHANNELS = {
    "CA": 8600,
    "GBA": 2352,
    "GLA": 3834,
    "SD": 716,
}
MODEL_MODULES = {
    "NLinear": "NLinear",
    "GRU": "GRU",
    "PatchTST": "PatchTST",
    "SegRNN": "SegRNN",
}
DEFAULT_MODELS = tuple(MODEL_MODULES)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "highdim_traffic",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CHANNELS),
    )
    parser.add_argument("--windows", nargs="+", type=int, default=[6, 12])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def model_config(window, channels, args):
    return SimpleNamespace(
        task_name="long_term_forecast",
        data="HighDimTraffic",
        seq_len=window,
        label_len=0,
        pred_len=window,
        enc_in=channels,
        dec_in=channels,
        c_out=channels,
        individual=0,
        eps=1e-5,
        d_model=args.d_model,
        d_core=args.d_model,
        d_ff=args.d_ff,
        e_layers=1,
        d_layers=1,
        factor=1,
        n_heads=4,
        dropout=0.0,
        activation="gelu",
        output_attention=False,
        embed="timeF",
        freq="15min",
        hidden_dim=args.hidden_dim,
        hidden_size=args.hidden_dim,
        num_layers=1,
        kernel_size=3,
        cycle=96,
        model_type="linear",
        use_revin=1,
        use_norm=1,
        patch_len=3,
        stride=3,
        seg_len=3,
        num_class=3,
        snow_clusters=3,
        snow_temperature=1.0,
        snow_residual_scale=0.1,
        snow_learnable_scale=1,
        features="M",
    )


def validate_names(args):
    unknown_datasets = sorted(set(args.datasets) - set(DATASET_CHANNELS))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {', '.join(unknown_datasets)}")
    unknown_models = sorted(set(args.models) - set(MODEL_MODULES))
    if unknown_models:
        raise ValueError(f"Unknown smoke models: {', '.join(unknown_models)}")
    invalid_windows = sorted(set(args.windows) - {6, 12})
    if invalid_windows:
        raise ValueError(f"Smoke windows must be 6 or 12, got {invalid_windows}")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")


def run_case(dataset_root, dataset_name, window, model_name, args):
    channels = DATASET_CHANNELS[dataset_name]
    dataset_path = dataset_root / dataset_name
    with (dataset_path / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if tuple(manifest["flow_shape"]) != (35040, channels):
        raise ValueError(
            f"{dataset_name} manifest shape does not match expected "
            f"(35040, {channels}): {manifest['flow_shape']}"
        )

    dataset = Dataset_HighDimTraffic(
        dataset_path,
        flag="train",
        size=[window, 0, window, channels, channels, channels],
    )
    loader_kwargs = {
        "batch_size": min(args.batch_size, len(dataset)),
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = False
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(
        iter(DataLoader(dataset, **loader_kwargs))
    )
    if batch_x.shape[-2:] != (window, channels):
        raise ValueError(f"{dataset_name} {window}: bad input shape {batch_x.shape}")
    if batch_y.shape[-2:] != (window, channels):
        raise ValueError(f"{dataset_name} {window}: bad target shape {batch_y.shape}")

    module = importlib.import_module(f"models.{MODEL_MODULES[model_name]}")
    model = module.Model(model_config(window, channels, args)).eval()
    started = time.perf_counter()
    with torch.no_grad():
        outputs = model(batch_x.float(), batch_x_mark.float(), None, batch_y_mark.float())
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not torch.isfinite(outputs).all():
        raise ValueError(f"{dataset_name} {window} {model_name}: non-finite output")
    expected_shape = (batch_x.shape[0], window, channels)
    if tuple(outputs.shape) != expected_shape:
        raise ValueError(
            f"{dataset_name} {window} {model_name}: "
            f"expected {expected_shape}, got {tuple(outputs.shape)}"
        )

    restored = dataset.inverse_transform(outputs)
    if not torch.isfinite(restored).all():
        raise ValueError(f"{dataset_name} {window} {model_name}: bad inverse transform")
    return {
        "dataset": dataset_name,
        "model": model_name,
        "seq_len": window,
        "pred_len": window,
        "channels": channels,
        "batch_size": int(batch_x.shape[0]),
        "input_dtype": str(batch_x.dtype),
        "output_shape": list(outputs.shape),
        "forward_ms": round(elapsed_ms, 3),
    }


def main():
    args = parse_args()
    validate_names(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))

    results = []
    for dataset_name in args.datasets:
        for window in args.windows:
            for model_name in args.models:
                result = run_case(
                    args.dataset_root,
                    dataset_name,
                    window,
                    model_name,
                    args,
                )
                results.append(result)
                print(
                    f"PASS {dataset_name} {model_name} "
                    f"{window}->{window} channels={result['channels']} "
                    f"forward_ms={result['forward_ms']}"
                )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Smoke test passed: {len(results)} cases.")


if __name__ == "__main__":
    main()
