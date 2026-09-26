import csv
import importlib
import json
import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_HighDimTraffic
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from scripts.data.prepare_highdim_traffic import DATASETS, convert_dataset, inspect_npz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAFFIC_SOURCE = PROJECT_ROOT / "traffic_reference" / "data"
EXPECTED_SHAPES = {
    "CA": (35040, 8600, 1),
    "GBA": (35040, 2352, 1),
    "GLA": (35040, 3834, 1),
    "SD": (35040, 716, 1),
}
MODELS = {
    "NLinear": "NLinear",
    "RLinear": "RLinear",
    "GRU": "GRU",
    "GRU_CI": "GRU_CI",
    "GRU_CD": "GRU_CD",
    "TCN": "TCN",
    "CycleNet": "cyclenet",
    "PatchTST": "PatchTST",
    "TSMixer": "TSMixer",
    "XLinear": "XLinear",
    "Amplifier": "amplifier",
    "SegRNN": "SegRNN",
    "iTransformer": "iTransformer",
    "NLinear_Snow": "NLinear_Snow",
    "RLinear_Snow": "RLinear_Snow",
    "GRU_Snow": "GRU_Snow",
    "GRU_CI_Snow": "GRU_CI_Snow",
    "GRU_CD_Snow": "GRU_CD_Snow",
    "TCN_Snow": "TCN_Snow",
    "CycleNet_Snow": "CycleNet_Snow",
    "PatchTST_Snow": "PatchTST_Snow",
    "TSMixer_Snow": "TSMixer_Snow",
    "XLinear_Snow": "XLinear_Snow",
    "Amplifier_Snow": "Amplifier_Snow",
    "SegRNN_Snow": "SegRNN_Snow",
    "iTransformer_Snow": "iTransformer_Snow",
    "iTransformer_Snow_Attn1": "iTransformer_Snow_Attn1",
    "iTransformer_Snow_AllAttn": "iTransformer_Snow_AllAttn",
}


def _write_metadata(path, nodes):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Lat", "Lng"])
        writer.writerows((index, 0.0, 0.0) for index in range(nodes))


@pytest.fixture()
def converted_dataset(tmp_path):
    time_steps = 800
    nodes = 4
    values = np.arange(time_steps * nodes, dtype=np.float64).reshape(time_steps, nodes, 1)
    values[::37, 0, 0] = 0
    source_path = tmp_path / "source.npz"
    metadata_path = tmp_path / "source_meta.csv"
    output_path = tmp_path / "converted"
    np.savez(source_path, data=values)
    _write_metadata(metadata_path, nodes)
    manifest = convert_dataset(
        source_path,
        metadata_path,
        output_path,
        chunk_bytes=113,
    )
    return values[..., 0], output_path, manifest


def test_streaming_conversion_float32_statistics_and_idempotency(converted_dataset):
    values, output_path, manifest = converted_dataset
    flow = np.load(output_path / "flow.npy", mmap_mode="r")
    train_end = int(len(values) * 0.6)

    assert isinstance(flow, np.memmap)
    assert flow.dtype == np.float32
    assert flow.shape == values.shape
    np.testing.assert_array_equal(flow, values.astype(np.float32))
    assert manifest["train_mean"] == pytest.approx(
        np.mean(values[:train_end].astype(np.float32), dtype=np.float64)
    )
    assert manifest["train_std"] == pytest.approx(
        np.std(values[:train_end].astype(np.float32), dtype=np.float64)
    )

    before = (output_path / "flow.npy").stat().st_mtime_ns
    repeated = convert_dataset(
        manifest["source"]["path"],
        output_path / "meta.csv",
        output_path,
    )
    assert repeated == manifest
    assert (output_path / "flow.npy").stat().st_mtime_ns == before


def test_streaming_conversion_supports_fortran_order(tmp_path):
    time_steps = 43
    nodes = 5
    values = np.arange(time_steps * nodes, dtype=np.float64).reshape(
        time_steps,
        nodes,
        1,
        order="F",
    )
    values = np.asfortranarray(values)
    source_path = tmp_path / "source_fortran.npz"
    metadata_path = tmp_path / "source_fortran_meta.csv"
    output_path = tmp_path / "converted_fortran"
    np.savez(source_path, data=values)
    _write_metadata(metadata_path, nodes)

    source_info = inspect_npz(source_path)
    assert source_info["fortran_order"]
    manifest = convert_dataset(
        source_path,
        metadata_path,
        output_path,
        chunk_bytes=311,
    )
    flow = np.load(output_path / "flow.npy", mmap_mode="r")
    train_end = int(time_steps * 0.6)

    np.testing.assert_array_equal(flow, values[..., 0].astype(np.float32))
    assert manifest["train_mean"] == pytest.approx(
        np.mean(values[:train_end].astype(np.float32), dtype=np.float64)
    )
    assert manifest["train_std"] == pytest.approx(
        np.std(values[:train_end].astype(np.float32), dtype=np.float64)
    )


@pytest.mark.parametrize("window", [6, 12])
def test_dataset_splits_normalization_targets_and_time_wrap(converted_dataset, window):
    values, output_path, manifest = converted_dataset
    size = [window, 0, window, values.shape[1], values.shape[1], values.shape[1]]
    train = Dataset_HighDimTraffic(output_path, flag="train", size=size)
    val = Dataset_HighDimTraffic(output_path, flag="val", size=size)
    test = Dataset_HighDimTraffic(output_path, flag="test", size=size)
    train_end = manifest["split"]["train_end"]
    val_end = manifest["split"]["val_end"]

    assert len(train) == train_end - 2 * window + 1
    assert len(val) == (val_end - train_end) + window - 2 * window + 1
    assert len(test) == (len(values) - val_end) + window - 2 * window + 1

    x, y, x_mark, y_mark = val[0]
    np.testing.assert_allclose(
        x,
        (values[train_end - window:train_end] - manifest["train_mean"])
        / manifest["train_std"],
        rtol=1e-6,
    )
    np.testing.assert_array_equal(y, values[train_end:train_end + window])
    assert x_mark.shape == (window, 2)
    assert y_mark.shape == (window, 2)
    assert tuple(y_mark[0]) == (
        train_end % 96,
        (train_end // 96) % 7,
    )

    wrap_index = 96 - window
    _, _, wrap_x_mark, wrap_y_mark = train[wrap_index]
    assert tuple(wrap_x_mark[-1]) == (95, 0)
    assert tuple(wrap_y_mark[0]) == (0, 1)

    restored = pickle.loads(pickle.dumps(val))
    np.testing.assert_array_equal(restored[0][1], y)


def test_dataset_reopens_memmap_in_spawn_workers(converted_dataset):
    values, output_path, _ = converted_dataset
    channels = values.shape[1]
    dataset = Dataset_HighDimTraffic(
        output_path,
        flag="train",
        size=[6, 0, 6, channels, channels, channels],
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        multiprocessing_context="spawn",
    )
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(loader))

    assert batch_x.shape == (2, 6, channels)
    assert batch_y.shape == (2, 6, channels)
    assert batch_x_mark.shape == (2, 6, 2)
    assert batch_y_mark.shape == (2, 6, 2)


def test_highdim_traffic_masked_loss_uses_original_units():
    experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    data_set = SimpleNamespace(
        is_highdim_traffic=True,
        inverse_transform=lambda value: value * 10.0 + 5.0,
    )
    predictions = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    targets = torch.tensor([[[0.0, 10.0], [25.0, 40.0]]])

    loss = experiment._forecast_loss(
        predictions,
        targets,
        data_set,
        torch.nn.MSELoss(),
    )
    assert loss.item() == pytest.approx((5.0 + 0.0 + 5.0) / 3.0)


def _model_config(window):
    return SimpleNamespace(
        task_name="long_term_forecast",
        data="HighDimTraffic",
        seq_len=window,
        label_len=0,
        pred_len=window,
        enc_in=3,
        dec_in=3,
        c_out=3,
        individual=0,
        eps=1e-5,
        d_model=8,
        d_core=8,
        d_ff=16,
        e_layers=1,
        d_layers=1,
        factor=1,
        n_heads=2,
        dropout=0.0,
        activation="gelu",
        output_attention=False,
        embed="timeF",
        freq="15min",
        hidden_dim=8,
        hidden_size=8,
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
        cut_freq=3,
        SCI=0,
        alpha=0.5,
        beta=0.5,
        t_ff=8,
        c_ff=8,
        usenorm=1,
        embed_dropout=0.0,
        head_dropout=0.0,
        t_dropout=0.0,
        c_dropout=0.0,
        features="M",
    )


@pytest.mark.parametrize("window", [6, 12])
@pytest.mark.parametrize("registry_name,module_name", MODELS.items())
def test_all_highdim_traffic_models_forward(registry_name, module_name, window):
    model = importlib.import_module(f"models.{module_name}").Model(_model_config(window))
    inputs = torch.randn(1, window, 3)
    marks = torch.zeros(1, window, 2)
    with torch.no_grad():
        outputs = model(inputs, marks, None, marks)
    assert outputs.shape == (1, window, 3), registry_name


def test_source_shapes_and_metadata_rows_without_loading_arrays():
    if not TRAFFIC_SOURCE.exists():
        pytest.skip("raw high-dimensional traffic source data is not included in the code release")
    for dataset, (flow_name, meta_name) in DATASETS.items():
        info = inspect_npz(TRAFFIC_SOURCE / flow_name)
        assert info["shape"] == EXPECTED_SHAPES[dataset]
        with (TRAFFIC_SOURCE / meta_name).open(
            "r", newline="", encoding="utf-8-sig"
        ) as handle:
            rows = sum(1 for _ in csv.reader(handle)) - 1
        assert rows == EXPECTED_SHAPES[dataset][1]


def test_experiment_scripts_define_224_jobs_without_standalone_snownet():
    powershell = (
        PROJECT_ROOT / "scripts" / "experiments" / "run_highdim_traffic_four_datasets.ps1"
    ).read_text(encoding="utf-8")
    bash = (
        PROJECT_ROOT / "scripts" / "experiments" / "run_highdim_traffic_four_datasets.sh"
    ).read_text(encoding="utf-8")

    assert len(MODELS) == 28
    assert "SnowNet" not in MODELS
    for model in MODELS:
        assert re.search(rf'(?<![A-Za-z0-9_]){re.escape(model)}(?![A-Za-z0-9_])', powershell)
        assert re.search(rf'(?<![A-Za-z0-9_]){re.escape(model)}(?![A-Za-z0-9_])', bash)
    assert len(DATASETS) * 2 * len(MODELS) == 224


def test_all_dataset_linux_entrypoint_defaults_to_four_datasets():
    entrypoint = (
        PROJECT_ROOT
        / "scripts"
        / "experiments"
        / "run_highdim_traffic_all_datasets.sh"
    ).read_text(encoding="utf-8")
    assert 'CA GBA GLA SD' in entrypoint
    assert "SMOKE_TEST" in entrypoint
    assert "run_highdim_traffic_four_datasets.sh" in entrypoint
    assert "smoke_highdim_traffic_four_datasets.py" in entrypoint


def test_converted_real_datasets_when_present():
    converted_root = PROJECT_ROOT / "dataset" / "highdim_traffic"
    if not converted_root.exists():
        pytest.skip("real high-dimensional traffic conversion has not been generated")
    for dataset, expected_source_shape in EXPECTED_SHAPES.items():
        manifest_path = converted_root / dataset / "manifest.json"
        if not manifest_path.exists():
            pytest.skip("real high-dimensional traffic conversion is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        flow = np.load(converted_root / dataset / "flow.npy", mmap_mode="r")
        assert flow.shape == expected_source_shape[:2]
        assert flow.dtype == np.float32
        assert manifest["metadata_rows"] == expected_source_shape[1]
