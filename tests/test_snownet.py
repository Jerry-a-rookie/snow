import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
from models.SnowNet import SnowNet, SnowResidualBlock


BASE_MODELS = [
    ("NLinear", "NLinear"),
    ("RLinear", "RLinear"),
    ("GRU", "GRU"),
    ("GRU_CI", "GRU_CI"),
    ("GRU_CD", "GRU_CD"),
    ("TCN", "TCN"),
    ("CycleNet", "cyclenet"),
    ("PatchTST", "PatchTST"),
    ("TSMixer", "TSMixer"),
    ("XLinear", "XLinear"),
    ("Amplifier", "amplifier"),
    ("SegRNN", "SegRNN"),
    ("iTransformer", "iTransformer"),
]

SNOW_MODELS = {
    "NLinear_Snow": [16, 8],
    "RLinear_Snow": [16, 8],
    "GRU_Snow": [16, 8],
    "GRU_CI_Snow": [16, 8],
    "GRU_CD_Snow": [16, 8],
    "TCN_Snow": [16],
    "CycleNet_Snow": [16, 8],
    "PatchTST_Snow": [8],
    "TSMixer_Snow": [16],
    "XLinear_Snow": [16],
    "Amplifier_Snow": [16, 16],
    "SegRNN_Snow": [8],
    "iTransformer_Snow": [8],
    "iTransformer_Snow_Attn1": [8],
    "iTransformer_Snow_AllAttn": [8, 8],
    "SnowNet": [8, 8],
}


def make_config(**overrides):
    values = {
        "task_name": "long_term_forecast",
        "seq_len": 16,
        "label_len": 8,
        "pred_len": 8,
        "enc_in": 4,
        "dec_in": 4,
        "c_out": 4,
        "individual": 0,
        "eps": 1e-5,
        "d_model": 8,
        "d_core": 6,
        "d_ff": 16,
        "e_layers": 2,
        "d_layers": 1,
        "factor": 1,
        "n_heads": 2,
        "dropout": 0.0,
        "activation": "gelu",
        "output_attention": False,
        "embed": "timeF",
        "freq": "h",
        "hidden_dim": 8,
        "hidden_size": 8,
        "num_layers": 1,
        "kernel_size": 3,
        "cycle": 4,
        "model_type": "linear",
        "use_revin": 1,
        "use_norm": 1,
        "patch_len": 4,
        "stride": 2,
        "seg_len": 4,
        "num_class": 3,
        "snow_clusters": 3,
        "snow_temperature": 1.0,
        "snow_residual_scale": 0.1,
        "snow_learnable_scale": 1,
        "cut_freq": 3,
        "SCI": 0,
        "alpha": 0.5,
        "beta": 0.5,
        "t_ff": 8,
        "c_ff": 8,
        "usenorm": 1,
        "embed_dropout": 0.0,
        "head_dropout": 0.0,
        "t_dropout": 0.0,
        "c_dropout": 0.0,
        "features": "M",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def load_model(module_name, config=None):
    module = importlib.import_module(f"models.{module_name}")
    return module.Model(config or make_config())


def capture_snow_shapes(model):
    shapes = []
    handles = []

    def record_shape(_module, inputs):
        shapes.append(tuple(inputs[0].shape))

    for module in model.modules():
        if isinstance(module, SnowNet):
            handles.append(module.register_forward_pre_hook(record_shape))
    return shapes, handles


def test_snownet_shape_routing_and_no_global_core():
    torch.manual_seed(7)
    model = SnowNet(d_series=16, d_core=8, n_clusters=4, dropout=0.0)
    x = torch.randn(2, 7, 16)

    output, route = model(x)

    assert output.shape == x.shape
    assert route.shape == (2, 7, 4)
    assert torch.allclose(route.abs().sum(dim=-1), torch.ones(2, 7), atol=1e-6)
    assert all("global" not in name.lower() for name, _ in model.named_modules())
    assert all("global" not in name.lower() for name, _ in model.named_parameters())


def test_signed_routes_support_both_directions_and_negative_coupling():
    model = SnowNet(d_series=4, d_core=2, n_clusters=4, dropout=0.0)
    with torch.no_grad():
        for parameter in model.route_proj.parameters():
            parameter.zero_()
        model.route_proj[-1].bias.copy_(torch.tensor([2.0, -1.0, 0.5, -0.25]))

    route = model._route(torch.randn(1, 1, 4))
    assert torch.any(route > 0)
    assert torch.any(route < 0)
    assert torch.allclose(route.abs().sum(dim=-1), torch.ones(1, 1), atol=1e-6)

    route = torch.tensor([[[0.8, 0.2], [-0.8, 0.2]]])
    values = torch.eye(2).unsqueeze(0)
    local = model._aggregate_local(values, route)
    mixing = model._write_back(local, route)
    assert mixing[0, 0, 1] < 0
    assert mixing[0, 1, 0] < 0


def test_absolute_mass_normalization_is_stable_under_cancellation():
    route = torch.tensor([[[1.0], [-1.0]]])
    values = torch.tensor([[[2.0, 1.0], [1.0, 3.0]]])

    local = SnowNet._aggregate_local(values, route)

    assert torch.isfinite(local).all()
    assert torch.allclose(local, torch.tensor([[[0.5, -1.0]]]), atol=1e-6)


def test_channel_permutation_equivariance_and_gradient_coverage():
    torch.manual_seed(11)
    model = SnowNet(d_series=16, d_core=8, n_clusters=4, dropout=0.0)
    model.eval()
    x = torch.randn(2, 7, 16)
    permutation = torch.randperm(x.size(1))

    output, route = model(x)
    permuted_output, permuted_route = model(x[:, permutation])

    assert torch.allclose(permuted_output, output[:, permutation], atol=1e-6)
    assert torch.allclose(permuted_route, route[:, permutation], atol=1e-6)

    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_residual_adapter_uses_the_documented_formula():
    block = SnowResidualBlock(
        d_series=8,
        d_core=4,
        n_clusters=3,
        dropout=0.0,
        residual_scale=0.25,
        learnable_scale=False,
    )
    block.eval()
    x = torch.randn(2, 4, 8)

    snow_output, _ = block.snow(x)
    actual = block.forward_tokens(x)

    assert torch.allclose(actual, x + 0.25 * (snow_output - x), atol=1e-6)


@pytest.mark.parametrize("registry_name,module_name", BASE_MODELS)
def test_pure_baseline_forward_and_source(registry_name, module_name):
    model = load_model(module_name)
    x = torch.randn(2, 16, 4)

    with torch.no_grad():
        output = model(x, None, None, None)

    assert output.shape == (2, 8, 4), registry_name
    assert not any(isinstance(module, SnowNet) for module in model.modules())
    source = Path(importlib.import_module(f"models.{module_name}").__file__).read_text(
        encoding="utf-8"
    )
    assert "models.SnowNet" not in source


@pytest.mark.parametrize("module_name,expected_last_dims", SNOW_MODELS.items())
def test_snow_model_forward_and_fixed_input_axes(module_name, expected_last_dims):
    model = load_model(module_name)
    shapes, handles = capture_snow_shapes(model)
    x = torch.randn(2, 16, 4)

    try:
        with torch.no_grad():
            output = model(x, None, None, None)
    finally:
        for handle in handles:
            handle.remove()

    assert output.shape == (2, 8, 4)
    assert len(shapes) == len(expected_last_dims)
    assert all(shape[1] == 4 for shape in shapes)
    assert [shape[-1] for shape in shapes] == expected_last_dims


def test_itransformer_replacement_counts_and_covariate_handling():
    first_module = importlib.import_module("models.iTransformer_Snow_Attn1")
    all_module = importlib.import_module("models.iTransformer_Snow_AllAttn")
    config = make_config(e_layers=3)

    first_model = first_module.Model(config)
    first_updates = [
        layer.attention
        for layer in first_model.encoder.attn_layers
        if isinstance(layer.attention, first_module.SnowAttentionUpdate)
    ]
    assert len(first_updates) == 1

    queries = torch.randn(2, 6, 8)
    update, _ = first_updates[0](queries, queries, queries)
    assert update.shape == queries.shape
    assert torch.count_nonzero(update[:, 4:, :]) == 0

    seen_shapes, handles = capture_snow_shapes(first_model)
    try:
        output = first_model(torch.randn(2, 16, 4), torch.randn(2, 16, 2))
    finally:
        for handle in handles:
            handle.remove()
    assert output.shape == (2, 8, 4)
    assert all(shape[1] == 4 for shape in seen_shapes)

    all_model = all_module.Model(config)
    all_updates = [
        layer.attention
        for layer in all_model.encoder.attn_layers
        if isinstance(layer.attention, all_module.SnowAttentionUpdate)
    ]
    assert len(all_updates) == config.e_layers

    embedded_shapes = []
    handle = all_model.enc_embedding.register_forward_hook(
        lambda _module, _inputs, output: embedded_shapes.append(tuple(output.shape))
    )
    try:
        output = all_model(torch.randn(2, 16, 4), torch.randn(2, 16, 2))
    finally:
        handle.remove()
    assert output.shape == (2, 8, 4)
    assert embedded_shapes == [(2, 4, 8)]


class RegistryProbe(Exp_Basic):
    def _build_model(self):
        return nn.Identity()


def test_registry_exposes_gru_cd_alias():
    registry = RegistryProbe(SimpleNamespace(use_gpu=False)).model_dict
    assert registry["GRU-CD"] is registry["GRU"]
    assert registry["GRU_CD"] is registry["GRU"]
    assert registry["GRU-CI"] is registry["GRU_CI"]
    assert registry["GRU-CI"] is not registry["GRU"]
    assert registry["GRU-CD-Snow"] is registry["GRU_Snow"]
    assert registry["GRU_CD_Snow"] is registry["GRU_Snow"]
    assert registry["GRU-CI-Snow"] is registry["GRU_CI_Snow"]
    assert registry["GRU-CI-Snow"] is not registry["GRU_Snow"]


def test_registry_contains_only_current_snow_model_names():
    registry = RegistryProbe(SimpleNamespace(use_gpu=False)).model_dict
    expected = set(SNOW_MODELS)
    assert expected.issubset(registry)

    removed_names = [
        "SOF" + "TS",
        "Snow" + "CI",
        "Snow" + "Replace",
        "DLinear",
        "Linear",
        "MLP",
        "LSTM",
        "FITS",
        "CrossLinear",
        "ModernTCN",
        "SnowDLinear",
        "SnowPatchTST",
        "SnowTSMixer",
        "SnowSegRNN",
        "SnowiTransformer",
        "DLinear_Snow",
        "Linear_Snow",
        "MLP_Snow",
        "LSTM_Snow",
        "FITS_Snow",
        "CrossLinear_Snow",
        "CrossLinear_Snow_Interaction",
        "ModernTCN_Snow",
    ]
    assert all(name not in registry for name in removed_names)
    module_names = [
        "SOF" + "TS",
        "Snow" + "CI",
        "Snow" + "Replace",
        "DLinear",
        "Linear",
        "mlp",
        "lstm",
        "FITS",
        "crosslinear",
        "ModernTCN",
        "DLinear_Snow",
        "Linear_Snow",
        "MLP_Snow",
        "LSTM_Snow",
        "FITS_Snow",
        "CrossLinear_Snow",
        "CrossLinear_Snow_Interaction",
        "ModernTCN_Snow",
        "FEDformer",
        "duet",
        "catch",
    ]
    assert all(importlib.util.find_spec(f"models.{name}") is None for name in module_names)


def test_only_structural_snow_cli_options_remain(monkeypatch):
    run_module = importlib.import_module("run")
    snow_options = {
        option
        for action in run_module.parser._actions
        for option in action.option_strings
        if option.startswith("--snow_")
    }
    assert snow_options == {
        "--snow_clusters",
        "--snow_temperature",
        "--snow_residual_scale",
        "--snow_learnable_scale",
    }
    assert not any(
        option.startswith("--ci_")
        for action in run_module.parser._actions
        for option in action.option_strings
    )

    removed_options = [
        "--snow_" + "version",
        "--snow_" + "component",
        "--snow_" + "components",
        "--snow_" + "layer",
        "--snow_" + "replace_mode",
        "--snow_" + "replace_layer",
        "--snow_" + "variant",
        "--snow_" + "topk",
        "--patchtst_" + "snow_attn",
        "--patchtst_" + "snow_layer",
        "--inter" + "action",
        "--ci_" + "base_model",
        "--ci_" + "use_snow",
    ]
    base_args = [
        "run.py",
        "--is_training",
        "0",
        "--model_id",
        "cli-check",
        "--model",
        "NLinear",
        "--data",
        "custom",
    ]
    for option in removed_options:
        monkeypatch.setattr(sys, "argv", base_args + [option, "1"])
        with pytest.raises(SystemExit):
            run_module.parser.parse_args()
