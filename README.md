<div align="center">

# Snow Forecasting

**A compact, reproducible code release for Snow and its forecasting baselines**

<sub>Anonymous release · PyTorch · long-term forecasting · PatchSTG traffic settings</sub>

</div>

<p align="center">
  <img src="assets/figures/pems_bay.png" alt="PEMS-BAY appendix interaction analysis" width="100%">
</p>

<p align="center"><sub>Example appendix analysis on PEMS-BAY: signed routing, Snow interaction, and correlation structure.</sub></p>

## At a glance

Snow is a forecasting architecture designed to model signed cross-channel
interactions while retaining a simple long-term forecasting interface. This
repository contains the model variants used in the paper, their baseline
counterparts, the main experiment launchers, and lightweight verification
tests.

| Included | Not included |
| --- | --- |
| Snow and baseline model implementations | Datasets and dataset downloads |
| Standard long-term forecasting runners | Checkpoints and generated results |
| PatchSTG four-dataset runners | Manuscript sources and review material |
| Data-preparation and smoke-test utilities | Author, server, and repository metadata |

## Visual overview

The appendix analyses below are included as a compact visual guide to the
experiments. They are not required to run the code.

<table>
<tr>
<td width="50%"><img src="assets/figures/ecl.png" alt="ECL appendix interaction analysis" width="100%"><br><sub><b>ECL.</b> Signed routing and interaction patterns.</sub></td>
<td width="50%"><img src="assets/figures/weather.png" alt="Weather appendix interaction analysis" width="100%"><br><sub><b>Weather.</b> The same analysis on a low-dimensional dataset.</sub></td>
</tr>
</table>

## Repository layout

```text
run.py                         Main training/evaluation entry point
models/                        Snow and baseline forecasting models
layers/                        Shared neural-network layers
data_provider/                 Dataset readers and preprocessing
exp/                           Training and evaluation loops
configs/                       Non-sensitive experiment configuration
scripts/data/                  PatchSTG data preparation helper
scripts/experiments/           Main PatchSTG traffic runners
scripts/long_term_forecast/    Main dataset-specific forecasting runners
tests/                         Lightweight model and runner tests
assets/figures/                Appendix figures shown in this README
requirements.txt               Python dependencies
LICENSE                        Release license
```

## Quick start

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: activate the .venv environment in PowerShell
python -m pip install -r requirements.txt
```

Datasets are intentionally not bundled. Pass their locations through the
existing `--root_path` argument or environment variables such as `PEMS_ROOT`,
`FORECASTING_ROOT`, and `RESULT_DIR`. Keep outputs, checkpoints, logs, and
temporary files outside the source tree or in ignored directories.

## Main experiment entry points

### Standard long-term forecasting

The dataset-specific wrappers share one interface and use the 96-step input
setting from the paper. For example:

```bash
bash scripts/long_term_forecast/ETT_script/SnowNet_ETTh1.sh
bash scripts/long_term_forecast/ECL_script/SnowNet.sh
bash scripts/long_term_forecast/Traffic_script/SnowNet.sh
```

Additional standard datasets are available under
`scripts/long_term_forecast/` (ETTh2, ETTm1, ETTm2, Solar, Weather, and PEMS).

### PatchSTG traffic settings

The main traffic runner covers CA, GBA, GLA, and SD with the 6-to-6 and
12-to-12 settings:

```bash
bash scripts/experiments/run_patchstg_all_datasets.sh
```

Useful controls are exposed as environment variables, for example:

```bash
DRY_RUN=1 DATASETS="CA GBA" MODELS="SnowNet PatchTST_Snow" \
  bash scripts/experiments/run_patchstg_all_datasets.sh
```

If the raw PatchSTG files are available, the runner can prepare the converted
dataset layout using `scripts/data/prepare_patchstg.py`.

## Verification

Run the lightweight model and runner checks before launching experiments:

```bash
python -B -m pytest -q
```

Most unit tests use synthetic inputs. The PatchSTG smoke runner expects
converted traffic files under `dataset/PatchSTG` and can be launched with:

```bash
SMOKE_TEST=1 bash scripts/experiments/run_patchstg_all_datasets.sh
```

## Anonymous-release checklist

- No author names, affiliations, email addresses, private URLs, usernames, or
  machine-specific absolute paths are included.
- No datasets, checkpoints, logs, result dumps, manuscript sources, review
  files, or Git metadata are included.
- Dataset and output paths are supplied by arguments or environment variables;
  the scripts contain no local server paths.
- Before archiving, inspect hidden files and rerun a path/privacy scan on the
  final archive.

## License

See [LICENSE](LICENSE). Dataset terms remain the responsibility of the user.
