$ErrorActionPreference = "Stop"

function Format-Arg {
  param([string]$Value)
  if ($Value -match '\s') {
    return '"' + $Value + '"'
  }
  return $Value
}

function Invoke-RunPy {
  param(
    [string[]]$RunArgs,
    [bool]$DryRun = $false
  )

  $cmd = "python -u run.py " + (($RunArgs | ForEach-Object { Format-Arg $_ }) -join " ")
  Write-Host $cmd

  if (-not $DryRun) {
    & python -u run.py @RunArgs
    if ($LASTEXITCODE -ne 0) {
      throw "run.py failed with exit code $LASTEXITCODE"
    }
  }
}

function Write-SnowNetSnapshot {
  param(
    [string]$ResultPath,
    [string]$SnapshotPath,
    [string]$DataName,
    [int]$PredLen
  )

  if (-not (Test-Path $ResultPath)) {
    return
  }

  $rows = Import-Csv $ResultPath |
    Where-Object {
      $_.data -eq $DataName -and [int]$_.pred_len -eq $PredLen
    } |
    Sort-Object { [double]$_.mse }

  if (-not $rows) {
    return
  }

  $snapshotDir = Split-Path -Parent $SnapshotPath
  if ($snapshotDir) {
    New-Item -ItemType Directory -Force -Path $snapshotDir | Out-Null
  }

  $rows |
    Select-Object data, model_id, model, pred_len, mse, mae |
    Export-Csv -NoTypeInformation -Encoding utf8 -Path $SnapshotPath
}

function New-SnowNetArgs {
  param(
    [hashtable]$Config,
    [int]$PredLen,
    [string]$ModelId,
    [string]$ResultPath
  )

  $runArgs = @(
    "--task_name", "long_term_forecast",
    "--is_training", "1",
    "--root_path", $Config.RootPath,
    "--data_path", $Config.DataPath,
    "--model_id", $ModelId,
    "--model", "SnowNet",
    "--data", $Config.Data,
    "--result_data", $Config.Name,
    "--features", "M",
    "--target", "OT",
    "--freq", $Config.Freq,
    "--seq_len", "96",
    "--label_len", "48",
    "--pred_len", [string]$PredLen,
    "--e_layers", [string]$Config.ELayers,
    "--enc_in", [string]$Config.EncIn,
    "--dec_in", [string]$Config.EncIn,
    "--c_out", [string]$Config.EncIn,
    "--d_model", [string]$Config.DModel,
    "--d_core", [string]$Config.DCore,
    "--d_ff", [string]$Config.DFF,
    "--learning_rate", [string]$Config.LearningRate,
    "--lradj", "cosine",
    "--train_epochs", [string]$Config.TrainEpochs,
    "--patience", [string]$Config.Patience,
    "--des", "Exp",
    "--itr", "1",
    "--result_path", $ResultPath
  )

  if ($Config.ContainsKey("BatchSize")) {
    $runArgs += @("--batch_size", [string]$Config.BatchSize)
  }
  if ($Config.ContainsKey("UseNorm")) {
    $runArgs += @("--use_norm", [string]$Config.UseNorm)
  }
  $runArgs += @("--snow_clusters", [string]$Config.SnowClusters)

  return $runArgs
}

function Invoke-SnowNetSuite {
  param(
    [object[]]$Datasets,
    [string]$ResultPath,
    [string]$SnapshotDir = "",
    [bool]$DryRun = $false
  )

  foreach ($config in $Datasets) {
    foreach ($predLen in $config.PredLens) {
      $snowModelId = "$($config.Name)_96_${predLen}_SnowNet"
      $snowArgs = New-SnowNetArgs `
        -Config $config `
        -PredLen $predLen `
        -ModelId $snowModelId `
        -ResultPath $ResultPath
      Invoke-RunPy -RunArgs $snowArgs -DryRun $DryRun

      if ($SnapshotDir -and (-not $DryRun)) {
        $snapshotPath = Join-Path $SnapshotDir "$($config.Name)_${predLen}.csv"
        Write-SnowNetSnapshot -ResultPath $ResultPath -SnapshotPath $snapshotPath -DataName $config.Name -PredLen $predLen
      }
    }
  }

  if ((Test-Path $ResultPath) -and (-not $DryRun)) {
    Import-Csv $ResultPath | Sort-Object data, pred_len, { [double]$_.mse } |
      Format-Table data, model_id, model, pred_len, mse, mae -AutoSize
  }
}
