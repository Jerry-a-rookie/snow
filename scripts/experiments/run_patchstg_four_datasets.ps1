param(
    [switch]$DryRun,
    [switch]$Resume,
    [string]$ResultDir = "",
    [string[]]$Models = @(),
    [string[]]$Datasets = @(),
    [int[]]$Windows = @(),
    [int]$ShardCount = 0,
    [int]$ShardIndex = -1,
    [int]$Gpu = -1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

$allModels = @(
    "NLinear", "RLinear", "GRU", "TCN", "CycleNet", "PatchTST",
    "TSMixer", "XLinear", "Amplifier", "SegRNN", "iTransformer",
    "NLinear_Snow", "RLinear_Snow", "GRU_Snow", "TCN_Snow",
    "CycleNet_Snow", "PatchTST_Snow", "TSMixer_Snow", "XLinear_Snow",
    "Amplifier_Snow", "SegRNN_Snow", "iTransformer_Snow",
    "iTransformer_Snow_Attn1", "iTransformer_Snow_AllAttn"
)

$datasetTable = [ordered]@{
    "CA"  = [pscustomobject]@{ Channels = 8600; BatchSize = 1 }
    "GBA" = [pscustomobject]@{ Channels = 2352; BatchSize = 2 }
    "GLA" = [pscustomobject]@{ Channels = 3834; BatchSize = 1 }
    "SD"  = [pscustomobject]@{ Channels = 716; BatchSize = 8 }
}

function Get-ListSetting {
    param([string]$Name)
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { return @() }
    return @($value -split "[,\s]+" | Where-Object { $_ })
}

function Get-IntSetting {
    param([string]$Name, [int]$Default)
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    return [int]$value
}

function Get-StringSetting {
    param([string]$Name, [string]$Default)
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    return $value
}

function Select-Known {
    param([object[]]$Available, [object[]]$Requested, [string]$Label)
    if ($null -eq $Requested -or $Requested.Count -eq 0) { return @($Available) }
    $unknown = @($Requested | Where-Object { $_ -notin $Available })
    if ($unknown.Count -gt 0) {
        throw "Unknown ${Label}: $($unknown -join ', '). Available: $($Available -join ', ')"
    }
    return @($Available | Where-Object { $_ -in $Requested })
}

function Quote-Argument {
    param([string]$Value)
    if ($Value -match '[\s"]') { return '"' + ($Value -replace '"', '\"') + '"' }
    return $Value
}

$dryRunValue = [Environment]::GetEnvironmentVariable("DRY_RUN")
if ($dryRunValue -and $dryRunValue.ToLowerInvariant() -in @("1", "true", "yes", "on")) {
    $DryRun = $true
}
$resumeValue = [Environment]::GetEnvironmentVariable("RESUME")
if ($resumeValue -and $resumeValue.ToLowerInvariant() -in @("1", "true", "yes", "on")) {
    $Resume = $true
}

$requestedModels = if ($Models.Count) { $Models } else { Get-ListSetting "MODELS" }
$selectedModels = @(Select-Known $allModels @($requestedModels) "model")
$requestedDatasets = if ($Datasets.Count) { $Datasets } else { Get-ListSetting "DATASETS" }
$selectedDatasets = @(Select-Known @($datasetTable.Keys) @($requestedDatasets) "dataset")
$requestedWindows = if ($Windows.Count) {
    $Windows
} else {
    @(Get-ListSetting "WINDOWS" | ForEach-Object { [int]$_ })
}
$selectedWindows = @(Select-Known @(6, 12) @($requestedWindows) "window")

if ($ShardCount -le 0) { $ShardCount = Get-IntSetting "SHARD_COUNT" 1 }
if ($ShardIndex -lt 0) { $ShardIndex = Get-IntSetting "SHARD_INDEX" 0 }
if ($Gpu -lt 0) { $Gpu = Get-IntSetting "GPU" 0 }
if ($ShardCount -lt 1 -or $ShardIndex -lt 0 -or $ShardIndex -ge $ShardCount) {
    throw "ShardIndex must be in [0, ShardCount), received $ShardIndex/$ShardCount."
}

$datasetRootValue = Get-StringSetting "PATCHSTG_ROOT" (Join-Path $projectRoot "dataset\PatchSTG")
$datasetRoot = if ([IO.Path]::IsPathRooted($datasetRootValue)) {
    $datasetRootValue
} else {
    Join-Path $projectRoot $datasetRootValue
}
$seed = Get-IntSetting "SEED" 2021
$trainEpochs = Get-IntSetting "TRAIN_EPOCHS" 20
$patience = Get-IntSetting "PATIENCE" 3
$numWorkers = Get-IntSetting "NUM_WORKERS" 4
$dModel = Get-IntSetting "D_MODEL" 32
$dCore = Get-IntSetting "D_CORE" 32
$dFf = Get-IntSetting "D_FF" 64
$eLayers = Get-IntSetting "E_LAYERS" 1
$nHeads = Get-IntSetting "N_HEADS" 4
$hiddenSize = Get-IntSetting "HIDDEN_SIZE" 32
$hiddenDim = Get-IntSetting "HIDDEN_DIM" 32
$batchOverride = Get-IntSetting "BATCH_SIZE" 0
$learningRate = Get-StringSetting "LEARNING_RATE" "0.0001"
$dropout = Get-StringSetting "DROPOUT" "0.0"

$missing = @()
foreach ($dataset in $selectedDatasets) {
    foreach ($file in @("flow.npy", "meta.csv", "manifest.json")) {
        $path = Join-Path (Join-Path $datasetRoot $dataset) $file
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { $missing += $path }
    }
}
if ($missing.Count -and -not $DryRun) {
    throw "Converted PatchSTG files are missing:`n$($missing -join "`n")"
}
if ($missing.Count) {
    Write-Warning "Dry-run is using unresolved converted dataset paths."
}

$resultDirValue = if ($ResultDir) {
    $ResultDir
} else {
    Get-StringSetting "RESULT_DIR" (Join-Path $projectRoot "results\patchstg_four_datasets")
}
$resultDir = if ([IO.Path]::IsPathRooted($resultDirValue)) {
    $resultDirValue
} else {
    Join-Path $projectRoot $resultDirValue
}
if ($ShardCount -gt 1) {
    $resultDir = Join-Path $resultDir ("shard_{0:D2}_of_{1:D2}" -f $ShardIndex, $ShardCount)
}
$logDir = Join-Path $resultDir "logs"
$commandDir = Join-Path $resultDir "commands"
$checkpointDir = Join-Path $resultDir "checkpoints"
$null = New-Item -ItemType Directory -Force -Path $resultDir, $logDir, $commandDir, $checkpointDir
$resultPath = Join-Path $resultDir "metrics.csv"
$manifestPath = Join-Path $resultDir "manifest.csv"

$completed = @{}
if ($Resume -and (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
    foreach ($row in Import-Csv -LiteralPath $resultPath) {
        $completed["$($row.data)|$($row.model)|$($row.seed)|$($row.seq_len)|$($row.pred_len)"] = $true
    }
}

$jobs = @()
$jobIndex = 0
foreach ($dataset in $selectedDatasets) {
    $meta = $datasetTable[$dataset]
    foreach ($window in $selectedWindows) {
        foreach ($model in $selectedModels) {
            $currentIndex = $jobIndex
            $jobIndex += 1
            if (($currentIndex % $ShardCount) -ne $ShardIndex) { continue }
            $batchSize = if ($batchOverride -gt 0) { $batchOverride } else { $meta.BatchSize }
            $modelId = "${dataset}_${model}_${window}to${window}_seed${seed}"
            $rootPath = Join-Path $datasetRoot $dataset
            $logPath = Join-Path $logDir "${modelId}.log"
            $commandPath = Join-Path $commandDir "${modelId}.ps1"
            $arguments = @(
                "-u", "run.py",
                "--task_name", "long_term_forecast",
                "--is_training", "1",
                "--model_id", $modelId,
                "--model", $model,
                "--seed", "$seed",
                "--data", "PatchSTG",
                "--result_data", $dataset,
                "--root_path", $rootPath,
                "--data_path", "flow.npy",
                "--features", "M",
                "--freq", "15min",
                "--seq_len", "$window",
                "--label_len", "0",
                "--pred_len", "$window",
                "--enc_in", "$($meta.Channels)",
                "--dec_in", "$($meta.Channels)",
                "--c_out", "$($meta.Channels)",
                "--d_model", "$dModel",
                "--d_core", "$dCore",
                "--d_ff", "$dFf",
                "--e_layers", "$eLayers",
                "--n_heads", "$nHeads",
                "--hidden_size", "$hiddenSize",
                "--hidden_dim", "$hiddenDim",
                "--individual", "0",
                "--dropout", "$dropout",
                "--patch_len", "3",
                "--stride", "3",
                "--seg_len", "3",
                "--cycle", "96",
                "--use_norm", "1",
                "--train_epochs", "$trainEpochs",
                "--patience", "$patience",
                "--batch_size", "$batchSize",
                "--num_workers", "$numWorkers",
                "--learning_rate", "$learningRate",
                "--lradj", "type1",
                "--itr", "1",
                "--gpu", "$Gpu",
                "--use_amp",
                "--des", "PatchSTGFourDatasets",
                "--result_path", $resultPath,
                "--checkpoints", $checkpointDir
            )
            $key = "$dataset|$model|$seed|$window|$window"
            $status = if ($Resume -and $completed.ContainsKey($key)) { "completed" } else { "pending" }
            $command = (@($python) + $arguments | ForEach-Object { Quote-Argument "$_" }) -join " "
            $jobs += [pscustomobject]@{
                JobIndex = $currentIndex
                Dataset = $dataset
                Model = $model
                Seed = $seed
                SeqLen = $window
                PredLen = $window
                BatchSize = $batchSize
                Gpu = $Gpu
                ModelId = $modelId
                Status = $status
                LogPath = $logPath
                CommandPath = $commandPath
                Command = $command
                Arguments = $arguments
            }
        }
    }
}

$jobs | Select-Object JobIndex, Dataset, Model, Seed, SeqLen, PredLen, BatchSize, Gpu, ModelId, Status, LogPath, Command |
    Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8

Write-Host "PatchSTG jobs: $($jobs.Count) selected / $($selectedDatasets.Count * $selectedWindows.Count * $selectedModels.Count) total"
Write-Host "Datasets: $($selectedDatasets -join ', '); models: $($selectedModels.Count); windows: $($selectedWindows -join ', ')"
Write-Host "Shard: $ShardIndex/$ShardCount; GPU: $Gpu; manifest: $manifestPath"

$runNumber = 0
foreach ($job in $jobs) {
    $runNumber += 1
    Write-Host "[$runNumber/$($jobs.Count)] $($job.Dataset) | $($job.Model) | $($job.SeqLen)->$($job.PredLen) | $($job.Status)"
    $scriptText = @(
        "`$ErrorActionPreference = `"Stop`""
        "Set-Location $(Quote-Argument $projectRoot)"
        "& $(Quote-Argument $python) $($job.Arguments | ForEach-Object { Quote-Argument `"$_`" } | Join-String -Separator ' ')"
        "exit `$LASTEXITCODE"
    ) -join [Environment]::NewLine
    Set-Content -LiteralPath $job.CommandPath -Value $scriptText -Encoding UTF8
    if ($job.Status -eq "completed" -or $DryRun) { continue }

    & $python @($job.Arguments) 2>&1 | Tee-Object -LiteralPath $job.LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed with exit code ${LASTEXITCODE}: $($job.ModelId)"
    }
}

if ($DryRun) {
    Write-Host "Dry-run complete. No training was started."
} else {
    Write-Host "Selected PatchSTG experiments complete."
}
