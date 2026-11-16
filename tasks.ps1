# Цели запуска. make в системе нет, поэтому это и есть «Makefile».
#   .\tasks.ps1 help
#   .\tasks.ps1 demo
#   .\tasks.ps1 video -Source video.mp4
#   .\tasks.ps1 morphology -Image frame.png
param(
    [Parameter(Position = 0)]
    [ValidateSet("help", "test", "synthetic", "demo", "video", "benchmark", "compare", "morphology", "schema", "robustness", "feature_errors", "plots")]
    [string]$Task = "help",

    [string]$Image,
    [string]$Source,
    [ValidateSet("classical", "cellpose_cpsam", "cellpose_cyto3", "omnipose_bact_phase", "stardist_fluo", "sam2_auto")]
    [string]$Model = "classical",
    [double]$MinArea = 20,
    [ValidateSet("auto", "dark", "bright")]
    [string]$Objects = "auto",
    [string]$Out = "results/morphology.csv",
    [string]$OutDir,
    [double]$Fps = 0,
    [double]$PixelSizeUm = 0,
    [int]$Stride = 0,
    [int]$MaxFrames = 0
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
$Sam2 = Join-Path $PSScriptRoot "weights\models\sam2.1_hiera_small.pt"
if (Test-Path $Sam2) { $env:SAM2_CHECKPOINT = $Sam2 }

function Invoke-Py {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Video([string]$Src, [string]$Dir) {
    $a = @("scripts/run_video.py", $Src, "--model", $Model)
    if ($Dir) { $a += @("--out", $Dir) }
    if ($Fps -gt 0) { $a += @("--fps", $Fps) }
    if ($PixelSizeUm -gt 0) { $a += @("--pixel-size-um", $PixelSizeUm) }
    if ($Stride -gt 0) { $a += @("--stride", $Stride) }
    if ($MaxFrames -gt 0) { $a += @("--max-frames", $MaxFrames) }
    Invoke-Py @a
}

switch ($Task) {
    "help" {
        @"
Цели (из корня проекта, окружение .venv подхватывается само):

  .\tasks.ps1 test
      все тесты (pytest)

  .\tasks.ps1 demo
      синтетическое видео → полный конвейер → results/demo/

  .\tasks.ps1 video -Source video.mp4
      видео, папка кадров или TIFF-стек → results/<имя>/:
      observations.csv/.parquet, tracks_summary.csv, overlay.mp4, preview.png, run.json
      -Model sam2_auto | cellpose_cpsam | omnipose_bact_phase | classical
      -Fps 5  -PixelSizeUm 0.1  -Stride 2  -MaxFrames 10  -OutDir results/run1

  .\tasks.ps1 morphology -Image frame.png
      один кадр без трекинга → results/morphology.csv, test.png, masks.png
      -Model ...  -MinArea 5  -Objects dark | bright | auto  -Out results/my.csv

  .\tasks.ps1 synthetic
      синтетический набор с эталоном → data/synthetic_demo

  .\tasks.ps1 benchmark
      классическая сегментация на синтетике → results/benchmark_*.csv

  .\tasks.ps1 compare
      готовые модели, 1 кадр сценария clean → results/quick/

  .\tasks.ps1 schema
      обновить docs/export_schema.md после изменения схемы таблицы

Размеры — в пикселях, пока не задана калибровка (-PixelSizeUm или
calibration.pixel_size_um в configs/default.yaml).
"@
    }
    "test" {
        Invoke-Py -m pytest -q
    }
    "synthetic" {
        Invoke-Py src/generate_synthetic.py --output data/synthetic_demo `
            --sequences 1 --frames 56 --width 256 --height 192 --objects 12 --seed 24
    }
    "demo" {
        Invoke-Py src/generate_synthetic.py --output data/synthetic_demo `
            --sequences 1 --frames 56 --width 256 --height 192 --objects 12 --seed 24
        Invoke-Video "data/synthetic_demo/videos/seq_000.mp4" ($(if ($OutDir) { $OutDir } else { "results/demo" }))
    }
    "video" {
        if (-not $Source) { throw "Нужен источник: .\tasks.ps1 video -Source video.mp4" }
        Invoke-Video $Source $OutDir
    }
    "benchmark" {
        Invoke-Py scripts/benchmark.py --images 10 --seed 1
    }
    "compare" {
        Invoke-Py scripts/compare_models.py --scenarios clean --images 1 --seed 1 --out results/quick
    }
    "morphology" {
        if (-not $Image) { throw "Нужен кадр: .\tasks.ps1 morphology -Image frame.png" }
        Invoke-Py scripts/measure_frame.py $Image --model $Model --min-area $MinArea --objects $Objects --out $Out
    }
    "schema" {
        Invoke-Py scripts/gen_schema_doc.py
    }
    "robustness" {
        Invoke-Py scripts/robustness.py --strides 1 2 3 --sigmas 0 8 16 --out results/robust
    }
    "feature_errors" {
        Invoke-Py scripts/feature_errors.py --table results/demo/observations.csv `
            --gt data/synthetic_demo/objects.csv --out results/demo
    }
    "plots" {
        Invoke-Py scripts/plot_sizes.py --table results/demo/observations.csv `
            --gt data/synthetic_demo/objects.csv --out results/demo/plots
    }
}
