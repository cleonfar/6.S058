Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot\..
try {
    $python = ".\.venv\Scripts\python.exe"

    & $python -m gtauav_loc.train_baseline `
        --split dataset/same-area-drone2sate-train.json `
        --dataset-root dataset `
        --precompute-dir data/precompute_res18_sift `
        --out-dir runs/baseline_res18 `
        --use-gpu `
        --warp-backend torch `
        --epochs 10 `
        --batch-size 16 `
        --hard-negative-top-k 128 `
        --window-size 8 `
        --stride 1

    & $python -m gtauav_loc.train_sequential `
        --split dataset/same-area-drone2sate-train.json `
        --dataset-root dataset `
        --precompute-dir data/precompute_res18_sift `
        --out-dir runs/sequential_noconf `
        --use-gpu `
        --warp-backend torch `
        --epochs 10 `
        --batch-size 16 `
        --hard-negative-top-k 128 `
        --window-size 8 `
        --stride 1 `
        --use-location-feedback `
        --teacher-forcing-start-prob 1.0 `
        --teacher-forcing-end-prob 0.25

    & $python -m gtauav_loc.train_sequential `
        --split dataset/same-area-drone2sate-train.json `
        --dataset-root dataset `
        --precompute-dir data/precompute_res18_sift `
        --out-dir runs/sequential_conf `
        --use-gpu `
        --warp-backend torch `
        --epochs 10 `
        --batch-size 16 `
        --hard-negative-top-k 128 `
        --window-size 8 `
        --stride 1 `
        --use-location-feedback `
        --teacher-forcing-start-prob 1.0 `
        --teacher-forcing-end-prob 0.25 `
        --with-confidence
}
finally {
    Pop-Location
}