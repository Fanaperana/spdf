# One-time setup for the spdf benchmark harness (Windows / PowerShell 7).
#   pwsh bench/setup.ps1
# Creates bench/.venv, installs pinned deps (incl. LiteParse), fetches the
# pinned opendataloader-bench checkout into bench/.cache, generates the edge
# corpus and builds spdf in release mode.
$ErrorActionPreference = 'Stop'
$bench = $PSScriptRoot
$repo = Split-Path $bench
$venv = Join-Path $bench '.venv'
$py = Join-Path $venv 'Scripts\python.exe'
$odlCommit = '7af1d8f4d0c09f51ea1a5c6ba5f66e993286d109'

if (-not (Test-Path $py)) {
    if (Get-Command uv -ErrorAction SilentlyContinue) { uv venv $venv --python 3.12 } else { python -m venv $venv }
}
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv pip install --python $py -r (Join-Path $bench 'requirements.txt')
} else {
    & $py -m pip install -r (Join-Path $bench 'requirements.txt')
}

$odl = Join-Path $bench '.cache\opendataloader-bench'
if (-not (Test-Path (Join-Path $odl '.git'))) {
    New-Item -ItemType Directory -Force (Split-Path $odl) | Out-Null
    git clone https://github.com/opendataloader-project/opendataloader-bench.git $odl
}
git -C $odl fetch --depth 1 origin $odlCommit 2>$null
git -C $odl checkout --quiet $odlCommit

& $py (Join-Path $bench 'edge\make_corpus.py')
cargo build --release -p spdf-cli --manifest-path (Join-Path $repo 'Cargo.toml')
Write-Host "setup done. Run: $py bench\run_bench.py"
