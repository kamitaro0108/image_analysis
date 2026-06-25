param(
    [int]$Port = 8888
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Error ".venv was not found. Run: python -m venv .venv"
}

Set-Location $Root
& $Python -m notebook --notebook-dir $Root --port $Port
