param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Streamlit = Join-Path $Root ".venv\Scripts\streamlit.exe"

if (-not (Test-Path $Streamlit)) {
    Write-Error "Streamlit was not found. Run: .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

Set-Location $Root
& $Streamlit run app.py --server.port $Port
