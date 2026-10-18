# Creates the conda environment and installs all dependencies.
# Run AFTER Miniconda is installed (see README.md step 1).
#
# Usage (from this folder, in PowerShell):
#     powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"
$conda = "C:\Users\shane\miniconda3\Scripts\conda.exe"

if (-not (Test-Path $conda)) {
    Write-Host "conda.exe not found at $conda" -ForegroundColor Red
    Write-Host "Install Miniconda first (see README.md)." -ForegroundColor Red
    exit 1
}

Write-Host "Creating conda environment 'nike-rag' with Python 3.11 ..."
& $conda create -n nike-rag python=3.11 -y
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Use the environment's own Python directly (most reliable in scripts).
$env_py = "C:\Users\shane\miniconda3\envs\nike-rag\python.exe"

Write-Host "Upgrading pip ..."
& $env_py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Installing project dependencies (this downloads a lot on first run) ..."
& $env_py -m pip install pandas chromadb sentence-transformers streamlit openai
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "All done! Next steps:" -ForegroundColor Green
Write-Host "  1. conda activate nike-rag"
Write-Host "  2. python load_data.py"
Write-Host "  3. python embed.py"
Write-Host "  4. streamlit run app.py"
