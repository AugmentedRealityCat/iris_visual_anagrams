param([string]$PythonVersion = '3.12')

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv first: python -m pip install uv'
}
Invoke-Checked -Program uv -Arguments @('venv', '--python', $PythonVersion, '.venv', '--allow-existing')
$Python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
Invoke-Checked -Program uv -Arguments @('pip', 'install', '--python', $Python, 'torch==2.11.0', 'torchvision==0.26.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
Invoke-Checked -Program uv -Arguments @('pip', 'install', '--python', $Python, '-r', 'requirements-iris.txt')
# The upstream package declares Linux training dependencies. Inference only
# needs the dependencies above, so install its pinned code without those extras.
Invoke-Checked -Program uv -Arguments @('pip', 'install', '--python', $Python, '--no-deps', 'iris-3b @ git+https://github.com/speridlabs/iris-3b.git@a8d15239dea469aba042cfa56ca3bb4e450d5ebc')
Invoke-Checked -Program uv -Arguments @('pip', 'install', '--python', $Python, '--no-deps', '-e', '.')
Write-Host 'Ready. Run .\.venv\Scripts\python.exe generate.py --help'
