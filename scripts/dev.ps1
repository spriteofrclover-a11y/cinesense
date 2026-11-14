# Thin wrapper so nobody has to remember which interpreter has pandas.
#
#   .\scripts\dev.ps1 build-catalog
#   .\scripts\dev.ps1 build-catalog --dataset ml-latest-small
#   .\scripts\dev.ps1 config
#
# The `python` on PATH is 3.11 and cannot import pandas. This always uses .venv.

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error @"
No virtual environment found at .venv

Set it up with:
    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
"@
    exit 1
}

Push-Location $root
try {
    # Python logs progress to stderr deliberately (it is a CLI). Windows
    # PowerShell 5.1 promotes native stderr to a terminating error while
    # $ErrorActionPreference is Stop, so relax it for the child process only
    # and judge success by the exit code instead.
    $ErrorActionPreference = "Continue"
    & $python -m cinesense @Arguments
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $code
