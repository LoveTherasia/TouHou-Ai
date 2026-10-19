# ============================================================
#  One-click entry: looping automation pipeline
#    real+mock (separate process) -> merge -> continue -> merge ... loop
#
#  Usage (from project root):
#    powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
#    powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Rounds 3 -Difficulty 2
#    powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -NoReal
#
#  Stop: press Ctrl-C (policy_final is saved on exit).
# ============================================================
param(
    [int]$Rounds = 0,
    [double]$RealSeconds = 600,
    [double]$MergeEvery = 120,
    [int]$Difficulty = 1,
    [string]$Out = ".runtime\auto_pipeline",
    [string]$Load = "",
    [string]$DanmakuDir = ".runtime\danmaku_real",
    [switch]$NoReal
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$cliArgs = @(
    "tools\auto_pipeline.py",
    "--rounds", "$Rounds",
    "--real-seconds", "$RealSeconds",
    "--merge-every", "$MergeEvery",
    "--difficulty", "$Difficulty",
    "--out", $Out,
    "--danmaku-dir", $DanmakuDir
)
if ($Load -ne "") { $cliArgs += @("--load", $Load) }
if ($NoReal) { $cliArgs += @("--no-real") }

Write-Host "== looping pipeline ==" -ForegroundColor Cyan
Write-Host ("  rounds={0} (0=inf)  real={1}s  merge_every={2}s  difficulty={3}" -f $Rounds, $RealSeconds, $MergeEvery, $Difficulty) -ForegroundColor Gray
Write-Host ("  out={0}   Ctrl-C to stop" -f $Out) -ForegroundColor Gray

Push-Location $root
try {
    & $py @cliArgs
} finally {
    Pop-Location
}
