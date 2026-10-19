# 构建 32 位时间缩放钩子 (speedhack32.dll) + 注入器 (injector32.exe) + 自检 (timer_test32.exe)
# 需要 Visual Studio (MSVC) 的 32 位工具链。产物落在 native\speedhack\bin\。
$ErrorActionPreference = 'Stop'

$src = $PSScriptRoot
$out = Join-Path $src 'bin'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { Write-Error '找不到 vswhere.exe (需要 Visual Studio)'; exit 1 }
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { Write-Error '找不到带 C++ 工具链的 Visual Studio'; exit 1 }

$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars32.bat'
$flags = '/nologo /utf-8 /O2 /W3'

$steps = @(
    "cl $flags /LD `"$src\speedhack.c`" /Fe:`"$out\speedhack32.dll`" /Fo:`"$out\speedhack.obj`" /link /DLL",
    "cl $flags `"$src\injector.c`" /Fe:`"$out\injector32.exe`" /Fo:`"$out\injector.obj`"",
    "cl $flags `"$src\timer_test.c`" /Fe:`"$out\timer_test32.exe`" /Fo:`"$out\timer_test.obj`" winmm.lib"
)

foreach ($step in $steps) {
    Write-Host "[build] $step"
    cmd /c "call `"$vcvars`" >nul && $step"
    if ($LASTEXITCODE -ne 0) { Write-Error '[build] FAILED'; exit 1 }
}
Write-Host "[build] OK -> $out"
