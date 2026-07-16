param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$IncludeDir = Join-Path $ProjectRoot "include"
$SourceDir = Join-Path $ProjectRoot "src"
$BinDir = Join-Path $ProjectRoot "bin"

New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

$Compiler = $null
foreach ($Name in @("gcc", "clang")) {
    $Command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -ne $Command) {
        $Compiler = $Command.Source
        break
    }
}

if ($null -eq $Compiler) {
    $WinGetPackages = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path -LiteralPath $WinGetPackages) {
        $WinLibsRoots = Get-ChildItem -LiteralPath $WinGetPackages -Directory -ErrorAction SilentlyContinue |
            Where-Object Name -Like "BrechtSanders.WinLibs.POSIX.UCRT*"
        foreach ($Root in $WinLibsRoots) {
            $Candidate = Join-Path $Root.FullName "mingw64\bin\gcc.exe"
            if (Test-Path -LiteralPath $Candidate) {
                $Compiler = $Candidate
                break
            }
        }
    }
}

if ($null -eq $Compiler) {
    foreach ($Candidate in @(
        "C:\mingw64\bin\gcc.exe",
        "C:\msys64\ucrt64\bin\gcc.exe",
        "C:\msys64\mingw64\bin\gcc.exe",
        "C:\Program Files\LLVM\bin\clang.exe"
    )) {
        if (Test-Path -LiteralPath $Candidate) {
            $Compiler = $Candidate
            break
        }
    }
}

if ($null -eq $Compiler) {
    throw "No C compiler was found. Install GCC/MinGW-w64 or a complete LLVM/MSVC toolchain."
}

$Common = @(
    "-std=c17",
    "-D_CRT_SECURE_NO_WARNINGS",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-I$IncludeDir"
)

if ($Configuration -eq "Release") {
    $Common += @("-O3", "-DNDEBUG", "-march=native")
}
else {
    $Common += @("-O0", "-g")
}

$CompilerName = [System.IO.Path]::GetFileName($Compiler).ToLowerInvariant()
$LinkArgs = @()
if ($CompilerName -like "gcc*") {
    $LinkArgs += @("-lm", "-static-libgcc")
}

function Build-Executable {
    param(
        [string]$Name,
        [string[]]$Sources
    )

    $Output = Join-Path $BinDir "$Name.exe"
    $Arguments = @($Common) + $Sources + @("-o", $Output) + $LinkArgs
    Write-Host "Building $Name ($Configuration) with $Compiler"
    & $Compiler @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Compilation failed for $Name (exit code $LASTEXITCODE)."
    }
}

Build-Executable "generate_datasets" @(
    (Join-Path $SourceDir "cm_dataset.c"),
    (Join-Path $SourceDir "cm_rng.c"),
    (Join-Path $SourceDir "generate_datasets.c")
)

Build-Executable "run_matching_from_datasets" @(
    (Join-Path $SourceDir "cm_dataset.c"),
    (Join-Path $SourceDir "cm_rng.c"),
    (Join-Path $SourceDir "cm_timer.c"),
    (Join-Path $SourceDir "cm_matching.c"),
    (Join-Path $SourceDir "run_matching_from_datasets.c")
)

Build-Executable "summarize_results" @(
    (Join-Path $SourceDir "summarize_results.c")
)

Write-Host "Built executables in $BinDir"
