param(
    [string]$Datasets = "datasets_sb01.npz",
    [int]$NMcmc = 1000,
    [double]$Beta = 1,
    [int]$K = 2,
    [double]$Pstar = 0.5,
    [int]$MaxSim = 100,
    [string]$ResultsDir = "results_c",
    [string]$SummariesDir = "summaries_c",
    [string]$RunId = "",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $ProjectRoot
$Runner = Join-Path $ProjectRoot "bin\run_matching_from_datasets.exe"
$Summarizer = Join-Path $ProjectRoot "bin\summarize_results.exe"

if (-not $SkipBuild) {
    & (Join-Path $PSScriptRoot "build.ps1") -Configuration Release
    if ($LASTEXITCODE -ne 0) {
        throw "Release build failed."
    }
}

if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Runner not found at $Runner. Build the project first."
}
if (-not (Test-Path -LiteralPath $Summarizer)) {
    throw "Summarizer not found at $Summarizer. Build the project first."
}

$Methods = @(
    "unconstrained_1to1",
    "unconstrained_1tok",
    "unconstrained_1tok_mc",
    "constrained_1to1",
    "constrained_1to1_gibbs",
    "constrained_1tok"
)

$BetaLabel = ([string]$Beta).Replace("-", "m").Replace(".", "p")
$EffectiveRunId = $RunId
if ([string]::IsNullOrWhiteSpace($EffectiveRunId)) {
    $EffectiveRunId = "run_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss_fff")
}
$RunResultsDir = Join-Path $ResultsDir $EffectiveRunId
$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

Push-Location $RepositoryRoot
try {
    foreach ($Method in $Methods) {
        $Prefix = "{0}_beta{1}_profile" -f $Method, $BetaLabel
        Write-Host ""
        Write-Host "=== $Method ==="
        $RunArguments = @(
            "--datasets", $Datasets,
            "--method", $Method,
            "--n-mcmc", $NMcmc,
            "--beta", $Beta,
            "--k", $K,
            "--pstar", $Pstar,
            "--max-sim", $MaxSim,
            "--out-dir", $RunResultsDir,
            "--out-prefix", $Prefix,
            "--profile"
        )
        & $Runner @RunArguments
        if ($LASTEXITCODE -ne 0) {
            throw "$Method failed with exit code $LASTEXITCODE."
        }
    }

    Write-Host ""
    Write-Host "=== Combined C summary ==="
    $SummaryPattern = "*_beta{0}_profile_*.csv" -f $BetaLabel
    $SummaryPrefix = "all_methods_beta{0}_summary" -f $BetaLabel
    $SummaryArguments = @(
        "--results-dir", $RunResultsDir,
        "--pattern", $SummaryPattern,
        "--true-tau", 1,
        "--save",
        "--out-dir", $SummariesDir,
        "--out-prefix", ("{0}_{1}" -f $SummaryPrefix, $EffectiveRunId)
    )
    & $Summarizer @SummaryArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Summarization failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
    $Stopwatch.Stop()
}

Write-Host ""
Write-Host ("All six methods finished in {0:N3} seconds." -f $Stopwatch.Elapsed.TotalSeconds)
Write-Host "Run results: $RunResultsDir"
