# Delete non-diagnostic generated CSV results and summaries.
# Be careful: this removes output files.
# The diagnostics tree is intentionally excluded.
# Run from project root:
#   .\scripts\clean_outputs.ps1

$NonDiagnosticResultRoots = @(
    "results\matching",
    "results\experiments"
)

foreach ($Root in $NonDiagnosticResultRoots) {
    if (Test-Path -LiteralPath $Root) {
        Get-ChildItem -LiteralPath $Root -Recurse -File -Filter *.csv |
            Remove-Item -Force
    }
}

if (Test-Path summaries) {
    Remove-Item summaries\*.csv
}

Write-Host "Removed CSV files from results/matching, results/experiments, and summaries."
