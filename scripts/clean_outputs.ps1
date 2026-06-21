# Delete generated results and summaries.
# Be careful: this removes output files.
# Run from project root:
#   .\scripts\clean_outputs.ps1

if (Test-Path results) {
    Remove-Item results\*.csv
}

if (Test-Path summaries) {
    Remove-Item summaries\*.csv
}

Write-Host "Removed CSV files from results/ and summaries/."
