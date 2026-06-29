# Summarize result CSV files.
# Run from project root:
#   .\scripts\summarize.ps1
#
# Default behavior:
#   - Summarizes files matching $Pattern in $ResultsDir
#   - Prints compact table
#   - DOES NOT save unless $SaveSummary = $true
#
# To summarize one specific file, set:
#   $ResultsFile = "results/unconstrained_1to1_beta1_2.csv"
#
# To summarize files by pattern, set:
#   $ResultsFile = ""
#   $Pattern = "unconstrained_1to1_beta1*.csv".\scripts\summarize.ps1
#
# To save with incrementing title, set:
#   $SaveSummary = $true
#   $OutPrefix = "unconstrained_1to1_beta1_summary"

$ResultsDir = "results"
$ResultsFile = ""   # Example: "results/unconstrained_1to1_beta1_2.csv"
# $Pattern = "unconstrained_1tok_beta1*.csv"
$Pattern = "constrained_1to1_*_profile*.csv"
# $Pattern = "unconstrained_1to1_*_profile*.csv"
$TrueTau = 1

$SaveSummary = $false
$OutDir = "summaries"
$OutPrefix = "unconstrained_1to1_beta1_summary"

if ($ResultsFile -ne "") {
    if ($SaveSummary) {
        python summarize_results.py `
            --results-file $ResultsFile `
            --true-tau $TrueTau `
            --save `
            --out-dir $OutDir `
            --out-prefix $OutPrefix
    }
    else {
        python summarize_results.py `
            --results-file $ResultsFile `
            --true-tau $TrueTau
    }
}
else {
    if ($SaveSummary) {
        python summarize_results.py `
            --results-dir $ResultsDir `
            --pattern $Pattern `
            --true-tau $TrueTau `
            --save `
            --out-dir $OutDir `
            --out-prefix $OutPrefix
    }
    else {
        python summarize_results.py `
            --results-dir $ResultsDir `
            --pattern $Pattern `
            --true-tau $TrueTau
    }
}
