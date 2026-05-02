#compare baseline vs param_tuned across multiple dates
#mix of recent dates (April) and earlier months (March, February)
$dates = @(
    # #recent april dates
    # "20260430",
    # "20260429",
    # "20260428",
    # "20260425",
    # "20260424",
    # "20260423",
    # "20260422",
    # "20260421",

    # #mid april
    # "20260418",
    # "20260417",

    # #march
    # "20260331",
    # "20260327",

    # #feb
    # "20260228",
    # "20260225"

    # "20250102", 
    # "20250116",
    # "20250207",
    # "20250224", 
    # "20250313",
    # "20250327",
    # "20250411",
    # "20250429",
    # "20250523", these dates aren't available? i thought data was downloaded from jan 2025
    "20250613",
    "20250630",  
    "20250728",  
    "20250818",
    "20250922",  
    "20251010",
    "20251111",
    "20251218",
    "20260126",  
    "20260202",
    "20260309",
    "20260425"
)

$results = @()

Write-Host "Running comparison tests..."
Write-Host ""

foreach ($date in $dates) {
    Write-Host "Testing date: $date"
    
    #run baseline
    $baseline_output = python baseline_xgboost.py --as-of $date 2>&1
    $baseline_ic = $baseline_output | Select-String "validation rank IC:" | ForEach-Object { 
        if ($_ -match '(\d+\.\d+)') { $matches[1] }
    } | Select-Object -First 1
    
    #run param_tuned
    $tuned_output = python param_tuned_xgboost.py --as-of $date 2>&1
    $tuned_ic = $tuned_output | Select-String "validation rank IC:" | ForEach-Object { 
        if ($_ -match '(\d+\.\d+)') { $matches[1] }
    } | Select-Object -First 1
    
    if ($baseline_ic -and $tuned_ic) {
        $diff = [double]$tuned_ic - [double]$baseline_ic
        $winner = if ($diff -gt 0) { "TUNED" } else { "BASELINE" }
        
        $results += [PSCustomObject]@{
            Date = $date
            Baseline = $baseline_ic
            Tuned = $tuned_ic
            Diff = "{0:+0.0000;-0.0000;0.0000}" -f $diff
            Winner = $winner
        }
        
        Write-Host "  Baseline: $baseline_ic | Tuned: $tuned_ic | Diff: $($results[-1].Diff) $winner"
    } else {
        Write-Host "  ERROR parsing output"
    }
    Write-Host ""
}

Write-Host "=== SUMMARY ==="
$results | Format-Table -AutoSize

$tuned_wins = ($results | Where-Object { $_.Winner -like "TUNED*" }).Count
$baseline_wins = ($results | Where-Object { $_.Winner -like "BASELINE*" }).Count

Write-Host ""
Write-Host "Tuned wins: $tuned_wins / $($results.Count)"
Write-Host "Baseline wins: $baseline_wins / $($results.Count)"
Write-Host ""

if ($tuned_wins -gt $baseline_wins) {
    Write-Host "TUNED PARAMS ARE BETTER on average"
} else {
    Write-Host "BASELINE IS BETTER - consider reverting or re-tuning"
}
