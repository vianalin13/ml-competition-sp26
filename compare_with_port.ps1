$tests = @(
    @{ AsOf="20250630"; Start="20250701"; End="20250704" },
    @{ AsOf="20250728"; Start="20250729"; End="20250801" },
    @{ AsOf="20250818"; Start="20250819"; End="20250822" },
    @{ AsOf="20250922"; Start="20250923"; End="20250926" },
    @{ AsOf="20251010"; Start="20251013"; End="20251017" },
    @{ AsOf="20251111"; Start="20251112"; End="20251118" },
    @{ AsOf="20251218"; Start="20251219"; End="20251224" },
    @{ AsOf="20260126"; Start="20260127"; End="20260130" },
    @{ AsOf="20260202"; Start="20260203"; End="20260206" },
    @{ AsOf="20260309"; Start="20260310"; End="20260313" },
    @{ AsOf="20260417"; Start="20260420"; End="20260424" }
)

New-Item -ItemType Directory -Force submissions | Out-Null

Write-Host "=== PORTFOLIO BACKTEST COMPARISON ==="
Write-Host ""

foreach ($test in $tests) {
    $date = $test.AsOf
    $start = $test.Start
    $end = $test.End

    Write-Host "Testing as-of: $date | score window: $start to $end"

    $baseline_file = "submissions/baseline_$date.csv"
    $tuned_file    = "submissions/tuned_$date.csv"

    python baseline_xgboost.py --as-of $date --out $baseline_file | Out-Null
    python param_tuned_xgboost.py --as-of $date --out $tuned_file | Out-Null

    if (!(Test-Path $baseline_file) -or !(Test-Path $tuned_file)) {
        Write-Host "  ERROR: portfolio file not created"
        Write-Host ""
        continue
    }

    $baseline_score_output = python score_submission.py $baseline_file --start $start --end $end 2>&1
    $tuned_score_output    = python score_submission.py $tuned_file --start $start --end $end 2>&1

    Write-Host "  Baseline raw:"
    $baseline_score_output
    Write-Host "  Tuned raw:"
    $tuned_score_output
    Write-Host ""
}