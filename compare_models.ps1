$tests = @(
    @{ AsOf="20250613"; Start="20250616"; End="20250619" },
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
New-Item -ItemType Directory -Force experiments | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out_csv = "experiments/tuning_vs_baseline_ic_portfolio_$timestamp.csv"

$results = @()

Write-Host "=== IC + PORTFOLIO BACKTEST COMPARISON ==="
Write-Host ""

foreach ($test in $tests) {
    $date = $test.AsOf
    $start = $test.Start
    $end = $test.End

    Write-Host "Testing date: $date | portfolio window: $start to $end"

    $baseline_file = "submissions/baseline_$date.csv"
    $tuned_file    = "submissions/tuned_$date.csv"

    $baseline_output = python baseline_xgboost.py --as-of $date --out $baseline_file 2>&1
    $tuned_output    = python param_tuned_xgboost.py --as-of $date --out $tuned_file 2>&1

    $baseline_ic = $baseline_output | Select-String "validation rank IC:" | ForEach-Object {
        if ($_ -match '(-?\d+\.\d+)') { $matches[1] }
    } | Select-Object -First 1

    $tuned_ic = $tuned_output | Select-String "validation rank IC:" | ForEach-Object {
        if ($_ -match '(-?\d+\.\d+)') { $matches[1] }
    } | Select-Object -First 1

    if (!($baseline_ic -and $tuned_ic)) {
        Write-Host "  IC: ERROR parsing model output"
        Write-Host ""
        continue
    }

    $ic_diff = [double]$tuned_ic - [double]$baseline_ic
    $ic_winner = if ($ic_diff -gt 0) { "TUNED" } else { "BASELINE" }
    $ic_diff_fmt = "{0:+0.0000;-0.0000;0.0000}" -f $ic_diff

    Write-Host "  IC: Baseline $baseline_ic | Tuned $tuned_ic | Diff $ic_diff_fmt $ic_winner"

    if (!(Test-Path $baseline_file) -or !(Test-Path $tuned_file)) {
        Write-Host "  Portfolio: ERROR file not created"
        Write-Host ""
        continue
    }

    $baseline_score_output = python score_submission.py $baseline_file --start $start --end $end 2>&1
    $tuned_score_output    = python score_submission.py $tuned_file --start $start --end $end 2>&1

    function ExtractPct($output, $label) {
        $value = $output | Select-String $label | ForEach-Object {
            if ($_ -match '([+-]?\d+\.\d+)%') { $matches[1] }
        } | Select-Object -First 1
        return $value
    }

    $baseline_portfolio = ExtractPct $baseline_score_output "portfolio return"
    $baseline_benchmark = ExtractPct $baseline_score_output "benchmark return"
    $baseline_excess    = ExtractPct $baseline_score_output "excess return"

    $tuned_portfolio = ExtractPct $tuned_score_output "portfolio return"
    $tuned_benchmark = ExtractPct $tuned_score_output "benchmark return"
    $tuned_excess    = ExtractPct $tuned_score_output "excess return"

    if ($baseline_excess -and $tuned_excess) {
        $port_diff = [double]$tuned_excess - [double]$baseline_excess
        $port_winner = if ($port_diff -gt 0) { "TUNED" } else { "BASELINE" }
        $port_diff_fmt = "{0:+0.000;-0.000;0.000}" -f $port_diff

        Write-Host "  Portfolio excess: Baseline $baseline_excess% | Tuned $tuned_excess% | Diff $port_diff_fmt% $port_winner"

        $results += [PSCustomObject]@{
            as_of = $date
            score_start = $start
            score_end = $end

            baseline_ic = [double]$baseline_ic
            tuned_ic = [double]$tuned_ic
            ic_diff = $ic_diff
            ic_winner = $ic_winner

            baseline_portfolio_return_pct = [double]$baseline_portfolio
            baseline_benchmark_return_pct = [double]$baseline_benchmark
            baseline_excess_return_pct = [double]$baseline_excess

            tuned_portfolio_return_pct = [double]$tuned_portfolio
            tuned_benchmark_return_pct = [double]$tuned_benchmark
            tuned_excess_return_pct = [double]$tuned_excess

            excess_diff_pct = $port_diff
            portfolio_winner = $port_winner
        }

        $results | Export-Csv $out_csv -NoTypeInformation
    } else {
        Write-Host "  Portfolio: ERROR parsing score output"
    }

    Write-Host ""
}

Write-Host "=== SUMMARY ==="
$results | Format-Table -AutoSize

$ic_tuned_wins = ($results | Where-Object { $_.ic_winner -eq "TUNED" }).Count
$port_tuned_wins = ($results | Where-Object { $_.portfolio_winner -eq "TUNED" }).Count

Write-Host ""
Write-Host "IC tuned wins: $ic_tuned_wins / $($results.Count)"
Write-Host "Portfolio tuned wins: $port_tuned_wins / $($results.Count)"
Write-Host "Saved results to: $out_csv"