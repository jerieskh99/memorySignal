$Root = (Get-Location).Path
$config = Get-Content -Raw -Path (Join-Path $Root "config.json") | ConvertFrom-Json
$rustDeltaCalculationProgram = $config.rustDeltaCalculationProgram

$qPath = $config.queueDir
$pending = Join-Path $qPath "pending"
$processing = Join-Path $qPath "processing"
$done = Join-Path $qPath "done"
$failed = Join-Path $qPath "failed"

Write-Host "[CONSUMER] Consumer started"
Write-Host "[CONSUMER] Queue dir: $qPath"
Write-Host "[CONSUMER] Watching: $pending"
Write-Host "[CONSUMER] Rust program: $rustDeltaCalculationProgram"

New-Item -ItemType Directory -Force -Path $pending, $processing, $done, $failed | Out-Null

while ($true) {
    $jobFile = Get-ChildItem -Path $pending -Filter *.json | Sort-Object Name | Select-Object -First 1

    if (-not $jobFile) {
        Start-Sleep -Milliseconds 100
        continue
    }

    Write-Host "[CONSUMER] Picked job: $($jobFile.Name)"

    $jobPathProc = Join-Path $processing $jobFile.Name
    Move-Item -Force $jobFile.FullName $jobPathProc

    try {
        $jobContent = Get-Content -Raw -Path $jobPathProc | ConvertFrom-Json
        
        Write-Host "[CONSUMER] Running rust: prev=$(Split-Path $jobContent.prev -Leaf) curr=$(Split-Path $jobContent.curr -Leaf) out=$(Split-Path $jobContent.output -Leaf)"
        & $rustDeltaCalculationProgram $jobContent.prev $jobContent.curr $jobContent.output
        $rc = $LASTEXITCODE

        if ($rc -ne 0) {
            throw "Rust program failed with exit code $rc"
        }

        Write-Host "[CONSUMER] Rust finished OK (rc=0)"

        # Cleanup + finalize
        Remove-Item $jobContent.prev -ErrorAction SilentlyContinue
        Move-Item -Force $jobPathProc (Join-Path $done $jobFile.Name)
        Write-Host "[CONSUMER] Job done -> done: $($jobFile.Name)"
    }
    catch{
        Write-Host "[CONSUMER] ERROR job=$($jobFile.Name): $($_.Exception.Message)"
        Move-Item $jobPathProc (Join-Path $failed $jobFile.name)
        Write-Host "[CONSUMER] Job moved -> failed: $($jobFile.Name)"
    }
}