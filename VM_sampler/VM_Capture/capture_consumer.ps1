$pending = "C:\thesis\queue\pending"
$processing = "C:\thesis\queue\processing"
$done = "C:\thesis\queue\done"
$failed = "C:\thesis\queue\failed"

New-Item -ItemType Directory -Force -Path $pending, $processing, $done, $failed | Out-Null

while ($true) {
    $jobFile = Get-ChildItem -Path $pending -Filter *.json | Sort-Object Name | Select-Object -First 1

    if (-not $jobFile) {
        Start-Sleep -Milliseconds 100
        continue
    }

    $jobPathProc = Join-Path $processing $jobFile.Name
    Move-Item $jobFile.FullName $jobPathProc

    try {
        $jobContent = Get-Content -Raw -Path $jobPathProc | ConvertFrom-Json

        & $rustDeltaCalculationProgram $jobContent.prev $jobContent.curr $jobContent.output
        $rc = $LASTEXITCODE

        if ($rc -ne 0) {
            throw "Rust program failed with exit code $rc"
        }

        Remove-Item $jobContent.prev -ErrorAction SilentlyContinue
        Move-Item $jobPathProc (Join-Path $done $jobFile.Name)
    }
    catch{
        Write-Host "Error processing job $($jobFile.Name): $_"
        Move-Item $jobPathProc (Join-Path $failed $jobFile.name)
    }
}