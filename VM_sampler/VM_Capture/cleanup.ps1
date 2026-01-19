param (
    [Parameter(Mandatory=$true)]
    [String] $Root,

    [switch] $DryRun
)

Write-Host "[CLEANUP] Starting cleanup"
Write-Host "[CLEANUP] Root: $Root"

if ($DryRun) {
    Write-Host "[CLEANUP] Dry-run mode (no files will be deleted)"
}

$ConfigPath = Join-Path $Root "config.json"
if (-not (Test-Path $ConfigPath)){
    throw "[CLEANUP] config.json not found under root"
}

$config     = Get-Content -Raw -Path $ConfigPath | ConvertFrom-Json
$queuePath  = @($config.queueDir)[0]
$outputPath = @($config.outputDir)[0]
$dumpPath   = @($config.imageDir)[0]

Write-Host "[CLEANUP] queuePath  : $queuePath"
Write-Host "[CLEANUP] outputPath : $outputPath"
Write-Host "[CLEANUP] dumpPath   : $dumpPath"

$queueTargets = @(
    (Join-Path $queuePath "pending"),
    (Join-Path $queuePath "processing"),
    (Join-Path $queuePath "done"),
    (Join-Path $queuePath "failed")
)

$outputTargets = @(
    (Join-Path $outputPath "cosine"),
    (Join-Path $outputPath "hamming")
)

# Clean queues:
Write-Host "[CLEANUP] Queues: Clean-up started:"
foreach ($dir in $queueTargets) {
    if (-not (Test-Path $dir)) {
        Write-Host "[CLEANUP] Skipping missing dir: $dir"
        continue
    }

    $files = Get-ChildItem -path $dir -File -ErrorAction SilentlyContinue
    if (-not $files) {
        Write-Host "[CLEANUP] Empty: $dir"
        continue
    }

    Write-Host "[CLEANUP] Cleaning $($files.Count) files in $(Split-Path $dir -Leaf)"

    foreach ($f in $files) {
        if ($DryRun) {
            Write-Host "  would delete $($f.FullName)"
        } else {
            Remove-Item -Force -Path $f.FullName
        }
    }
}
Write-Host "[CLEANUP] Queues DONE clean-up"

Write-Host "[CLEANUP] Output: Clean-up started:"
foreach ($dir in $outputTargets) {
    if (-not (Test-Path $dir)) {
        Write-Host "[CLEANUP] Skipping missing dir: $dir"
        continue
    }

    $files = Get-ChildItem -path $dir -File -ErrorAction SilentlyContinue
    if (-not $files) {
        Write-Host "[CLEANUP] Empty: $dir"
        continue
    }

    Write-Host "[CLEANUP] Cleaning $($files.Count) files in $(Split-Path $dir -Leaf)"

    foreach ($f in $files) {
        if ($DryRun) {
            Write-Host "  would delete $($f.FullName)"
        } else {
            Remove-Item -Force -Path $f.FullName
        }
    }
}
Write-Host "[CLEANUP] Output DONE clean-up"

Write-Host "[CLEANUP] Dumps: Clean-up started:"
if (-not (Test-Path $dumpPath)) {
        Write-Host "[CLEANUP] Skipping missing dir: $dumpPath"
        return
}

$dumps = Get-ChildItem -path $dumpPath -File -ErrorAction SilentlyContinue
if (-not $dumps) {
    Write-Host "[CLEANUP] Empty: $dumpPath"
    return
}

foreach ($f in $dumps) {
    if ($DryRun) {
        Write-Host "  would delete $($f.FullName)"
    } else {
        Remove-Item -Force -Path $f.FullName
    }
}
Write-Host "[CLEANUP] Dumps DONE clean-up"

