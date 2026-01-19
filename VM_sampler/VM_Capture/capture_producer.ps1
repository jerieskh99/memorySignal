$Root = (Get-Location).Path
$config = Get-Content -Raw -Path (Join-Path $Root "config.json") | ConvertFrom-Json
$vmName = $config.vmName
$imageDir = $config.imageDir
$outputDir = $config.outputDir  # New variable for output directory
$intervalMsec = $config.intervalMsec  # prefer seconds explicitly
$qPath = $config.queueDir

$imageFilePrefix = "memory_dump"
$prevImage = ""

$qPending = Join-Path $qPath "pending"
$qProcessing = Join-Path $qPath "processing"
New-Item -ItemType Directory -Force -Path $qPending, $qProcessing | Out-Null

function Wait-VMState {
    param (
        [string]$vmName,
        [string]$desiredState,
        [int]$timeoutSeconds = 30,
        [int]$pollIntervalMs = 200
    )

    $deadLine = (Get-Date).Addseconds($timeoutSeconds)

    while ((Get-Date) -lt $deadLine) {
        $vmStateLine = VBoxManage showvminfo $vmName --machinereadable | Select-String -Pattern '^VMState='

        if ($vmStateLine) {
            $currState = $vmStateLine.ToString().Split('=')[1].Trim('"')
            if ($currState -eq $desiredState) {
                return $true
            }
        }
        Start-Sleep -Milliseconds $pollIntervalMs
    }
    throw "Timeout waiting for VM '$vmName' to reach state '$desiredState'"
}

Write-Host "Starting producer..."

while ($true) {
    try {
        # Backpressure:
        $pendingJobsCount = (Get-ChildItem -Path $qPending -Filter *.json).Count
        $proccessingJobsCount = (Get-ChildItem -Path $qProcessing -Filter *.json).Count

        if ($pendingJobsCount + $proccessingJobsCount -gt 10) {
            Write-Host "Backpressure: queue too large. Sleeping..."
            Start-Sleep -Seconds 1 
            continue
        }

        Write-Host "Pausing VM..."
        VBoxManage controlvm $vmName pause
        Wait-VMState -vmName $vmName -desiredState "paused" -timeoutSeconds 30 -pollIntervalMs 200

        # Step 2: Capture VM's memory
        $timestamp = Get-Date -Format "yyyyMMddHHmmssfff"
        $newImage = Join-Path $imageDir "$imageFilePrefix-$timestamp.elf"

        Write-Host "Dumping memory to $newImage ..."
        VBoxManage debugvm $vmName dumpvmcore --filename $newImage

        if (Test-Path $newImage) {
            Write-Host "Memory dump successful: $newImage"
        } 
        else {
            throw "Memory dump failed."
        }

        if ($prevImage -ne "") {
            $jobId = $timestamp

            $tmp = Join-Path $qPending "$jobId.json.tmp"
            $job = Join-Path $qPending "$jobId.json"
            
            @{
                prev = $prevImage
                curr = $newImage
                output = $outputDir
            } | ConvertTo-Json | Set-Content -Path $tmp -Encoding UTF8

            Move-Item -Force $tmp $job
        }
        
        $prevImage = $newImage
        
        Write-Host "Resuming VM..."
        VBoxManage controlvm $vmName resume
        Wait-VMState -vmName $vmName -desiredState "running" -timeoutSeconds 30 -pollIntervalMs 200
        Start-Sleep -Milliseconds $intervalMsec
    }
    catch {
        Write-Host "An error occurred: $_"
        Start-Sleep -Milliseconds 500
    }
}
