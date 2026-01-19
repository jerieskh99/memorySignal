$vmName = $config.vmName
$imageDir = $config.imageDir
$rustDeltaCalculationProgram = $config.rustDeltaCalculationProgram
$pauseDuration = $config.pauseDuration
$longPauseDuration = $config.longPauseDuration
$resumeDuration = $config.resumeDuration
$intervalMinutes = $config.intervalMinutes  # Convert minutes to seconds for Start-Sleep
$outputDir = $config.outputDir  # New variable for output directory

$queuePending = "C:\thesis\queue\pending"

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
            $currState = $vmStateLine.Tostring().Split('=')[1].Trim('"')
            if ($currState -eq $desiredState) {
                return $true
            }
        }
    }

    Start-Sleep -Milliseconds $pollIntervalMs
}

New-item -ItemType Directory -Force -Path $queuePending | Out-Null

Write-Host "Starting memory dump and delta calculation process..."

while ($true) {
    try {
        Write-Host "Pausing VM..."
        VBoxManage controlvm $vmName pause
        Wait-VMState - vmName $vmName -desiredState "paused" -timeoutSeconds 30 -pollIntervalMs 200

        # Step 2: Capture VM's memory
        $timestamp = Get-Date -Format "yyyyMMddHHmmss"
        $newImage = "$imageDir\$imageFilePrefix-$timestamp.elf"

        Write-Host "Dumping memory to $newImage ..."
        VBoxMnage debugvm $vmName dumpvmcore $newImage

        if (Test-Path = $newImage) {
            Write-Host "Memory dump successful: $newImage"
        } 
        else {
            Write-Host "Memory dump failed."
        }

        if ($prevImage -ne "") {
            $jobId = (Get-Date -Format "yyyyMMddHHmmss")

            $tmp = Join-Path $queuePending "$jobId.json.tmp"
            $job = Join-Path $queuePending "$jobId.json"
            
            @{
                prev = $prevImage
                curr = $newImage
                output = $outputDir
            } | ConvertTo-Json | Set-Content -Path $tmp -Encoding UTF8
        }
        
        $prevImage = $newImage
        
        Write-Host "Resuming VM..."
        VBoxManage controlvm $vmName resume
        Wait-VMState - vmName $vmName -desiredState "resumed" -timeoutSeconds 30 -pollIntervalMs 200


        # Backpressure:
        $pendingJobsCount = (Get-ChildItem -Path $queuePending -Filter *.json).count
        $proccessingJobsCount (Get-ChildItem -Path $processing -Filter *.json).count

        if ($pendingJobsCount + $proccessingJobsCount -gt 10) {
            Write-Host "Backpressure detected. Pausing for longer duration..."
            Start-Sleep -Seconds 1 
            continue
        }
    }
    catch {
        Write-Host "An error occurred: $_"
    }
}
