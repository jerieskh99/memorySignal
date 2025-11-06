# Load configuration from JSON file
$config = Get-Content -Raw -Path "C:\Users\jeries\Desktop\thesis\config.json" | ConvertFrom-Json

$vmName = $config.vmName
$imageDir = $config.imageDir
$rustDeltaCalculationProgram = $config.rustDeltaCalculationProgram
$pauseDuration = $config.pauseDuration
$longPauseDuration = $config.longPauseDuration
$resumeDuration = $config.resumeDuration
$intervalMinutes = $config.intervalMinutes  # Convert minutes to seconds for Start-Sleep
$outputDir = $config.outputDir  # New variable for output directory

$imageFilePrefix = "memory_dump"
$prevImage = ""
$newImage = ""

Write-Host "Starting memory dump and delta calculation process..."

# Infinite loop to capture memory dumps and process deltas
while ($true) {
    try {
        # Step 1: Pause the VM
        Write-Host "Pausing VM..."
        VBoxManage controlvm $vmName pause
        
        # Wait for the VM to fully pause
        Start-Sleep -Seconds $pauseDuration
        
        # Step 2: Capture VM's memory
        $timestamp = Get-Date -Format "yyyyMMddHHmmss"
        $newImage = "$imageDir\$imageFilePrefix-$timestamp.elf"
        
        Write-Host "Dumping VM's memory to file: $newImage"
        VBoxManage debugvm $vmName dumpvmcore --filename $newImage
        
        # Wait for the memory dump process to complete
        Start-Sleep -Seconds $resumeDuration
        
        # Check if the new image file was created successfully
        if (Test-Path $newImage) {
            Write-Host "New image file created successfully: $newImage"
        } else {
            Write-Host "Error: New image file was not created: $newImage"
            break
        }

        # Step 3: Create delta if there is a previous image
        if ($prevImage -ne "") {
            Write-Host "Calculating delta between $prevImage and $newImage"
            # Run the Rust delta calculation program
            & $rustDeltaCalculationProgram $prevImage $newImage $outputDir

            # Wait to ensure the Rust program executed successfully
            Start-Sleep -Seconds $longPauseDuration

            # Check if the Rust program executed successfully
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Error: Rust program failed with exit code $LASTEXITCODE"
                break
            }

            # Step 4: Delete previous image
            Write-Host "Deleting previous image: $prevImage"
            Remove-Item $prevImage -ErrorAction SilentlyContinue

            # Check if the previous image file was deleted successfully
            if (Test-Path $prevImage) {
                Write-Host "Error: Previous image file was not deleted: $prevImage"
                break
            }
        }

        # Step 5: Resume the VM
        Write-Host "Resuming VM..."
        VBoxManage controlvm $vmName resume

        # Update the previous image path
        Write-Host "Updating previous image path to: $newImage"
        $prevImage = $newImage

        # Step 6: Wait for the next interval
        Write-Host "Waiting for the next interval"
        Start-Sleep -Seconds $intervalMinutes  # Adjust the interval as needed
    }
    catch {
        Write-Error "An error occurred: $_"
        break
    }
}
