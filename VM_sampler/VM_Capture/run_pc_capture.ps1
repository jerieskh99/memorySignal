
param (
    [string]$root = (Get-Location).Path,
    [string]$prod = "capture_producer.ps1",
    [string]$cons = "capture_consumer.ps1",
    [switch]$hidden    
)

$producerPath = Join-Path $root $prod
$consumerPath = Join-Path $root $cons

if (-not (Test-Path $producerPath)) { throw ("Producer not found: $producerPath")}
if (-not (Test-Path $consumerPath)) { throw ("Producer not found: $consumerPath")}

$commonArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File")
$startInfo = @(
    FilePath = "powershell.exe",
    WorkingDirectory = $root
)

if ($Hidden) {
    # Run both in backgroud (No new windows)
    Start-Process @startInfo -WindowStyle Hidden -ArgumentList ($commonArgs + ($consumerPath))
    Start-Process @startInfo -WindowStyly Hidden -ArgumentList ($commonArgs + ($producerPath))
    Write-Host "Started consumer + producer in hidden Background Processes."
} else {
    # Run both in separate visible windows
    Start-Process @startInfo -ArgumentList ($commonArgs + ($consumerPath))
    Start-Process @startInfo -ArgumentList ($commonArgs + ($producerPath))
    Write-Host "Started consumer + producer in separate PowerShell windows."
}

Write-Host "Root: $Root"
Write-Host "Consumer: $consumerPath"
Write-Host "Producer: $producerPath"