# PowerShell script to find processes using the camera
Write-Host "Checking for processes that might be using the camera..." -ForegroundColor Yellow
Write-Host ""

# Common camera-using applications
$cameraApps = @(
    "Teams",
    "Zoom",
    "Skype",
    "Discord",
    "Slack",
    "obs64",
    "obs32",
    "WindowsCamera",
    "Camera",
    "Loom",
    "Webex"
)

$found = $false

foreach ($app in $cameraApps) {
    $processes = Get-Process | Where-Object { $_.ProcessName -like "*$app*" }
    if ($processes) {
        $found = $true
        Write-Host "❌ Found: $app" -ForegroundColor Red
        foreach ($proc in $processes) {
            Write-Host "   PID: $($proc.Id) - $($proc.ProcessName)" -ForegroundColor Red
        }
    }
}

if (-not $found) {
    Write-Host "✅ No obvious camera-using apps found running" -ForegroundColor Green
}

Write-Host ""
Write-Host "To kill a process, run: Stop-Process -Id <PID>" -ForegroundColor Cyan
Write-Host "Or use Task Manager to end these applications" -ForegroundColor Cyan


