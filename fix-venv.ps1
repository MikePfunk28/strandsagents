# Fix Virtual Environment - Recreate for Windows
# This script deletes the WSL-created .venv and recreates it for Windows

Write-Host "`nFixing Virtual Environment for Windows" -ForegroundColor Cyan
Write-Host "=================================================="

# Step 1: Check if we're in a venv and deactivate
if ($env:VIRTUAL_ENV) {
    Write-Host "`nDeactivating current virtual environment..." -ForegroundColor Yellow
    deactivate
}

# Step 2: Backup requirements if they exist
if (Test-Path "requirements.txt") {
    Write-Host "`nFound requirements.txt - will reinstall packages" -ForegroundColor Green
} else {
    Write-Host "`nNo requirements.txt found - you'll need to reinstall packages manually" -ForegroundColor Yellow
}

# Step 3: Remove the old WSL-created venv
if (Test-Path ".venv") {
    Write-Host "`nRemoving WSL-created .venv folder..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv
    Write-Host "Removed old .venv" -ForegroundColor Green
}

# Step 4: Check Python availability
Write-Host "`nChecking for Windows Python..." -ForegroundColor Cyan
$pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
if ($pythonPath) {
    Write-Host "Found Python at: $pythonPath" -ForegroundColor Green
} else {
    Write-Host "Python not found in PATH!" -ForegroundColor Red
    exit 1
}

# Step 5: Create new Windows venv
Write-Host "`nCreating new Windows virtual environment..." -ForegroundColor Cyan
python -m venv .venv

if ($LASTEXITCODE -eq 0) {
    Write-Host "Created new .venv successfully" -ForegroundColor Green
} else {
    Write-Host "Failed to create venv" -ForegroundColor Red
    exit 1
}

# Step 6: Verify the new venv configuration
Write-Host "`nVerifying new venv configuration..." -ForegroundColor Cyan
$pyvenvContent = Get-Content .venv\pyvenv.cfg
Write-Host ($pyvenvContent | Out-String) -ForegroundColor Gray

# Step 7: Activate and upgrade pip
Write-Host "`nActivating venv and upgrading pip..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

if ($env:VIRTUAL_ENV) {
    Write-Host "Virtual environment activated" -ForegroundColor Green
    python -m pip install --upgrade pip
} else {
    Write-Host "Auto-activation failed - activate manually with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
}

# Step 8: Reinstall packages
if (Test-Path "requirements.txt") {
    Write-Host "`nReinstalling packages from requirements.txt..." -ForegroundColor Cyan
    Write-Host "This may take several minutes..." -ForegroundColor Yellow
    pip install -r requirements.txt

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Packages installed successfully" -ForegroundColor Green
    } else {
        Write-Host "Some packages failed to install" -ForegroundColor Yellow
    }
}

# Step 9: Install Playwright browsers
Write-Host "`nInstalling Playwright Chromium browser..." -ForegroundColor Cyan
playwright install chromium

if ($LASTEXITCODE -eq 0) {
    Write-Host "Playwright Chromium installed" -ForegroundColor Green
} else {
    Write-Host "Playwright installation failed - run manually: playwright install chromium" -ForegroundColor Yellow
}

Write-Host "`n=================================================="
Write-Host "Virtual environment setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. If not already activated: .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. Test your script: python agent.py" -ForegroundColor White
Write-Host ""
