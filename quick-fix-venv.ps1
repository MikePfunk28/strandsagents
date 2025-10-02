# Quick Fix for MSYS2/MINGW Python Virtual Environment
# This venv was created with MSYS2 Python which is incompatible with Windows pip/playwright

Write-Host "`n🔧 FIXING MSYS2/MINGW VIRTUAL ENVIRONMENT" -ForegroundColor Cyan
Write-Host "=" * 60

# Step 1: Deactivate if active
if ($env:VIRTUAL_ENV) {
    Write-Host "`n⚠️  Deactivating current venv..." -ForegroundColor Yellow
    deactivate
}

# Step 2: Remove broken venv
Write-Host "`n🗑️  Removing MSYS2-based .venv folder..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Remove-Item -Recurse -Force .venv
    Write-Host "✅ Removed corrupted .venv" -ForegroundColor Green
}

# Step 3: Find Windows Python (not MSYS2)
Write-Host "`n🔍 Finding Windows Python..." -ForegroundColor Cyan
$pythons = Get-Command python -ErrorAction SilentlyContinue | Where-Object {
    $_.Source -notlike "*msys64*" -and $_.Source -notlike "*mingw*"
}

if ($pythons) {
    $pythonPath = $pythons[0].Source
    Write-Host "✅ Found Windows Python: $pythonPath" -ForegroundColor Green
} else {
    Write-Host "❌ No Windows Python found! Install Python from python.org" -ForegroundColor Red
    exit 1
}

# Step 4: Create new Windows venv
Write-Host "`n🔨 Creating new Windows virtual environment..." -ForegroundColor Cyan
& $pythonPath -m venv .venv

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Created new .venv successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to create venv" -ForegroundColor Red
    exit 1
}

# Step 5: Verify configuration
Write-Host "`n📋 New venv configuration:" -ForegroundColor Cyan
Get-Content .venv\pyvenv.cfg | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }

# Step 6: Activate and install packages
Write-Host "`n📦 Activating venv and installing packages..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

if (Test-Path "requirements.txt") {
    Write-Host "`nInstalling from requirements.txt (this will take several minutes)..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Packages installed" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Some packages may have failed" -ForegroundColor Yellow
    }
}

# Step 7: Install Playwright
Write-Host "`n🌐 Installing Playwright browsers..." -ForegroundColor Cyan
playwright install chromium

Write-Host "`n" + ("=" * 60)
Write-Host "🎉 Virtual environment fixed!" -ForegroundColor Green
Write-Host "`nNow test with:" -ForegroundColor Cyan
Write-Host "  python agent.py" -ForegroundColor White
