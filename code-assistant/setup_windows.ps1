# Windows PowerShell setup script for adversarial coding system
# Run this in the code-assistant directory

Write-Host "Setting up Adversarial Coding System on Windows..." -ForegroundColor Green

# Check Python installation
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Blue
} catch {
    Write-Host "ERROR: Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "Removing existing .venv directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv
}

python -m venv .venv

# Activate virtual environment (Windows PowerShell style)
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install packages one by one (as requested)
Write-Host "Installing StrandsAgents packages individually..." -ForegroundColor Yellow

$packages = @(
    "strands-agents>=0.1.0",
    "strands-agents-tools>=0.1.0",
    "mcp>=0.1.0",
    "boto3>=1.28.0",
    "requests>=2.31.0",
    "python-dotenv>=1.0.0",
    "black>=23.0.0",
    "pytest>=7.0.0"
)

foreach ($package in $packages) {
    Write-Host "Installing $package..." -ForegroundColor Cyan
    pip install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Failed to install $package" -ForegroundColor Red
    } else {
        Write-Host "SUCCESS: Installed $package" -ForegroundColor Green
    }
}

# Verify installations
Write-Host "Verifying installations..." -ForegroundColor Yellow
python -c "import strands; print('strands-agents imported successfully')"
python -c "import strands_tools; print('strands-agents-tools imported successfully')"
python -c "import mcp; print('MCP imported successfully')"

Write-Host "Setup complete! To activate the environment, run:" -ForegroundColor Green
Write-Host ".\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Make sure Ollama is installed and running" -ForegroundColor White
Write-Host "2. Pull required models: ollama pull llama3.2:3b" -ForegroundColor White
Write-Host "3. Run: python main.py" -ForegroundColor White