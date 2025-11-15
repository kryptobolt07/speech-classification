# =========================================
# Audio Classifier Setup (CPU-only)
# =========================================

# ---- 1. Check Python Installation ----
Write-Host "Checking Python installation..."

$python = Get-Command python -ErrorAction SilentlyContinue

if (-Not $python) {
    Write-Host "`nERROR: Python not found!" -ForegroundColor Red
    Write-Host "Please install Python from: https://www.python.org/downloads/" -ForegroundColor Cyan
    Write-Host "Make sure to check 'Add Python to PATH' during installation."
    exit 1
}

$ver = python --version
Write-Host "Python version detected: $ver" -ForegroundColor Green

# ---- 2. Create Virtual Environment ----
Write-Host "`nCreating virtual environment 'venv'..."
python -m venv venv

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create virtual environment!" -ForegroundColor Red
    exit 1
}

# ---- 3. Activate Virtual Environment ----
Write-Host "`nActivating virtual environment..."
& .\venv\Scripts\Activate.ps1

# ---- 4. Upgrade pip ----
Write-Host "`nUpgrading pip..."
python -m pip install --upgrade pip

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: pip upgrade failed, continuing anyway..." -ForegroundColor Yellow
}

# ---- 5. Install PyTorch CPU-only ----
Write-Host "`nInstalling PyTorch (CPU-only)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyTorch installation failed!" -ForegroundColor Red
    exit 1
}

# ---- 6. Install audio & numeric libraries ----
Write-Host "`nInstalling librosa, numpy, soundfile, scipy..."
pip install librosa numpy soundfile scipy

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Audio libraries installation failed!" -ForegroundColor Red
    exit 1
}

# ---- 7. (Optional) Utilities ----
Write-Host "`nInstalling optional utilities (tqdm, matplotlib)..."
pip install tqdm matplotlib

# ---- 8. Verify Installation ----
Write-Host "`nVerifying installation..."
python -c "import torch, librosa, numpy, soundfile; print('✓ All dependencies loaded successfully!')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=====================================" -ForegroundColor Green
    Write-Host " Setup Complete! ✓" -ForegroundColor Green
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host "`nTo use the classifier:"
    Write-Host "1. Activate environment: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host "2. Run classifier: python classify_audio.py <audio_file> [model_path]" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Green
} else {
    Write-Host "`nWARNING: Installation completed but verification failed." -ForegroundColor Yellow
    Write-Host "Please check for errors above." -ForegroundColor Yellow
}
