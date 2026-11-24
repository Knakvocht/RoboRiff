$ErrorActionPreference = "Stop"

Write-Host "Building RoboRiff 0.1.0..."

py -3.13 -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --windowed `
  --name RoboRiff `
  --paths vendor `
  --collect-submodules tabgen `
  --collect-all basic_pitch `
  --collect-all demucs `
  --hidden-import torchlibrosa `
  app.py

Write-Host "Build complete: dist\RoboRiff.exe"
