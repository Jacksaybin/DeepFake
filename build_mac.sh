#!/bin/bash
# Script to build Live DeepFake into a standalone macOS App

echo "=== Starting Live DeepFake Build ==="
source venv/bin/activate

# Install PyInstaller
pip install pyinstaller

# Create the PyInstaller spec/build
echo "Packaging application..."
pyinstaller --noconfirm \
    --name "Live DeepFake" \
    --windowed \
    --add-data "locales:locales" \
    --add-data "modules/ui.json:modules" \
    --add-data "models:models" \
    --hidden-import "pyvirtualcam" \
    --hidden-import "PIL" \
    --hidden-import "cv2" \
    --hidden-import "customtkinter" \
    run.py

echo "=== Build Complete! ==="
echo "You can find your standalone app in the 'dist' folder."
