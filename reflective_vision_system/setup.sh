#!/bin/bash
set -e

echo "============================================"
echo " Smart Vision System - Environment Setup"
echo "============================================"

# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download SAM checkpoint (ViT-H ~2.4GB)
mkdir -p models
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_PATH="models/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_PATH" ]; then
    echo "Downloading SAM checkpoint..."
    wget -q --show-progress "$SAM_URL" -O "$SAM_PATH"
else
    echo "SAM checkpoint already present."
fi

# 5. Download YOLOv8 base weights (auto-downloaded by ultralytics on first run)
echo "YOLOv8 weights will be downloaded automatically on first run."

# 6. Create output directories
mkdir -p outputs/images outputs/videos outputs/logs

echo ""
echo "Setup complete. Activate env with:  source venv/bin/activate"
