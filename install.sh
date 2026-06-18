#!/bin/bash
# Installation script for ROS Max-k-Cut project

echo "======================================"
echo "ROS Max-k-Cut Installation Script"
echo "======================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
echo ""

# Ask for CUDA version
echo "Select your CUDA version:"
echo "1) CPU only"
echo "2) CUDA 11.8"
echo "3) CUDA 12.1"
read -p "Enter choice (1-3): " cuda_choice

case $cuda_choice in
    1)
        TORCH_INDEX=""
        PYG_INDEX="https://data.pyg.org/whl/torch-2.2.2+cpu.html"
        echo "Installing for CPU..."
        ;;
    2)
        TORCH_INDEX="--index-url https://download.pytorch.org/whl/cu118"
        PYG_INDEX="https://data.pyg.org/whl/torch-2.2.2+cu118.html"
        echo "Installing for CUDA 11.8..."
        ;;
    3)
        TORCH_INDEX="--index-url https://download.pytorch.org/whl/cu121"
        PYG_INDEX="https://data.pyg.org/whl/torch-2.2.2+cu121.html"
        echo "Installing for CUDA 12.1..."
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Step 1: Installing core dependencies..."
pip install numpy scipy networkx scikit-learn joblib tqdm pyyaml requests matplotlib

echo ""
echo "Step 2: Installing optimization libraries..."
pip install cvxpy cvxopt clarabel ecos osqp qdldl scs

echo ""
echo "Step 3: Installing PyTorch..."
if [ -z "$TORCH_INDEX" ]; then
    pip install torch==2.2.2 torchvision torchaudio
else
    pip install torch==2.2.2 torchvision torchaudio $TORCH_INDEX
fi

echo ""
echo "Step 4: Installing PyTorch Geometric..."
pip install torch-geometric

echo ""
echo "Step 5: Installing PyG dependencies..."
# Note: pyg-lib is not available on macOS, so we skip it
# The core functionality should work without it
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - installing PyG dependencies without pyg-lib..."
    pip install torch-scatter torch-sparse torch-cluster -f $PYG_INDEX || {
        echo "Warning: Pre-built wheels not available. Installing from source (this may take a while)..."
        pip install torch-scatter torch-sparse torch-cluster
    }
else
    pip install pyg-lib torch-scatter torch-sparse torch-cluster -f $PYG_INDEX
fi

echo ""
echo "======================================"
echo "Installation complete!"
echo "======================================"
echo ""
echo "To test the installation, try:"
echo "  python -c 'import torch; import torch_geometric; print(\"Success!\")'"
