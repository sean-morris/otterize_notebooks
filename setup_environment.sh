#!/bin/bash

# Setup script for [YOUr_NAME] environment
# Creates conda environment, installs packages, and sets up Jupyter kernel

set -e  # Exit on error

echo "=================================================="
echo "Setting up environment"
echo "=================================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

echo "🔧 Updating conda itself..."
conda update -n base -c conda-forge conda -y

echo "🔧 Creating/updating conda environment from environment.yaml..."
conda env create -f environment.yaml 2>/dev/null || conda env update -f environment.yaml --prune

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to create/update environment"
    exit 1
fi

echo "✓ Environment created/updated"
echo ""

# Activate environment (note: this only works within the script)
echo "🔧 Installing Jupyter kernel for ..."
eval "$(conda shell.bash hook)"
conda activate data271

# Install kernel so nbclient can use it
python -m ipykernel install --user --name [YOUR_NAME] --display-name "Python ([YOUR_NAME])"
echo "✓ Kernel installed"
echo ""

# Verify installation
echo "🔍 Verifying setup..."
echo ""

# Check kernel
if jupyter kernelspec list | grep -q "data271"; then
    echo "✓ Kernel 'data271' is registered with Jupyter"
else
    echo "❌ WARNING: Kernel not found in jupyter kernelspec list"
fi

# Check otter-grader
if conda run -n data271 python -c "import otter" 2>/dev/null; then
    OTTER_VERSION=$(conda run -n [YOUR_NAME] python -c "import otter; print(otter.__version__)")
    echo "✓ otter-grader ${OTTER_VERSION} installed"
else
    echo "❌ ERROR: otter-grader not found"
    exit 1
fi

# Check nbclient
if conda run -n [YOUR_NAME] python -c "import nbclient" 2>/dev/null; then
    echo "✓ nbclient installed"
else
    echo "❌ ERROR: nbclient not found"
    exit 1
fi

# Check datascience
if conda run -n [YOUR_NAME] python -c "import datascience" 2>/dev/null; then
    echo "✓ datascience package installed"
else
    echo "❌ ERROR: datascience package not found"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo "=================================================="
echo ""
echo "The [YOUR_NAME] environment is now set up with all dependencies."
echo ""
echo "To activate it, run:"
echo "  conda activate [YOUR_NAME]"
echo ""
echo "Or source this command (to activate in current shell):"
echo "  eval \"\$(conda shell.bash hook)\" && conda activate data271"
echo ""
echo ""
