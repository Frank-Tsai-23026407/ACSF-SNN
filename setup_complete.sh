#!/bin/bash
# Complete setup script for ACSF-SNN environment
# This script will:
# 1. Download and install MuJoCo 210
# 2. Create Python 3.9 conda environment
# 3. Install all required packages (PyTorch, gym, mujoco-py, spikingjelly, etc.)
# 4. Set up environment variables

set -e  # Exit on error

echo "============================================"
echo "ACSF-SNN Complete Environment Setup"
echo "============================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# ============================================
# Step 1: Install MuJoCo 210
# ============================================
echo "Step 1/5: Installing MuJoCo 210..."
mkdir -p ~/.mujoco
cd ~/.mujoco

if [ ! -d "mujoco210" ]; then
    echo "  Downloading MuJoCo 210..."
    wget -q --show-progress https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
    echo "  Extracting..."
    tar -xzf mujoco210-linux-x86_64.tar.gz
    rm mujoco210-linux-x86_64.tar.gz
    echo "  ✓ MuJoCo 210 installed to ~/.mujoco/mujoco210"
else
    echo "  ✓ MuJoCo 210 already exists"
fi

# Set environment variables for this session
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210

# Add to bashrc if not already present
if ! grep -q "MUJOCO_PY_MUJOCO_PATH" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# MuJoCo paths for ACSF-SNN" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin" >> ~/.bashrc
    echo "export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210" >> ~/.bashrc
    echo "  ✓ Environment variables added to ~/.bashrc"
else
    echo "  ✓ Environment variables already in ~/.bashrc"
fi

echo ""

# ============================================
# Step 2: Create conda environment
# ============================================
echo "Step 2/5: Creating conda environment (acsf-py39)..."

# Check if environment already exists
if conda env list | grep -q "^acsf-py39 "; then
    echo "  Environment 'acsf-py39' already exists."
    read -p "  Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  Removing existing environment..."
        conda env remove -n acsf-py39 -y
        echo "  Creating new environment..."
        conda create -n acsf-py39 python=3.9 -y
        echo "  ✓ Environment recreated"
    else
        echo "  Using existing environment"
    fi
else
    echo "  Creating new environment..."
    conda create -n acsf-py39 python=3.9 -y
    echo "  ✓ Environment created"
fi

echo ""

# ============================================
# Step 3: Install Python packages
# ============================================
echo "Step 3/5: Installing Python packages..."
echo "  This may take 5-10 minutes..."
echo ""

# Use conda run to execute pip commands in the environment
CONDA_RUN="conda run -n acsf-py39 --no-capture-output"

# Check CUDA availability
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "  Detected CUDA version: $CUDA_VERSION"
    
    # Determine PyTorch version and index URL based on CUDA version
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        # CUDA 12.x: use PyTorch 2.1+ which supports cu121
        TORCH_VERSION="2.1.2"
        TORCHVISION_VERSION="0.16.2"
        TORCHAUDIO_VERSION="2.1.2"
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        echo "  Using PyTorch 2.1.2 for CUDA 12.x"
    elif [[ "$CUDA_VERSION" == "11.8"* ]] || [[ "$CUDA_VERSION" == "11.7"* ]]; then
        # CUDA 11.7/11.8: use PyTorch 2.0.1 with cu118
        TORCH_VERSION="2.0.1"
        TORCHVISION_VERSION="0.15.2"
        TORCHAUDIO_VERSION="2.0.2"
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        echo "  Using PyTorch 2.0.1 for CUDA 11.x"
    else
        # Unknown CUDA version: default to cu118
        TORCH_VERSION="2.0.1"
        TORCHVISION_VERSION="0.15.2"
        TORCHAUDIO_VERSION="2.0.2"
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        echo "  ⚠️  Unknown CUDA version, defaulting to PyTorch 2.0.1 with cu118"
    fi
else
    echo "  ⚠️  CUDA not detected, installing CPU-only PyTorch"
    TORCH_VERSION="2.0.1"
    TORCHVISION_VERSION="0.15.2"
    TORCHAUDIO_VERSION="2.0.2"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

echo ""
echo "  [1/7] Installing PyTorch ${TORCH_VERSION}..."
$CONDA_RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url $TORCH_INDEX

echo "  [2/7] Installing Cython and numpy..."
$CONDA_RUN pip install 'Cython<3' 'numpy==1.23.5'

echo "  [3/7] Installing gym..."
$CONDA_RUN pip install 'setuptools<66' wheel
# Install gym without upgrading numpy
$CONDA_RUN pip install --no-deps gym==0.26.2
# Install gym dependencies without upgrading numpy
$CONDA_RUN pip install 'cloudpickle>=1.2.0' 'gym-notices>=0.0.4' importlib_metadata

echo "  [4/7] Installing mujoco-py (this may take a while)..."
$CONDA_RUN pip install mujoco-py==2.1.2.14

echo "  [5/7] Installing spikingjelly..."
$CONDA_RUN pip install spikingjelly==0.0.0.0.14

echo "  [6/7] Installing scientific computing packages..."
$CONDA_RUN pip install 'scipy==1.11.1'

echo "  [7/7] Installing visualization and utilities..."
# Install packages with strict numpy compatibility
$CONDA_RUN pip install tqdm pyyaml 'matplotlib<3.9' 'opencv-python==4.8.1.78' 'imageio==2.31.6' 'cloudpickle<4' 'pyglet==2.0.10' 'glfw<3' 'pillow>=8.3.2,<10.1'

echo ""
echo "  ✓ All packages installed"

# ============================================
# Step 4: Test installation
# ============================================
echo ""
echo "Step 4/5: Testing installation..."

# Test basic imports
echo "  Testing basic imports..."
$CONDA_RUN python -c "import torch; import numpy; import gym; import spikingjelly; print('  ✓ Basic packages OK')" || {
    echo "  ❌ Basic package test failed"
    exit 1
}

# Test gym compatibility
echo "  Testing gym compatibility layer..."
cd ~/ACSF-SNN
$CONDA_RUN python -c "import gym_compat; import gym; env = gym.make('CartPole-v1'); state = env.reset(); assert not isinstance(state, tuple), 'Compatibility layer failed'; print('  ✓ Gym compatibility OK')" || {
    echo "  ❌ Gym compatibility test failed"
    exit 1
}

# Test MuJoCo (this will trigger compilation on first run)
echo "  Testing MuJoCo (may take 2-5 minutes on first run)..."
$CONDA_RUN python -c "import mujoco_py; print('  ✓ mujoco-py OK')" || {
    echo "  ⚠️  mujoco-py import failed (may need manual fixing)"
    echo "  Common fix: sudo apt-get install -y libosmesa6-dev libgl1 libglew-dev patchelf"
}

# Test MuJoCo environment
echo "  Testing MuJoCo environment..."
$CONDA_RUN python -c "import gym_compat; import gym; env = gym.make('Ant-v3'); env.reset(); print('  ✓ MuJoCo environment OK')" || {
    echo "  ⚠️  MuJoCo environment test failed"
}

echo ""

# ============================================
# Step 5: Final setup
# ============================================
echo "Step 5/5: Final setup..."

# Create necessary directories
cd ~/ACSF-SNN
mkdir -p models buffers results videos

echo "  ✓ Created workspace directories (models, buffers, results, videos)"
echo ""

# ============================================
# Summary
# ============================================
echo "============================================"
echo "✓ Setup Complete!"
echo "============================================"
echo ""
echo "Environment: acsf-py39"
echo "Python: 3.9"
echo "PyTorch: ${TORCH_VERSION}"
echo "Gym: 0.26.2"
echo "MuJoCo: 210"
echo ""
echo "To activate the environment:"
echo "  conda activate acsf-py39"
echo ""
echo "To start training:"
echo "  cd ~/ACSF-SNN"
echo "  conda activate acsf-py39"
echo ""
echo "  # Train behavioral policy (TD3)"
echo "  python main.py --env=Ant-v3 --seed=9853 --gpu=0 --train_behavioral --mode=TD3"
echo ""
echo "  # Generate replay buffer"
echo "  python main.py --env=Ant-v3 --seed=9853 --gpu=0 --generate_buffer --mode=TD3"
echo ""
echo "  # Train SNN (ACSF)"
echo "  python main.py --env=Ant-v3 --seed=9853 --gpu=0 --mode=AEAD --buffer=TD3 --T=4"
echo ""
echo "For detailed instructions, see QUICKSTART_ZH.md"
echo ""
echo "Note: Environment variables have been added to ~/.bashrc"
echo "      Restart your shell or run: source ~/.bashrc"
echo ""
