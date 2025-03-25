# Gaussian Hierarchy

A hierarchical 3D Gaussian splatting implementation with efficient spatial hierarchy construction and traversal. Based on [hierarchical-3d-gaussians](https://github.com/graphdeco-inria/hierarchical-3d-gaussians).

Refactored for clean code structure and better Python integration with PyTorch bindings.

## Installation

### Prerequisites

- CUDA 11.0+
- PyTorch 2.0+
- Eigen 3.4
- Python 3.8+

### Install from GitHub

```bash
pip install git+https://github.com/JesseSenior/gaussianhierarchy.git
```

### Build from Source

```bash
git clone https://github.com/JesseSenior/gaussianhierarchy.git
cd gaussianhierarchy

# Install build dependencies
pip install torch ninja
# Install Eigen (Ubuntu example)
sudo apt-get install libeigen3-dev

# Build and install
pip install .
```
