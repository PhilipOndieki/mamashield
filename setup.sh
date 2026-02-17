#!/bin/bash
#
# MamaShield Quick Setup Script for Ubuntu
# This script automates the installation process
#
# Usage: chmod +x setup.sh && ./setup.sh
#

set -e  # Exit on error

echo "========================================="
echo "  MamaShield Installation Script"
echo "  GEGIS Hackathon 2026 - Track 5"
echo "========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Ubuntu
if [ ! -f /etc/lsb-release ]; then
    echo -e "${RED}Error: This script is designed for Ubuntu.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Detected Ubuntu system${NC}"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python 3.8 or higher required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"
echo ""

# Install system dependencies
echo "Installing system dependencies..."
echo "This may require your password."
echo ""

sudo apt update -qq

echo "Installing Python development tools..."
sudo apt install -y python3-pip python3-venv python3-dev build-essential > /dev/null 2>&1
echo -e "${GREEN}✓ Python tools installed${NC}"

echo "Installing geospatial libraries..."
sudo apt install -y gdal-bin libgdal-dev libspatialindex-dev libgeos-dev libproj-dev > /dev/null 2>&1
echo -e "${GREEN}✓ Geospatial libraries installed${NC}"
echo ""

# Check for data files
echo "Checking for data files..."
if [ ! -d "../hackathon_data" ]; then
    echo -e "${YELLOW}⚠ Warning: Data directory not found at ../hackathon_data${NC}"
    echo "Please ensure your data files are in: $(pwd)/../hackathon_data/"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    DATA_FILES=("kenya_dhs_dataset_complete.csv" "kenya_dhs_covariates.csv" "kenya_dhs_dataset_gps.geojson")
    MISSING_FILES=()
    
    for file in "${DATA_FILES[@]}"; do
        if [ ! -f "../hackathon_data/$file" ]; then
            MISSING_FILES+=("$file")
        fi
    done
    
    if [ ${#MISSING_FILES[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ All data files found${NC}"
    else
        echo -e "${YELLOW}⚠ Missing data files:${NC}"
        for file in "${MISSING_FILES[@]}"; do
            echo "  - $file"
        done
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Removing old one...${NC}"
    rm -rf venv
fi

python3 -m venv venv
echo -e "${GREEN}✓ Virtual environment created${NC}"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ Pip upgraded${NC}"
echo ""

# Install Python packages
echo "Installing Python packages..."
echo "This may take 5-10 minutes..."
echo ""

pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All Python packages installed successfully${NC}"
else
    echo ""
    echo -e "${RED}✗ Error installing Python packages${NC}"
    echo "Please check the error messages above and try manual installation."
    exit 1
fi
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "import streamlit; import pandas; import geopandas; import plotly" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All packages imported successfully${NC}"
else
    echo -e "${RED}✗ Error importing packages${NC}"
    exit 1
fi
echo ""

# Success message
echo "========================================="
echo -e "${GREEN}  Installation Complete! 🎉${NC}"
echo "========================================="
echo ""
echo "To run MamaShield:"
echo ""
echo "  1. Activate virtual environment:"
echo "     ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo "  2. Run the application:"
echo "     ${YELLOW}streamlit run app.py${NC}"
echo ""
echo "  3. Open your browser to:"
echo "     ${YELLOW}http://localhost:8501${NC}"
echo ""
echo "========================================="
echo ""

# Ask if user wants to run now
read -p "Would you like to start the application now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting MamaShield..."
    echo ""
    streamlit run app.py
fi
