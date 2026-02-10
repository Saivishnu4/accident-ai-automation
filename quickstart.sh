#!/bin/bash

# Quick Start Script for Accident FIR Automation System
# This script sets up the environment and starts the application

set -e  # Exit on error

echo "========================================="
echo "Accident FIR Automation - Quick Start"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}[1/6] Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python3 -c 'import sys; assert sys.version_info >= (3,8)' 2>/dev/null; then
    echo -e "${RED}Error: Python 3.8+ required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Create virtual environment
echo -e "${YELLOW}[2/6] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo "Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo -e "${YELLOW}[3/6] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Install dependencies
echo -e "${YELLOW}[4/6] Installing dependencies...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Create necessary directories
echo -e "${YELLOW}[5/6] Creating directories...${NC}"
mkdir -p data/{raw,processed,uploads,outputs} logs models
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Set up environment variables
echo -e "${YELLOW}[6/6] Setting up environment...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env file from template"
    echo -e "${YELLOW}Please edit .env with your configuration${NC}"
else
    echo ".env file already exists"
fi
echo -e "${GREEN}✓ Environment setup complete${NC}"
echo ""

# Display next steps
echo "========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Download pre-trained models:"
echo "   python scripts/download_models.py --all"
echo ""
echo "2. Start the API server:"
echo "   uvicorn src.api.main:app --reload"
echo ""
echo "3. Access the API documentation:"
echo "   http://localhost:8000/docs"
echo ""
echo "For detailed instructions, see PROJECT_GUIDE.md"
echo ""

# Ask if user wants to start the server
read -p "Would you like to start the API server now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Starting API server...${NC}"
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
fi
