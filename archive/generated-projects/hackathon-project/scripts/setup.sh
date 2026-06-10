#!/bin/bash
# PredictPulse — Quick Setup Script
# Run this to install local dependencies for development/testing

set -e

echo "========================================="
echo "  PredictPulse — Setup"
echo "========================================="

# Check Python version
python3 --version || { echo "Error: Python 3 required"; exit 1; }

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install requests pandas numpy scikit-learn plotly anthropic pytrends

# Check for API keys
echo ""
echo "========================================="
echo "  Environment Setup"
echo "========================================="

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
    echo "Please edit .env with your API keys:"
    echo "  - ANTHROPIC_API_KEY (optional, for Claude AI analysis)"
    echo "  - FRED_API_KEY (optional, for economic indicators)"
else
    echo ".env already exists"
fi

echo ""
echo "Setup complete! To run locally:"
echo "  source .venv/bin/activate"
echo "  python src/01_data_collection.py"
echo ""
echo "For Zerve deployment:"
echo "  1. Create account at zerve.ai"
echo "  2. Create new Canvas"
echo "  3. Copy blocks 01-06 from src/ in order"
echo "  4. Run each block sequentially"
echo "  5. Deploy Block 06 as API"
echo "========================================="
