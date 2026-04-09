#!/bin/bash
# PredictPulse — Demo Script
# Runs the complete pipeline locally for testing/demo purposes

set -e

echo "========================================="
echo "  PredictPulse — Full Pipeline Demo"
echo "========================================="

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo ""
echo "[1/6] Data Collection..."
python3 src/01_data_collection.py

echo ""
echo "[2/6] Feature Engineering..."
python3 src/02_feature_engineering.py

echo ""
echo "[3/6] Model Training..."
python3 src/03_model_training.py

echo ""
echo "[4/6] Visualization..."
python3 src/04_visualization.py

echo ""
echo "[5/6] Claude AI Analysis..."
python3 src/05_claude_analysis.py

echo ""
echo "[6/6] API Test..."
python3 src/06_deploy_api.py

echo ""
echo "========================================="
echo "  Pipeline Complete!"
echo "========================================="
