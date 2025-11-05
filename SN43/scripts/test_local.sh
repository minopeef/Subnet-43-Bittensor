#!/bin/bash
set -e

echo "======================================"
echo "SN43 TSP Local Testing"
echo "======================================"
echo ""

# Setup
echo "1. Setting up environment..."
pip install -e . -q
python scripts/setup_datasets.py

echo ""
echo "2. Running tests..."
pytest tests/test_local.py -v -s

echo ""
echo "======================================"
echo "✓ All tests completed successfully!"
echo "======================================"