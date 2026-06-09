#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Setting up Atmogen environment..."

# 1. Create a virtual environment named 'venv' if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# 2. Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# 3. Upgrade pip for good measure
echo "Upgrading pip..."
pip install --upgrade pip

# 4. Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "Setup complete! To start using Atmogen, run:"
echo "source venv/bin/activate"
echo "python main.py --help"
echo "============================================================"
