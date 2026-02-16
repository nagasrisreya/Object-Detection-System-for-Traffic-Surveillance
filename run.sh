#!/bin/bash

# Traffic Vision - Setup & Launch Script for Linux/Mac

echo "========================================"
echo "  Traffic Vision - Setup & Launch"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "Python found!"
python3 --version
echo

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo

# Launch the application
echo "Starting Traffic Vision..."
streamlit run app.py

