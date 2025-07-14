#!/bin/bash

echo "Starting VR Body Tracker..."
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create output directory if it doesn't exist
mkdir -p output

# Start the application
echo "Starting server..."
cd backend
python main.py