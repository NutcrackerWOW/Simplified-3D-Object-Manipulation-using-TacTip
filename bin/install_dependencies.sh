#!/bin/bash

# Install Drake and dependencies
echo "Installing Drake and required packages..."
pip install drake numpy

echo ""
echo "Installation complete!"
echo ""
echo "You can now run the simulator scripts:"
echo ""
echo "1. Quick start (minimal example):"
echo "   python quick_simulate.py"
echo ""
echo "2. Full simulator with motion:"
echo "   python simulate_robot.py"
echo ""
echo "3. Advanced controller (PD control):"
echo "   python advanced_controller.py"
echo ""
echo "4. Data logging simulator:"
echo "   python logging_simulator.py"
echo ""
