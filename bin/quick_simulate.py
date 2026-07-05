#!/usr/bin/env python3
"""
Quick Drake Robot Simulator
Simple example to load and visualize robot.urdf
"""

import numpy as np
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Parser,
    SceneGraph,
    Simulator,
    JointActuatorIndex,
    MeshcatVisualizer,
)

# Create builder
builder = DiagramBuilder()

# Add scene graph and plant
scene_graph = builder.AddSystem(SceneGraph())
plant = builder.AddSystem(MultibodyPlant(time_step=0.001))
plant.RegisterAsSourceForSceneGraph(scene_graph)

# Load URDF
parser = Parser(plant)
parser.AddModels("model/robot.urdf")
plant.Finalize()

# Add visualizer
meshcat_visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph)

# Build and create simulator
diagram = builder.Build()
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)

# Run simulation
print("Simulator running... Press Ctrl+C to stop")
print("Open MeshCat visualizer from the terminal output")

simulator.AdvanceTo(60)  # Run for 60 seconds
