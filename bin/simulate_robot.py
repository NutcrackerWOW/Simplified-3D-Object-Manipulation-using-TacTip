#!/usr/bin/env python3
"""
Drake Robot Simulator with MeshCat Visualization
Loads and simulates robot.urdf with real-time visualization
"""

import numpy as np
import os
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Parser,
    SceneGraph,
    Simulator,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
)


def create_robot_simulator(urdf_path, show_visualization=True):
    """
    Create and configure a robot simulator using Drake's default system setup.
    
    Args:
        urdf_path: Path to the URDF file
        show_visualization: Whether to show MeshCat visualization
        
    Returns:
        simulator: Configured simulator
        plant: MultibodyPlant
    """
    # Create the builder
    builder = DiagramBuilder()
    
    # Create and add the scene graph
    scene_graph = builder.AddSystem(SceneGraph())
    scene_graph.set_name("scene_graph")
    
    # Create and add the multibody plant
    plant = builder.AddSystem(MultibodyPlant(time_step=0.001))
    plant.set_name("plant")
    
    # Parse the URDF file BEFORE registering with scene graph
    parser = Parser(plant)
    model_instance = parser.AddModels(urdf_path)
    
    # Finalize the plant (must be done before registering with scene graph)
    plant.Finalize()
    
    # Now register the plant as a source for the scene graph
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    
    print(f"Robot loaded successfully!")
    print(f"Number of bodies: {plant.num_bodies()}")
    print(f"Number of joints: {plant.num_joints()}")
    print(f"Number of actuators: {plant.num_actuators()}")
    
    # Print joint information
    print("\nJoints:")
    for joint_index in plant.GetJointIndices():
        joint = plant.get_joint(joint_index)
        print(f"  - {joint.name()}: {joint.type_name()}")
    
    # Add default visualization if requested
    if show_visualization:
        meshcat = StartMeshcat()
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph.get_query_output_port(),
            meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration)
        )
        print(f"\nMeshCat listening at http://localhost:7000")
    
    # Connect geometry query output from scene graph
    builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port()
    )
    
    # Connect the necessary plant input ports
    from pydrake.systems.primitives import ConstantVectorSource
    
    # Connect generalized force input if needed
    try:
        zero_forces = builder.AddSystem(
            ConstantVectorSource(np.zeros(plant.num_velocities()))
        )
        builder.Connect(zero_forces.get_output_port(),
                       plant.get_applied_generalized_force_input_port())
    except Exception as e:
        pass
    
    # Connect spatial forces input if needed
    try:
        zero_spatial_forces = builder.AddSystem(
            ConstantVectorSource(np.zeros(6 * plant.num_bodies()))
        )
        builder.Connect(zero_spatial_forces.get_output_port(),
                       plant.get_applied_spatial_force_input_port())
    except Exception as e:
        pass
    
    # Build the diagram
    diagram = builder.Build()
    
    # Create simulator
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)  # Run at real-time speed
    
    # Get the context and set initial conditions
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)
    
    # Set initial joint angles (all zeros)
    positions = np.zeros(plant.num_positions())
    
    # For the floating base (quaternion_floating), set a valid quaternion [w, x, y, z]
    # Position 0-3 are the quaternion (w, x, y, z), 4-6 are xyz position
    if plant.num_positions() > 7:
        # Set quaternion to identity [1, 0, 0, 0] if floating base is present
        positions[0] = 1.0  # w component
        # x, y, z components are already 0
        # Position [4:7] are x, y, z position - keep at 0
    
    plant.SetPositions(plant_context, positions)
    
    return simulator, plant


def run_simulation(simulator, duration=10.0):
    """
    Run simulation for visualization.
    
    Args:
        simulator: The simulator instance
        duration: Simulation duration in seconds
    """
    print(f"\nRunning simulation for {duration} seconds...")
    
    # Run the simulation
    end_time = simulator.get_context().get_time() + duration
    
    while simulator.get_context().get_time() < end_time:
        # Advance simulation by fixed step
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
        
        # Show progress every second
        if int(simulator.get_context().get_time()) % 1 == 0:
            elapsed = simulator.get_context().get_time()
            progress = (elapsed / duration) * 100
            if progress <= 100:
                print(f"  Progress: {progress:6.1f}% ({elapsed:6.2f}s / {duration:6.2f}s)")
    
    print("Simulation completed!")


def main():
    """Main entry point"""
    # Get the path to the URDF file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "model", "robot.urdf")
    
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found at {urdf_path}")
        return
    
    print(f"Loading URDF from: {urdf_path}\n")
    
    # Create the simulator
    simulator, plant = create_robot_simulator(urdf_path)
    
    # Run the simulation
    run_simulation(simulator, duration=10.0)


if __name__ == "__main__":
    main()
