#!/usr/bin/env python3
"""
Advanced Drake Robot Controller
Example with joint control, trajectory tracking, and data logging
"""

import numpy as np
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Parser,
    SceneGraph,
    Simulator,
    BasicVector,
    VectorSystem,
    LeafSystem,
    MeshcatVisualizer,
)
import os


class SimpleJointController(LeafSystem):
    """
    Simple PD-like joint controller for smooth motion.
    """
    def __init__(self, plant, target_trajectory_func):
        """
        Args:
            plant: MultibodyPlant
            target_trajectory_func: Function(t) -> desired_positions
        """
        super().__init__()
        
        self.plant = plant
        self.target_trajectory_func = target_trajectory_func
        self.num_joints = plant.num_positions()
        
        # Input: current state from plant
        self.state_port = self.DeclareVectorInputPort("state", BasicVector(2 * self.num_joints))
        
        # Output: control torques
        self.control_port = self.DeclareVectorOutputPort("control", BasicVector(self.num_joints),
                                                          self.CalcControlOutput)
        
        # Controller gains
        self.Kp = 10.0  # Proportional gain
        self.Kd = 2.0   # Derivative gain
    
    def CalcControlOutput(self, context, output):
        # Get current state
        state = self.state_port.Eval(context)
        q = state[:self.num_joints]  # Positions
        v = state[self.num_joints:]  # Velocities
        
        # Get target positions
        t = context.get_time()
        q_target = self.target_trajectory_func(t)
        
        # Simple PD control
        position_error = q_target - q
        control = self.Kp * position_error - self.Kd * v
        
        # Limit control magnitude
        control = np.clip(control, -5.0, 5.0)
        
        output.SetFromVector(control)


def create_controlled_simulator(urdf_path):
    """
    Create simulator with joint controller.
    """
    builder = DiagramBuilder()
    
    scene_graph = builder.AddSystem(SceneGraph())
    plant = builder.AddSystem(MultibodyPlant(time_step=0.001))
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    
    parser = Parser(plant)
    parser.AddModels(urdf_path)
    plant.Finalize()
    
    # Add visualizer
    meshcat_visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph)
    
    # Define trajectory function
    def trajectory_func(t):
        """Generate smooth joint trajectory"""
        num_joints = plant.num_positions()
        positions = np.zeros(num_joints)
        
        # Circular motion for first joint
        amplitude = 0.5  # radians
        frequency = 0.25  # Hz
        positions[0] = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Triangle wave for second joint
        if num_joints > 1:
            period = 4.0
            t_normalized = (t % period) / period
            if t_normalized < 0.5:
                positions[1] = t_normalized * 2 * amplitude
            else:
                positions[1] = (1 - t_normalized) * 2 * amplitude
        
        return positions
    
    # Add controller
    controller = SimpleJointController(plant, trajectory_func)
    builder.AddSystem(controller)
    
    # Connect state output to controller input
    builder.Connect(plant.get_state_output_port(), controller.state_port)
    
    # Connect controller output to plant input
    for i in range(plant.num_actuators()):
        builder.Connect(controller.control_port, plant.get_actuation_input_port())
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    
    return simulator, plant


def run_controlled_simulation(simulator, plant, duration=30.0):
    """Run simulation with active control"""
    print(f"Running controlled simulation for {duration} seconds...")
    
    # Get initial context and set starting position
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)
    
    # Set all positions to zero initially
    plant.SetPositions(plant_context, np.zeros(plant.num_positions()))
    plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
    
    # Run simulation
    end_time = simulator.get_context().get_time() + duration
    while simulator.get_context().get_time() < end_time:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)
        
        # Print periodic status
        if int(simulator.get_context().get_time()) % 5 == 0:
            print(f"Simulation time: {simulator.get_context().get_time():.1f}s")
    
    print("Controlled simulation completed!")


def main():
    """Main entry point"""
    urdf_path = "model/robot.urdf"
    
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found at {urdf_path}")
        return
    
    print("Drake Robot Controller - Advanced Example\n")
    print("Features:")
    print("- Loads robot from URDF")
    print("- Implements PD joint controller")
    print("- Applies smooth trajectories to joints")
    print("- Visualizes with MeshCat\n")
    
    simulator, plant = create_controlled_simulator(urdf_path)
    
    print("Robot structure:")
    print(f"  Bodies: {plant.num_bodies()}")
    print(f"  Joints: {plant.num_joints()}")
    print(f"  Positions: {plant.num_positions()}")
    print(f"  Velocities: {plant.num_velocities()}\n")
    
    run_controlled_simulation(simulator, plant, duration=20.0)


if __name__ == "__main__":
    main()
