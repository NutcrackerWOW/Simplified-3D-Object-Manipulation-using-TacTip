#!/usr/bin/env python3
"""
Drake Robot Simulator with Data Logging
Records joint positions, velocities, and torques during simulation
"""

import numpy as np
import csv
import os
from datetime import datetime
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Parser,
    SceneGraph,
    Simulator,
    BasicVector,
    LeafSystem,
    MeshcatVisualizer,
)


class DataLogger(LeafSystem):
    """System that logs robot state data during simulation"""
    
    def __init__(self, plant, output_file="simulation_data.csv"):
        super().__init__()
        self.plant = plant
        self.output_file = output_file
        self.data = []
        
        # Input: robot state
        state_size = 2 * plant.num_positions()  # positions + velocities
        self.state_port = self.DeclareVectorInputPort("state", BasicVector(state_size))
        
        # Open CSV file
        self.csv_file = open(output_file, 'w', newline='')
        self.csv_writer = None
    
    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        """Log data at each time step"""
        state = self.state_port.Eval(context)
        t = context.get_time()
        
        q = state[:self.plant.num_positions()]
        v = state[self.plant.num_positions():]
        
        row = [t] + list(q) + list(v)
        self.data.append(row)
        
        if self.csv_writer is None:
            # Write header
            header = ['time']
            for i in range(self.plant.num_positions()):
                header.append(f'q{i}')
            for i in range(self.plant.num_velocities()):
                header.append(f'v{i}')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(header)
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()
    
    def close(self):
        """Close the CSV file"""
        if self.csv_file:
            self.csv_file.close()
    
    def get_data(self):
        """Get logged data as numpy array"""
        return np.array(self.data)


def create_logging_simulator(urdf_path, log_file="simulation_data.csv"):
    """Create simulator with data logging"""
    builder = DiagramBuilder()
    
    scene_graph = builder.AddSystem(SceneGraph())
    plant = builder.AddSystem(MultibodyPlant(time_step=0.001))
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    
    parser = Parser(plant)
    parser.AddModels(urdf_path)
    plant.Finalize()
    
    # Add visualizer
    meshcat_visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph)
    
    # Add logger
    logger = DataLogger(plant, log_file)
    builder.AddSystem(logger)
    
    # Connect state output
    builder.Connect(plant.get_state_output_port(), logger.state_port)
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    
    return simulator, plant, logger


def run_logged_simulation(simulator, plant, logger, duration=15.0):
    """Run simulation with data logging"""
    print(f"Running simulation with data logging for {duration} seconds...\n")
    
    # Get context and set initial state
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, np.zeros(plant.num_positions()))
    plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))
    
    # Run simulation
    start_time = datetime.now()
    end_time = simulator.get_context().get_time() + duration
    
    while simulator.get_context().get_time() < end_time:
        current_time = simulator.get_context().get_time()
        
        # Simple sinusoidal actuation
        t = current_time
        for i in range(plant.num_actuators()):
            # You could apply forces here
            pass
        
        simulator.AdvanceTo(current_time + 0.01)
        
        # Progress indicator
        progress = (current_time / duration) * 100
        if int(progress) % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"Progress: {progress:6.1f}% - Sim time: {current_time:7.2f}s - Wall time: {elapsed:6.1f}s")
    
    logger.close()
    print("\nSimulation complete!")
    return logger.get_data()


def analyze_logged_data(data, num_positions):
    """Analyze and print statistics from logged data"""
    print("\n" + "="*60)
    print("SIMULATION DATA ANALYSIS")
    print("="*60)
    
    times = data[:, 0]
    positions = data[:, 1:1+num_positions]
    velocities = data[:, 1+num_positions:]
    
    print(f"\nSimulation Duration: {times[-1] - times[0]:.2f} seconds")
    print(f"Data Points Logged: {len(data)}")
    print(f"Average Frequency: {len(data)/(times[-1] - times[0]):.1f} Hz")
    
    print("\nPosition Statistics (radians):")
    for i, pos_data in enumerate(positions.T):
        print(f"  Joint {i}: min={np.min(pos_data):8.4f}, max={np.max(pos_data):8.4f}, mean={np.mean(pos_data):8.4f}")
    
    print("\nVelocity Statistics (rad/s):")
    for i, vel_data in enumerate(velocities.T):
        print(f"  Joint {i}: min={np.min(vel_data):8.4f}, max={np.max(vel_data):8.4f}, mean={np.mean(vel_data):8.4f}")


def main():
    """Main entry point"""
    urdf_path = "model/robot.urdf"
    log_file = "simulation_data.csv"
    
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found at {urdf_path}")
        return
    
    print("Drake Robot Simulator with Data Logging\n")
    print(f"Loading robot from: {urdf_path}")
    print(f"Data will be saved to: {log_file}\n")
    
    # Create and run simulator
    simulator, plant, logger = create_logging_simulator(urdf_path, log_file)
    
    print(f"Robot has {plant.num_positions()} positions and {plant.num_velocities()} velocities")
    print("-" * 60)
    
    # Run simulation
    data = run_logged_simulation(simulator, plant, logger, duration=15.0)
    
    # Analyze results
    analyze_logged_data(data, plant.num_positions())
    
    print("\n" + "="*60)
    print(f"Data saved to: {log_file}")
    print("="*60)


if __name__ == "__main__":
    main()
