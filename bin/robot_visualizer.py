#!/usr/bin/env python3
"""
Robot URDF Visualization and Manipulation with Drake and MeshCat

This script demonstrates how to load, visualize, and interactively manipulate 
a robot URDF file using Drake simulator and MeshCat visualization.
"""

import numpy as np
import sys
from pathlib import Path

# Drake imports
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.geometry import StartMeshcat
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer, MeshcatVisualizerParams
from pydrake.systems.analysis import Simulator


class RobotVisualizer:
    """A class to handle robot visualization and manipulation with Drake and MeshCat."""
    
    def __init__(self, urdf_path="model/robot.urdf"):
        """
        Initialize the robot visualizer.
        
        Args:
            urdf_path: Path to the robot URDF file
        """
        self.urdf_path = urdf_path
        self.plant = None
        self.plant_copy = None
        self.diagram = None
        self.simulator = None
        self.meshcat = None
        self.joint_names = []
        self.joint_limits = {}
        self.num_q = 0
        
        print("=" * 60)
        print("Robot URDF Visualization with Drake and MeshCat")
        print("=" * 60)
        
    def load_urdf(self):
        """Load the URDF file into Drake."""
        print("\n[Step 1] Loading URDF File...")
        
        # Create a MultibodyPlant and SceneGraph
        self.plant = MultibodyPlant(time_step=0.0)
        scene_graph = self.plant.scene_graph()
        
        # Load the robot URDF file
        parser = Parser(self.plant)
        
        try:
            robot_model = parser.AddModelFromFile(self.urdf_path)
            print(f"  ✓ Successfully loaded robot from: {self.urdf_path}")
        except Exception as e:
            print(f"  ✗ Error loading URDF: {e}")
            print(f"  Make sure the robot.urdf file exists at: {self.urdf_path}")
            sys.exit(1)
        
        # Finalize the plant
        self.plant.Finalize()
        
        # Get plant information
        self.num_q = self.plant.num_positions()
        num_v = self.plant.num_velocities()
        
        print(f"\n  Robot Configuration:")
        print(f"    • Number of generalized positions: {self.num_q}")
        print(f"    • Number of generalized velocities: {num_v}")
        print(f"    • Number of bodies: {self.plant.num_bodies()}")
        print(f"    • Number of actuators: {self.plant.num_actuators()}")
        
        # Get joint information
        print(f"\n  Joints in the robot:")
        for joint_idx in range(self.plant.num_joints()):
            joint = self.plant.get_joint(joint_idx)
            self.joint_names.append(joint.name())
            print(f"    [{joint_idx}] {joint.name()} ({type(joint).__name__})")
    
    def setup_meshcat(self):
        """Set up and start the MeshCat visualizer."""
        print("\n[Step 2] Setting Up MeshCat Visualizer...")
        
        try:
            self.meshcat = StartMeshcat()
            print(f"  ✓ MeshCat started successfully")
            print(f"  ✓ Open http://localhost:7000 in your browser to view the robot")
        except Exception as e:
            print(f"  ✗ Error starting MeshCat: {e}")
            sys.exit(1)
    
    def display_robot(self):
        """Display the robot in MeshCat."""
        print("\n[Step 3] Displaying Robot in MeshCat...")
        
        # Create a diagram and add the plant
        builder = DiagramBuilder()
        self.plant_copy = AddMultibodyPlantSceneGraph(builder, self.plant.time_step())
        
        # Parse the URDF again for the copy
        parser_copy = Parser(self.plant_copy.plant)
        parser_copy.AddModelFromFile(self.urdf_path)
        self.plant_copy.plant.Finalize()
        
        # Add visualizer
        meshcat_params = MeshcatVisualizerParams(delete_on_initialization_event=False)
        visualizer = builder.AddSystem(
            MeshcatVisualizer(self.meshcat, params=meshcat_params)
        )
        
        # Connect geometry query output to visualizer
        builder.Connect(
            self.plant_copy.scene_graph.get_query_output_port(),
            visualizer.GetInputPort("geometry_query")
        )
        
        # Build the diagram
        self.diagram = builder.Build()
        
        # Create a simulator
        self.simulator = Simulator(self.diagram)
        
        print("  ✓ Robot successfully displayed in MeshCat!")
    
    def configure_joint_controls(self):
        """Configure joint limits and control interface."""
        print("\n[Step 4] Configuring Joint Controls...")
        
        for joint_idx in range(self.plant.num_joints()):
            joint = self.plant.get_joint(joint_idx)
            joint_name = joint.name()
            
            # Get joint limits
            if hasattr(joint, 'position_lower_limits'):
                lower = joint.position_lower_limits()
                upper = joint.position_upper_limits()
                if len(lower) > 0 and len(upper) > 0:
                    self.joint_limits[joint_name] = (lower[0], upper[0])
                else:
                    self.joint_limits[joint_name] = (-np.pi, np.pi)
            else:
                self.joint_limits[joint_name] = (-np.pi, np.pi)
        
        print(f"  ✓ Configured {len(self.joint_limits)} joint limits")
        for joint_name, (lower, upper) in self.joint_limits.items():
            print(f"    {joint_name}: [{lower:.3f}, {upper:.3f}]")
    
    def update_robot_pose(self, joint_positions):
        """
        Update the robot pose based on joint positions.
        
        Args:
            joint_positions: Array of joint positions
        """
        context = self.diagram.CreateDefaultContext()
        plant_context = self.plant_copy.plant.GetMyContextFromRoot(context)
        
        # Set plant positions
        self.plant_copy.plant.SetPositions(plant_context, joint_positions)
        
        # Update visualization
        self.simulator.Publish(context)
    
    def get_current_pose(self):
        """Get current robot pose."""
        context = self.diagram.CreateDefaultContext()
        plant_context = self.plant_copy.plant.GetMyContextFromRoot(context)
        return self.plant_copy.plant.GetPositions(plant_context)
    
    def set_pose(self, q):
        """Set robot to specified configuration."""
        if len(q) != self.num_q:
            print(f"Error: Expected {self.num_q} joint positions, got {len(q)}")
            return False
        self.update_robot_pose(q)
        return True
    
    def reset_pose(self):
        """Reset all joints to zero."""
        self.update_robot_pose(np.zeros(self.num_q))
        print("  ✓ Robot reset to zero configuration")
    
    def print_pose(self):
        """Print current joint positions."""
        q = self.get_current_pose()
        print("\n  Current Joint Positions:")
        for i, joint_name in enumerate(self.joint_names):
            if i < self.num_q:
                print(f"    {joint_name}: {q[i]:8.4f}")
    
    def animate_trajectory(self, trajectory_func, num_steps=50):
        """
        Animate the robot using a trajectory function.
        
        Args:
            trajectory_func: Function that takes time (0 to 1) and returns joint positions
            num_steps: Number of animation frames
        """
        print(f"\n  Animating robot ({num_steps} frames)...")
        for i in range(num_steps):
            t = i / num_steps  # Normalized time [0, 1]
            joint_positions = trajectory_func(t)
            self.update_robot_pose(joint_positions)
        print("  ✓ Animation completed")
    
    def interactive_mode(self):
        """Enter interactive mode for manual joint manipulation."""
        print("\n" + "=" * 60)
        print("Interactive Mode - Manual Joint Manipulation")
        print("=" * 60)
        
        q = np.zeros(self.num_q)
        
        while True:
            print("\nOptions:")
            print("  1. Adjust joint angle")
            print("  2. Print current pose")
            print("  3. Reset to zero")
            print("  4. Show joint limits")
            print("  5. Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                try:
                    joint_idx = int(input(f"Enter joint index (0-{self.num_q-1}): "))
                    if 0 <= joint_idx < self.num_q:
                        joint_name = self.joint_names[joint_idx]
                        lower, upper = self.joint_limits[joint_name]
                        
                        angle = float(input(f"Enter angle [{lower:.3f}, {upper:.3f}]: "))
                        if lower <= angle <= upper:
                            q[joint_idx] = angle
                            self.update_robot_pose(q)
                            print(f"  ✓ Joint {joint_idx} set to {angle:.4f}")
                        else:
                            print(f"  ✗ Angle out of range [{lower:.3f}, {upper:.3f}]")
                    else:
                        print(f"  ✗ Invalid joint index")
                except ValueError:
                    print("  ✗ Invalid input")
            
            elif choice == "2":
                self.print_pose()
            
            elif choice == "3":
                q = np.zeros(self.num_q)
                self.update_robot_pose(q)
                self.reset_pose()
            
            elif choice == "4":
                print("\n  Joint Limits:")
                for i, (joint_name, (lower, upper)) in enumerate(self.joint_limits.items()):
                    print(f"    [{i}] {joint_name}: [{lower:.3f}, {upper:.3f}]")
            
            elif choice == "5":
                print("\n  Exiting interactive mode...")
                break
            
            else:
                print("  ✗ Invalid option")
    
    def demo_oscillation(self):
        """Demonstrate robot movement with oscillating first joint."""
        print("\n[Demo] Oscillating first joint...")
        
        def oscillation_trajectory(t):
            q = np.zeros(self.num_q)
            q[0] = np.sin(2 * np.pi * t) * 0.5
            return q
        
        self.animate_trajectory(oscillation_trajectory, num_steps=100)
    
    def run_full_setup(self):
        """Run the complete setup procedure."""
        self.load_urdf()
        self.setup_meshcat()
        self.display_robot()
        self.configure_joint_controls()
        
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print(f"\nAccess the robot visualization at: http://localhost:7000")
        print(f"\nRobot has {self.num_q} configurable joints.")


def main():
    """Main function to run the robot visualizer."""
    # Create visualizer instance
    visualizer = RobotVisualizer(urdf_path="model/robot.urdf")
    
    # Run setup
    visualizer.run_full_setup()
    
    # Menu loop
    while True:
        print("\n" + "=" * 60)
        print("Main Menu")
        print("=" * 60)
        print("1. Enter interactive mode (manual joint control)")
        print("2. Run demo (oscillating first joint)")
        print("3. Print current pose")
        print("4. Reset to zero pose")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            visualizer.interactive_mode()
        
        elif choice == "2":
            visualizer.demo_oscillation()
        
        elif choice == "3":
            visualizer.print_pose()
        
        elif choice == "4":
            visualizer.reset_pose()
        
        elif choice == "5":
            print("\nExiting...")
            break
        
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
