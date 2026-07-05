# Drake Robot Simulator - Complete Guide

This package contains multiple Python scripts to simulate your robot using Drake and MeshCat visualization. Choose the script that best fits your needs.

## 📋 Quick Reference

| Script | Purpose | Complexity | Use When |
|--------|---------|------------|----------|
| `quick_simulate.py` | Minimal working example | ⭐ Simple | Just want to see robot load |
| `simulate_robot.py` | Full-featured simulator | ⭐⭐ Medium | Want a solid baseline simulator |
| `advanced_controller.py` | PD joint control | ⭐⭐⭐ Advanced | Need joint trajectory control |
| `logging_simulator.py` | Data recording | ⭐⭐⭐ Advanced | Need to analyze simulation data |

---

## 🚀 Getting Started (5 minutes)

### 1. Install Drake

```bash
pip install drake
```

Or run the provided script:
```bash
bash install_dependencies.sh
```

### 2. Run the Quick Example

```bash
python quick_simulate.py
```

You'll see terminal output with a MeshCat URL (usually `http://localhost:7000`). Open this in your browser.

---

## 📖 Script Descriptions

### 1. `quick_simulate.py` - Minimal Example

**Best for:** Learning Drake basics, quick verification

**What it does:**
- Loads robot.urdf
- Creates MeshCat visualizer
- Runs for 60 seconds
- No motion (static display)

**Usage:**
```bash
python quick_simulate.py
```

**Customization:**
```python
simulator.AdvanceTo(60)  # Change to duration in seconds
```

---

### 2. `simulate_robot.py` - Full Simulator

**Best for:** Main simulation work, starting point for custom work

**What it does:**
- Loads robot.urdf with full diagnostic output
- Prints robot structure (bodies, joints, actuators)
- Displays MeshCat visualization
- Applies sinusoidal joint motion for 10 seconds
- Runs at real-time speed

**Usage:**
```bash
python simulate_robot.py
```

**Output example:**
```
Robot loaded successfully!
Number of bodies: 5
Number of joints: 4
Number of actuators: 0

Joints:
  - 旋转 2: continuous
  - 旋转 3: continuous
  - 旋转 5: continuous
  - 旋转 6: continuous

Running simulation for 10 seconds...
Simulation completed!
```

**Customization:**

Change simulation duration:
```python
run_simulation_with_motion(simulator, plant, diagram, duration=20.0)
```

Change joint motion pattern:
```python
amplitude = np.pi / 2  # 90 degrees instead of 45
frequency = 1.0        # Faster motion (1 Hz)
```

Modify which joints move:
```python
for i in range(plant.num_positions()):  # All joints now
    positions[i] = initial_state[i] + amplitude * np.sin(2 * np.pi * frequency * t)
```

---

### 3. `advanced_controller.py` - Joint Control

**Best for:** Trajectory tracking, closed-loop control, motion planning

**What it does:**
- Implements a PD (Proportional-Derivative) joint controller
- Applies smooth trajectory setpoints to joints
- Visualizes controlled motion in MeshCat
- Runs for 20 seconds

**Key Features:**
- Smooth sinusoidal motion on Joint 0
- Triangle wave motion on Joint 1
- PD control with configurable gains

**Usage:**
```bash
python advanced_controller.py
```

**Customization:**

Change control gains:
```python
self.Kp = 20.0  # Higher = more stiff response
self.Kd = 3.0   # Higher = more damped response
```

Define custom trajectories:
```python
def trajectory_func(t):
    positions = np.zeros(num_joints)
    positions[0] = np.sin(2 * np.pi * 0.5 * t)  # Your custom motion
    return positions
```

---

### 4. `logging_simulator.py` - Data Recording

**Best for:** Analysis, debugging, performance validation

**What it does:**
- Records joint positions and velocities at each time step
- Saves data to `simulation_data.csv`
- Displays real-time progress
- Provides statistical analysis after simulation
- Runs for 15 seconds

**Output files:**
- `simulation_data.csv` - Raw timestamped data

**Usage:**
```bash
python logging_simulator.py
```

**Output example:**
```
SIMULATION DATA ANALYSIS
============================================================

Simulation Duration: 15.00 seconds
Data Points Logged: 15000
Average Frequency: 1000.0 Hz

Position Statistics (radians):
  Joint 0: min=-0.5000, max= 0.5000, mean= 0.0000
  Joint 1: min= 0.0000, max= 0.5000, mean= 0.2500

Velocity Statistics (rad/s):
  Joint 0: min=-3.1416, max= 3.1416, mean= 0.0000
  Joint 1: min=-0.2500, max= 0.2500, mean= 0.0000

============================================================
Data saved to: simulation_data.csv
```

**CSV Format:**
```
time,q0,q1,q2,q3,v0,v1,v2,v3
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
0.001,0.001,0.0,0.0,0.0,1.0,0.0,0.0,0.0
...
```

**Analyzing the data:**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('simulation_data.csv')
plt.plot(df['time'], df['q0'])
plt.xlabel('Time (s)')
plt.ylabel('Joint 0 Position (rad)')
plt.show()
```

---

## 🔧 Common Customizations

### Add Joint Limits

In `simulate_robot.py`:
```python
# In run_simulation_with_motion function
joint_limits = [np.pi, np.pi, np.pi/2, 2*np.pi]
for i in range(min(len(joint_limits), plant.num_positions())):
    positions[i] = np.clip(positions[i], -joint_limits[i], joint_limits[i])
```

### Change Simulation Time Step

In any script:
```python
plant = builder.AddSystem(MultibodyPlant(time_step=0.0001))  # Finer time step
```

### Add Gravity

By default gravity is enabled. To disable:
```python
plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
```

### Visualize at Different Resolution

```python
meshcat_params = MeshcatVisualizerParams()
meshcat_params.role = "illustration"  # Faster but less detailed
# or
meshcat_params.role = "proximity"     # Show collision geometry
```

---

## 🎓 Learning Progression

1. **Start here:** `quick_simulate.py` - Get familiar with the pipeline
2. **Add motion:** `simulate_robot.py` - See the robot move
3. **Add control:** `advanced_controller.py` - Close the loop
4. **Analyze:** `logging_simulator.py` - Quantify performance

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'pydrake'"
```bash
pip install drake
```

### "FileNotFoundError: model/robot.urdf"
- Ensure you're running from the Drake directory
- Check the path is correct relative to your working directory

### MeshCat visualizer not showing
1. Check console for URL (usually `http://localhost:7000`)
2. Open URL in web browser
3. If port 7000 is blocked, Drake will use 7001, 7002, etc.
4. Check firewall settings

### Simulation runs but robot doesn't move
- Check `simulate_robot.py` - uses passive motion (no controllers)
- Use `advanced_controller.py` for active control
- Verify `plant.num_positions() > 0` (check console output)

### Slow visualization
- Reduce simulation duration
- Close other applications
- Use `role = "illustration"` instead of `"proximity"`

---

## 📚 Drake Documentation

- **Main Documentation:** https://drake.mit.edu/
- **Python API:** https://drake.mit.edu/python_bindings.html
- **Multibody Systems:** https://drake.mit.edu/doxygen_cxx/group__multibody.html
- **MeshCat Visualizer:** https://github.com/rdeits/meshcat

---

## 💡 Next Steps

1. **Modify trajectories** - Edit trajectory functions in the scripts
2. **Add controllers** - Implement your control laws
3. **Multi-robot simulation** - Load multiple URDFs into one diagram
4. **Export animations** - Record and save motion videos
5. **Real-world integration** - Connect to actual robot hardware

---

## 📝 File Structure

```
Drake/
├── robot.urdf                    # Robot definition
├── quick_simulate.py             # Minimal example
├── simulate_robot.py             # Full simulator
├── advanced_controller.py        # PD control
├── logging_simulator.py          # Data logging
├── requirements.txt              # Dependencies
├── install_dependencies.sh       # Installation script
├── README.md                     # Installation guide
├── USAGE_GUIDE.md               # This file
└── model/
    └── robot.urdf               # (Link to your robot)
```

---

Generated for Drake robot simulation with MeshCat visualization
