# SDL_xarm

ROS2 workspace for xArm7 motion experiments.

## Packages

| Package | Description |
|---|---|
| `testapp` | Circle motion demo and connection test for xArm7 |
| `xarm_ros2` | Upstream xArm ROS2 driver (submodule) |

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules <repo-url> sdl_ws
cd sdl_ws
```

### 2. Enable required xArm API services

The xArm driver disables most services by default. Create the user override file to enable the ones needed for the connection test and collision monitoring:

```bash
cat > src/xarm_ros2/xarm_api/config/xarm_user_params.yaml << 'EOF'
ufactory_driver:
  ros__parameters:
    services:
      get_state: true
      get_err_warn_code: true
      get_position: true
EOF
```

> Without this, `/xarm/get_state`, `/xarm/get_err_warn_code`, and `/xarm/get_position` will not be available.

### 3. Build

```bash
cd sdl_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Configuration

### circle_motion.py

Edit [`src/testapp/testapp/circle_motion.py`](src/testapp/testapp/circle_motion.py) before running:

**Initial joint angles** — the robot moves here first before approaching the circle:
```python
angles_deg = [0, -45, 0, 45, 0, 90, 0]   # change to suit your setup
```
Set `INITIAL_JOINT_ANGLES = None` to skip homing and start from the current pose.

**Circle geometry** (metres):
```python
CIRCLE_CENTER_X = 0.30
CIRCLE_CENTER_Y = 0.00
CIRCLE_CENTER_Z = 0.30
CIRCLE_RADIUS   = 0.08
N_WAYPOINTS     = 36     # 10° steps
```

**Collision sensitivity** (real robot only, 1 = least sensitive, 5 = most):
```python
COLLISION_SENSITIVITY = 5
```

**End-effector orientation** — default is gripper pointing straight down (RPY = π, 0, 0):
```python
ORIENT_X, ORIENT_Y, ORIENT_Z, ORIENT_W = 1.0, 0.0, 0.0, 0.0
```

## Usage

### Check robot connection

```bash
# Terminal 1 — launch driver
ros2 launch testapp circle_motion_realmove.launch.py robot_ip:=<IP>

# Terminal 2 — verify connection
ros2 run testapp robot_connection_test
```

### Run circle motion

**Simulation (fake controllers):**
```bash
ros2 launch testapp circle_motion_fake.launch.py
```

**Real robot:**
```bash
ros2 launch testapp circle_motion_realmove.launch.py robot_ip:=<IP>
```

> **WARNING:** The real robot launch moves physical hardware. Clear the workspace and keep the E-stop within reach.

### Clear robot errors

```bash
ros2 service call /xarm/clean_error xarm_msgs/srv/Call
```
