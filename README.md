# Rustoracer

## Devcontainer setup

Rebuild and reopen in container. This installs all dependencies including ROS 2 Jazzy, torch, and the AutoDrive devkit.

## Sim (learnrl)

```bash
uv run --with torch --with tensordict --with tqdm --with tyro --with "wandb[media]" --with av scripts/learnrl.py
```

With memory profiling:
```bash
uv run --with torch --with tensordict --with tqdm --with tyro --with "wandb[media]" --with av --with memray \
  memray run -o output.bin scripts/learnrl.py
```

## Real car (Roboracer)

### 1. Start devkit
```bash
cd scripts/autodrive_devkit && colcon build && source install/setup.bash
ros2 launch autodrive_roboracer bringup_headless.launch.py
```

### 2. Start Foxglove
```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

### 3. Map the track (first time only)
```bash
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=autodrive_online_async_roboracer.yaml
```

Drive around with keyboard control to build the map:
```bash
python3 scripts/autodrive_devkit/autodrive_roboracer/teleop_keyboard.py
```

Save the map:
```bash
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap \
  "{name: {data: '/workspaces/rustoracer/maps/my_map'}}"
```

### 4. Run the RL agent
```bash
cd racing_agent && colcon build && source install/setup.bash
ros2 launch racing_agent racing.launch.py \
  map_yaml:=/workspaces/rustoracer/maps/my_map.yaml \
  checkpoint:=/workspaces/rustoracer/checkpoints/agent_final.pt \
  throttle_scale:=1.0
```

## Other controllers

Wall follow:
```bash
cd wall_follow && colcon build && source install/setup.bash
ros2 launch wall_follow wall_follow.launch.py
```

Disparity extender:
```bash
cd disparity_extender && colcon build && source install/setup.bash
ros2 launch disparity_extender disparity_extender.launch.py
```
