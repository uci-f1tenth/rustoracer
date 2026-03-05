# Install instructions
### learnrl.py 
```bash
uv run --with torch --with tensordict --with tqdm --with tyro --with wandb[media] scripts/learnrl.py
```
### learnrl.py w/ CUDA
```bash
uv run --with torch --with tensordict --with tqdm --with tyro --with wandb[media] --index https://download.pytorch.org/whl/cu130 scripts/learnrl.py
```
Make sure you have CUDA installed! :P
### autodrive learnrl
```bash
uv run --python 3.10 scripts/learnrl_autodrive.py
```
## Devcontainer
Rebuild and reopen in container
## Starting up scripts
```bash
cargo run --release --features ros
```
```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```
```bash
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=autodrive_online_async.yaml
```
```bash
cd wall_follow && colcon build && cd ..
source wall_follow/install/setup.bash
ros2 launch wall_follow wall_follow.launch.py
```
```bash
cd disparity_extender && colcon build && cd ..
source disparity_extender/install/setup.bash
ros2 launch disparity_extender disparity_extender.launch.py
```
# Roboracer install
## Start devkit
```bash
cd scripts/autodrive_devkit/
colcon build
source install/setup.bash
ros2 launch autodrive_roboracer bringup_headless.launch.py
```
## Start foxglove
```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```
## Start slam_tooblox
```bash
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=autodrive_online_async_roboracer.yaml
```
## Start wall follow
```bash
cd scripts/wall_follow
colcon build
source install/setup.bash
ros2 launch wall_follow wall_follow.launch.py
```
## Save map
```bash
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: '/workspaces/rustoracer/my_map'}}"
```