# Install instructions

### learnrl_jax.py (pure JAX — fastest)
Single-file rewrite using JAX + Optax + scipy/skimage.  Everything — physics,
environment, and PPO training — runs as one compiled XLA program on the GPU.
Zero CPU↔GPU copies per step; `lax.scan` compiles the full rollout loop; `vmap`
parallelises all N environments natively.

CPU:
```bash
uv run --with "jax[cpu]" --with optax --with scipy --with scikit-image \
       --with pillow --with pyyaml --with tqdm --with tyro \
       scripts/learnrl_jax.py
```
GPU (CUDA 12):
```bash
uv run --with "jax[cuda12]" --with optax --with scipy --with scikit-image \
       --with pillow --with pyyaml --with tqdm --with tyro \
       scripts/learnrl_jax.py
```

### learnrl.py (Rust sim + PyTorch)
```bash
uv run --with torch --with tensordict --with tqdm --with tyro --with wandb[media] --with av scripts/learnrl.py
```
### learnrl.py memray tracking
```bash
uv run --with torch --with tensordict --with tqdm --with tyro --with wandb[media] --with av --with memray memray run -o output.bin scripts/learnrl.py
```
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
## Keyboard control
```bash
python3 scripts/autodrive_devkit/autodrive_roboracer/teleop_keyboard.py
```
## Save map
```bash
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: '/workspaces/rustoracer/my_map'}}"
```
