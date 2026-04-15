from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    map_yaml   = LaunchConfiguration("map_yaml")
    checkpoint = LaunchConfiguration("checkpoint")
    thr_scale  = LaunchConfiguration("throttle_scale")

    return LaunchDescription([
        DeclareLaunchArgument("map_yaml",       default_value="maps/my_map.yaml"),
        DeclareLaunchArgument("checkpoint",     default_value="checkpoints/agent_final.pt"),
        DeclareLaunchArgument("throttle_scale", default_value="1.0"),

        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[{"yaml_filename": map_yaml, "topic_name": "/map", "frame_id": "map"}],
        ),
        Node(
            package="nav2_amcl",
            executable="amcl",
            name="amcl",
            output="screen",
            parameters=[{
                "global_frame_id": "map",
                "odom_frame_id": "world",
                "base_frame_id": "roboracer_1",
                "scan_topic": "/autodrive/roboracer_1/lidar",
                "min_particles": 500,
                "max_particles": 3000,
                "laser_model_type": "likelihood_field",
                "laser_likelihood_max_dist": 2.0,
                "laser_max_range": 10.0,
                "laser_min_range": 0.06,
                "max_beams": 60,
                "z_hit": 0.7,
                "z_rand": 0.3,
                "sigma_hit": 0.15,
                "robot_model_type": "nav2_amcl::DifferentialMotionModel",
                "alpha1": 0.15, "alpha2": 0.15, "alpha3": 0.15,
                "alpha4": 0.15, "alpha5": 0.1,
                "update_min_a": 0.1,
                "update_min_d": 0.1,
                "tf_broadcast": True,
                "set_initial_pose": True,
                "initial_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            }],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager",
            output="screen",
            parameters=[{
                "autostart": True,
                "node_names": ["map_server", "amcl"],
                "bond_timeout": 0.0,
            }],
        ),
        Node(
            package="racing_agent",
            executable="agent_node",
            name="racing_agent",
            output="screen",
            parameters=[{
                "map_yaml":       map_yaml,
                "checkpoint":     checkpoint,
                "throttle_scale": thr_scale,
            }],
        ),
    ])