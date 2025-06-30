# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import csv
import math
import torch
import imageio
import gymnasium as gym
from pxr import UsdShade
from datetime import datetime

import omni.usd
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg
from omni.isaac.lab.sensors import Camera, CameraCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms, transform_points, quat_box_minus, quat_from_euler_xyz

##
# Pre-defined configs
##
from omni.isaac.lab_assets import RAYNOR_CFG, RECTANGLE_GATE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip


class RaynorEnvWindow(BaseEnvWindow):
    """Window manager for the Raynor environment."""

    def __init__(self, env: RaynorEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class RaynorEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 1
    action_space = 4
    observation_space = 41
    state_space = 0
    debug_vis = True
    sim_dt = 1/100
    log_data = False
    enable_camera = False
    keep_traversing = True
    random_stop = False # if True, robot will stop travering after traversing some gate
    # traversing task table
    # keep_traversing    random_stop       task
    #   False               False    1 gate traversing
    #   False               True     1 ~ n gate traversing (not fully tested)
    #   True                False    n gate traversing
    #   True                True     1 | n gate traversing (not fully tested)
    
    ui_window_class_type = RaynorEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    space_length = 50.0
    space_width = 50.0
    space_height = 3.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=max(space_length, space_width), replicate_physics=True)

    # robot
    robot: ArticulationCfg = RAYNOR_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_length = 0.15
    robot_width = 0.15
    robot_height = 0.1
    rotor_force_max = 2.0
    
    # sensors
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Gate/body",
        filter_prim_paths_expr = ["/World/envs/env_.*/Robot/body",
                                  "/World/envs/env_.*/Robot/rotor0",
                                  "/World/envs/env_.*/Robot/rotor1",
                                  "/World/envs/env_.*/Robot/rotor2",
                                  "/World/envs/env_.*/Robot/rotor3",
                                  "/World/envs/env_.*/Robot/camera",
                                  ],
        history_length=3,
        debug_vis=True,
    )
    camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/camera/Depth",
        data_types=["rgb", "depth"],
        spawn=None,
        width=256,
        height=192,
    )
    
    # rectanglular gate
    rectanglular_gate: RigidObjectCfg = RECTANGLE_GATE_CFG.replace(prim_path="/World/envs/env_.*/Gate")
    gate_width = 0.4
    gate_height = 0.3
    gate_range = {
        "x": (1.0, 2.0),  # x position range of the gate
        "y": (-1.0, 1.0),  # y position range of the gate
        "z": (0.5, 1.5),  # z position range of the gate
        "roll": (-torch.pi / 2.0, torch.pi / 2.0),  # roll angle range of the gate
        "pitch": (0.0, 0.0),  # pitch angle range of the gate
        "yaw": (0.0, 0.0),  # yaw angle range of the gate
        "vx": (-0.5, 0.5),  # x velocity range of the gate
        "vy": (0.0, 0.0),   # y velocity range of the gate
        "vz": (0.0, 0.0),   # z velocity range of the gate
        "omega_x": (-torch.pi / 2.0, torch.pi / 2.0),  # x angular velocity range of the gate
        "omega_y": (0.0, 0.0),    # y angular velocity range of the gate
        "omega_z": (0.0, 0.0),    # z angular velocity range of the gate
    }
    
    # goal position
    goal_range = {
        "x": (2.5, 3.0),  # x position range of the goal
        "y": (-0.5, 0.5),  # y position range of the goal
        "z": (0.5, 1.0),  # z position range of the goal
    }
    
    # constraints
    thrust_to_weight = 2.5
    x_moment_scale = 0.5 # moment scale for x-axis
    y_moment_scale = 0.5 # moment scale for y-axis
    z_moment_scale = 0.5 # moment scale for z-axis
    v_max = 3.0 # max velocity of the robot
    t_max = 5.0 # max time for one gate traversing
    reaching_threshold = math.sqrt(robot_length**2 + robot_width**2 + robot_height**2) # threshold for reaching the goal

    # reward factors
    alpha = 1.0 # sensitivity of the reward for approaching the gate
    beta = 5.0 # sensitivity of the reward for traversing the gate
    gamma = 10.0 # sensitivity of the reward for reaching the goal
    
    # penalty factors
    lambda_collision = 0.5 # penalty for collision with the gate
    lambda_time = 0.1 # penalty for time
    lambda_died = 5.0 # penalty for dying
    lambda_jerk = 5e-3 # penalty for jerk
    lambda_accel = 1e-3 # penalty for acceleration
    lambda_velocity = 0.1 # penalty for velocity
    lambda_rotation = 1e-3 # penalty for rotation
    
    # traversal length
    traversal_length = 2 * robot_length # length of the traversal
    traversal_width = gate_width - robot_width # width of the traversal
    traversal_height = gate_height - robot_height # height of the traversal
    
    # log directory for recording data
    log_dir: str = "/home/longbin/.local/share/ov/pkg/IsaacLab-1.4.1/logs/raynor_env"

class RaynorEnv(DirectRLEnv):
    cfg: RaynorEnvCfg

    def __init__(self, cfg: RaynorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Gate points position
        self._gate_points_pos_b = torch.zeros(self.num_envs, 4, 3, device=self.device)

        # Total thrust and moment applied to the base of the quadcopter
        self._last_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._stop_time = torch.zeros(self.num_envs, device=self.device)
        
        # Traversing status
        self._traversed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._died = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._collided = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._gate_moved = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._keep_traversing = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "approaching",
                "traversing",
                "reaching",
                "collision",
                "time",
                "died",
                "aggressive",
            ]
        }
        self.success_num = 0.0
        self.partial_success_num = 0.0
        self.total_num = 0.0
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        
        # data logging
        log_dir = self.cfg.log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._log_file = open(log_dir + datetime.now().strftime("/%Y-%m-%d_%H-%M-%S.csv"), "w", newline="")
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow([
            "env_id", "step",
            "x", "y", "z",
            "vx", "vy", "vz",
            # "ax", "ay", "az",
            "thrust", "moment_x", "moment_y", "moment_z"
        ])
        self._log_step = 0

    def _setup_scene(self):
        # Add robot to the scene
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        # Add rectanglular gate to the scene
        self._gate = RigidObject(self.cfg.rectanglular_gate)
        self.scene.rigid_objects["gate"] = self._gate
        # Add contact sensor to the scene
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # Add camera to the scene
        if self.cfg.enable_camera:
            self._camera = TiledCamera(self.cfg.camera)
            self.scene.sensors["camera"] = self._camera
        # Add terrain to the scene
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._last_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        desired_thrust = self._actions[:, 0]
        desired_moment = self._actions[:, 1:]
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (desired_thrust + 1.0) / 2.0
        self._moment[:, 0, 0] = self.cfg.x_moment_scale * desired_moment[:, 0]
        self._moment[:, 0, 1] = self.cfg.y_moment_scale * desired_moment[:, 1]
        self._moment[:, 0, 2] = self.cfg.z_moment_scale * desired_moment[:, 2]
        
        # reset the gate position if the robot has traversed the gate without collision
        mask = self._traversed & (~self._collided) & (~self._gate_moved) & self._keep_traversing
        env_ids = torch.nonzero(mask, as_tuple=False).view(-1)
        self.move_gate(env_ids) # move the gate to another position after traversing
        self._gate_moved[env_ids] = True
        # reset the goal position if the robot reaches the goal without collision
        is_reaching = self.is_reaching_goal(self._desired_pos_b)
        mask = (~self._collided) & is_reaching & self._keep_traversing
        env_ids = torch.nonzero(mask, as_tuple=False).view(-1)
        self.move_goal(env_ids) # move the goal to another position after reaching
        self._traversed[env_ids] = False
        self._gate_moved[env_ids] = False

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        last_gate_points_pos_b = self._gate_points_pos_b.clone()
        self._gate_points_pos_b = self.get_gate_points_pos(relative_to_robot=True) # shape: (N, 4, 3)
        self._desired_pos_b = self._desired_pos_w - self._robot.data.root_pos_w # shape: (N, 3)
        obs = torch.cat(
            [   
                self._gate_points_pos_b.reshape(-1, 12), # shape: (N, 12)
                self._desired_pos_b, # shape: (N, 3)
                self._robot.data.root_quat_w, # shape: (N, 4)
                self._robot.data.root_lin_vel_b, # shape: (N, 3)
                self._robot.data.root_ang_vel_b, # shape: (N, 3)
                last_gate_points_pos_b.reshape(-1, 12), # shape: (N, 12)
                self._actions, # shape: (N, 4)
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        self.record_camera()
        self.log_data()
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Pre-computation
        gate_points_pos_b = self.get_gate_points_pos(relative_to_robot=True) # shape: (N, 4, 3)
        gate_center = torch.mean(gate_points_pos_b, dim=1) # shape: (N, 3)
        x_proj, y_proj, z_proj = self.project_robot_pos(gate_points_pos_b) # shape: (N, 3)
        last_gate_center = torch.mean(self._gate_points_pos_b, dim=1) # shape: (N, 3)
        last_normal_gate = self.get_gate_normal(self._gate_points_pos_b) # shape: (N, 3)
        last_x_proj = torch.sum(-last_gate_center * last_normal_gate, dim=1) # shape: N
        last_goal_pos = self._desired_pos_b
        goal_pos = self._desired_pos_w - self._robot.data.root_pos_w
        
        # check traversing status
        is_traversing = self.is_traversing_gate(x_proj, last_x_proj, gate_points_pos_b, self._gate_points_pos_b) # check if the robot has traversed the gate
        first_traversing = is_traversing & (~self._traversed) # mask for first time traversing
        other_traversing = is_traversing & self._traversed # mask for traversing after the first time
        
        # reward for traversing
        reward_traversing = self.cfg.beta * (x_proj - last_x_proj + is_traversing.float())
        mask = (torch.abs(x_proj) > self.cfg.traversal_length) | (torch.abs(y_proj) >= self.cfg.traversal_width / 2) | (torch.abs(z_proj) >= self.cfg.traversal_height / 2) | (self._traversed)
        # # alternative reward for traversing (not good)
        # reward_traversing = self.cfg.beta * ((x_proj - last_x_proj) + first_traversing.float() - self.cfg.traversal_length * other_traversing.float()) # reward first traversing and penalize other traversing
        # mask = (torch.abs(x_proj) > self.cfg.traversal_length) | (torch.abs(y_proj) >= self.cfg.traversal_width / 2) | (torch.abs(z_proj) >= self.cfg.traversal_height / 2)
        reward_traversing[mask] = 0.0
        self._traversed = torch.logical_or(self._traversed, is_traversing) # update the traversed status
        
        # reward for approaching
        reward_approaching = self.cfg.alpha * (torch.linalg.norm(last_gate_center, dim=1) - torch.linalg.norm(gate_center, dim=1))
        mask = (x_proj >= -self.cfg.traversal_length) | (self._traversed)
        reward_approaching[mask] = 0.0
        
        # reward for reaching the goal
        reward_reaching = self.cfg.gamma * (torch.linalg.norm(last_goal_pos, dim=1) - torch.linalg.norm(goal_pos, dim=1))
        mask = ((x_proj <= self.cfg.traversal_length) | (~self._traversed)) & (~self._gate_moved)
        reward_reaching[mask] = 0.0
        
        # penalty for collision with the gate
        contact_force = torch.linalg.norm(self._contact_sensor.data.net_forces_w, dim=2).squeeze() # shape: (N,)
        penalty_collision = -self.cfg.lambda_collision * torch.ones_like(contact_force)
        colliding = (contact_force > 0.0)
        mask = ~colliding
        penalty_collision[mask] = 0.0
        self._collided = torch.logical_or(self._collided, colliding)
        
        # penalty for over time
        t_now = self.episode_length_buf / self.max_episode_length * self.cfg.episode_length_s
        self._keep_traversing = (t_now < self._stop_time) # check if the robot is keep traversing
        t_max = torch.ones_like(t_now) * self.cfg.t_max + self._stop_time
        penalty_time = -self.cfg.lambda_time * (t_now - t_max)
        mask = (t_now <= t_max) | (self._traversed)
        penalty_time[mask] = 0.0
        
        # penalty for dying
        penalty_died = -self.cfg.lambda_died * torch.ones_like(self._died, dtype=torch.float)
        mask = (~self._died) | (self._traversed)
        penalty_died[mask] = 0.0
        
        # penalty for aggressive motion
        v_drone = torch.linalg.norm(self._robot.data.root_lin_vel_w, dim=1)
        w_drone = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)
        is_reaching = self.is_reaching_goal(self._desired_pos_b)
        penalty_jerk = -self.cfg.lambda_jerk * torch.linalg.norm(self._actions - self._last_actions, dim=1)
        penalty_accel = -self.cfg.lambda_accel * torch.linalg.norm(self._actions, dim=1)
        penalty_velocity = 1 - torch.exp(self.cfg.lambda_velocity * (v_drone - self.cfg.v_max))
        mask = (v_drone <= self.cfg.v_max)
        penalty_velocity[mask] = 0.0
        penalty_rotation = 1 - torch.exp(self.cfg.lambda_rotation * w_drone)
        mask = ~is_reaching
        penalty_rotation[mask] = 0.0
        penalty_aggressive = penalty_jerk + penalty_accel + penalty_velocity + penalty_rotation
        
        rewards = {
            "approaching": reward_approaching,
            "traversing": reward_traversing,
            "reaching": reward_reaching,
            "collision": penalty_collision,
            "time": penalty_time,
            "died": penalty_died,
            "aggressive": penalty_aggressive,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Check if the robot has exceeded the space limits
        robot_pos = self._robot.data.root_pos_w - self._terrain.env_origins
        over_height = torch.logical_or(robot_pos[:, 2] < 0.1, robot_pos[:, 2] > self.cfg.space_height)
        over_length = torch.logical_or(robot_pos[:, 0] < -0.1, robot_pos[:, 0] > self.cfg.space_length)
        over_width = torch.logical_or(robot_pos[:, 1] < -self.cfg.space_width / 2.0, robot_pos[:, 1] > self.cfg.space_width / 2.0)
        # Only check the height for multi gate traversing
        self._died = over_height if (self.cfg.keep_traversing or self.cfg.random_stop) else (over_height | over_length | over_width)
    
        return self._died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        if len(env_ids) != self.num_envs:
            # for one gate traversing
            success = self._traversed[env_ids] & (~self._collided[env_ids]) & (~self._keep_traversing[env_ids])
            partial_success = self._traversed[env_ids] & (~self._keep_traversing[env_ids])
            # for multi gate traversing
            success = success | ((~self._died[env_ids]) & (~self._collided[env_ids]) & self._keep_traversing[env_ids])
            partial_success = partial_success | ((~self._died[env_ids]) & self._keep_traversing[env_ids])
            
            self.success_num += torch.count_nonzero(success).item()
            self.partial_success_num += torch.count_nonzero(partial_success).item()
            self.total_num += len(env_ids)
            success_rate = self.success_num / self.total_num
            partial_success_rate = self.partial_success_num / self.total_num
        else:
            success_rate = 0.0
            partial_success_rate = 0.0
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        extras["Metrics/success_rate"] = success_rate
        extras["Metrics/partial_success_rate"] = partial_success_rate
        self.extras["log"].update(extras)
        
        print(f"Success rate: {100*success_rate:.2f}%, Partial success rate: {100*partial_success_rate:.2f}%", end="\r")

        self._robot.reset(env_ids)
        self._gate.reset(env_ids)
        self._contact_sensor.reset(env_ids)
        if self.cfg.enable_camera:
            self._camera.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset the status
        self._last_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._traversed[env_ids] = False
        self._died[env_ids] = False
        self._collided[env_ids] = False
        self._gate_moved[env_ids] = False
        self._keep_traversing[env_ids] = self.cfg.keep_traversing | self.cfg.random_stop
        
        if self.cfg.random_stop:
            mask = (torch.rand_like(self._keep_traversing[env_ids], dtype=torch.float) < 0.5)
            mask = env_ids[mask]
            if self.cfg.keep_traversing:
                # same tasks are keep traversing, the other tasks are one gate traversing
                self._keep_traversing[mask] = False
            else:
                # randomly stop some traversing tasks
                self._stop_time[mask] =  torch.zeros_like(self._stop_time[mask]).uniform_(0.0, 1.0) * (self.cfg.episode_length_s - self.cfg.t_max)

        # Reset the stop time
        episode_time = self.episode_length_buf[env_ids] / self.max_episode_length * self.cfg.episode_length_s # Get the episode starting time
        self._stop_time[env_ids] = episode_time
        mask = self._keep_traversing[env_ids]
        self._stop_time[env_ids[mask]] = self.cfg.episode_length_s
        
        # Sample new goal
        self._desired_pos_w[env_ids] = self._generate_goal_pos(env_ids) # shape: (N, 3)
        self._desired_pos_w[env_ids, :3] += self._terrain.env_origins[env_ids, :3]
        self._desired_pos_b = self._desired_pos_w - self._robot.data.root_pos_w # shape: (N, 3)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_robot_state = self._robot.data.default_root_state[env_ids]
        default_robot_state[:, :3] += self._terrain.env_origins[env_ids]
        
        self._robot.write_root_pose_to_sim(default_robot_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_robot_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Reset gate state
        default_gate_state = self._generate_gate_state(env_ids) # shape: (N, 13)
        default_gate_state[:, :3] += self._terrain.env_origins[env_ids, :3]
        self._gate.write_root_pose_to_sim(default_gate_state[:, :7], env_ids)
        self._gate.write_root_velocity_to_sim(default_gate_state[:, 7:], env_ids)
        
        # Reset last gate points position
        self._gate_points_pos_b[env_ids] = self.get_gate_points_pos(gate_state=default_gate_state, robot_state=default_robot_state, relative_to_robot=True) # shape: (N, 4, 3)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        # update the gate color
        self.update_gate_color()
    
    def _generate_gate_state(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Generate a random state for the gate."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        gate_state = self._gate.data.default_root_state[env_ids]
        # Randomize the gate position and orientation
        gate_state[:, 0] = torch.zeros_like(gate_state[:, 0]).uniform_(*self.cfg.gate_range["x"])  # x position
        gate_state[:, 1] = torch.zeros_like(gate_state[:, 1]).uniform_(*self.cfg.gate_range["y"])  # y position
        gate_state[:, 2] = torch.zeros_like(gate_state[:, 2]).uniform_(*self.cfg.gate_range["z"])  # z position
        gate_state[:, 3:7] = quat_from_euler_xyz(
            roll=torch.zeros_like(gate_state[:, 4]).uniform_(*self.cfg.gate_range["roll"]),
            pitch=torch.zeros_like(gate_state[:, 5]).uniform_(*self.cfg.gate_range["pitch"]),
            yaw=torch.zeros_like(gate_state[:, 6]).uniform_(*self.cfg.gate_range["yaw"])
        )  # quaternion rotation
        gate_state[:, 7] = torch.zeros_like(gate_state[:, 7]).uniform_(*self.cfg.gate_range["vx"])  # linear velocity of x-axis
        gate_state[:, 8] = torch.zeros_like(gate_state[:, 8]).uniform_(*self.cfg.gate_range["vy"])  # linear velocity of y-axis
        gate_state[:, 9] = torch.zeros_like(gate_state[:, 9]).uniform_(*self.cfg.gate_range["vz"])  # linear velocity of z-axis
        gate_state[:, 10] = torch.zeros_like(gate_state[:, 10]).uniform_(*self.cfg.gate_range["omega_x"])  # angular velocity of x-axis
        gate_state[:, 11] = torch.zeros_like(gate_state[:, 11]).uniform_(*self.cfg.gate_range["omega_y"])  # angular velocity of y-axis
        gate_state[:, 12] = torch.zeros_like(gate_state[:, 12]).uniform_(*self.cfg.gate_range["omega_z"])  # angular velocity of z-axis
        
        return gate_state # shape: (N, 13)
    
    def _generate_goal_pos(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Generate a random position for the goal."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        goal_pos = self._desired_pos_w[env_ids]
        # Randomize the goal position
        goal_pos[:, 0] = torch.zeros_like(goal_pos[:, 0]).uniform_(*self.cfg.goal_range["x"])  # x position
        goal_pos[:, 1] = torch.zeros_like(goal_pos[:, 1]).uniform_(*self.cfg.goal_range["y"])  # y position
        goal_pos[:, 2] = torch.zeros_like(goal_pos[:, 2]).uniform_(*self.cfg.goal_range["z"])  # z position
        
        return goal_pos # shape: (N, 3)
    
    def get_gate_points_pos(self, gate_state: torch.Tensor|None = None, robot_state: torch.Tensor|None = None, relative_to_robot: bool = False) -> torch.Tensor:
        """Get the diagonal points of the gate."""
        
        # Get the world positions of the gate
        if gate_state is None:
            gate_pos_w = self._gate.data.root_pos_w
            gate_quat_w = self._gate.data.root_quat_w
        else:
            gate_pos_w = gate_state[:, :3]
            gate_quat_w = gate_state[:, 3:7]
        
        # Hardcoded the diagonal points of the gate
        half_width = self.cfg.gate_width / 2.0
        half_height = self.cfg.gate_height / 2.0
        top_left_pos_b = torch.tensor([0.0, -half_width, half_height], device=self.device)
        top_right_pos_b = torch.tensor([0.0, half_width, half_height], device=self.device)
        bottom_left_pos_b = torch.tensor([0.0, -half_width, -half_height], device=self.device)
        bottom_right_pos_b = torch.tensor([0.0, half_width, -half_height], device=self.device)
        points_pos_b = torch.stack([top_left_pos_b, top_right_pos_b, bottom_left_pos_b, bottom_right_pos_b])
        
        # Get the world positions of the diagonal points
        points_pos_w = transform_points(points_pos_b, gate_pos_w, gate_quat_w)
        if gate_pos_w.shape[0] == 1:
            # expand the batch dimension when batch size is 1
            points_pos_w.unsqueeze_(0)
        top_left_pos_w = points_pos_w[:, 0]
        top_right_pos_w = points_pos_w[:, 1]
        bottom_left_pos_w = points_pos_w[:, 2]
        bottom_right_pos_w = points_pos_w[:, 3]
        
        # Get the relative positions of the gate
        if relative_to_robot:
            if robot_state is None:
                robot_pos_w = self._robot.data.root_pos_w
                robot_quat_w = self._robot.data.root_quat_w
            else:
                robot_pos_w = robot_state[:, :3]
                robot_quat_w = robot_state[:, 3:7]
            top_left_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, top_left_pos_w)
            top_right_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, top_right_pos_w)
            bottom_left_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, bottom_left_pos_w)
            bottom_right_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, bottom_right_pos_w)
            return torch.stack([top_left_pos_b, top_right_pos_b, bottom_left_pos_b, bottom_right_pos_b], dim=1)
        else:
            return torch.stack([top_left_pos_w, top_right_pos_w, bottom_left_pos_w, bottom_right_pos_w], dim=1)

    def get_gate_normal(self, gate_points_pos, full_basis = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the normal vector of the gate plane."""
        # Get the diagonal points of the gate
        top_left_pos = gate_points_pos[:, 0]
        top_right_pos = gate_points_pos[:, 1]
        bottom_left_pos = gate_points_pos[:, 2]
        
        # Get the basis vectors of the gate
        basis_0 = bottom_left_pos - top_left_pos
        basis_1 = top_right_pos - top_left_pos
        
        # Normalize the basis vectors
        basis_0 /= torch.linalg.norm(basis_0, dim=1, keepdim=True)
        basis_1 /= torch.linalg.norm(basis_1, dim=1, keepdim=True)
        
        # Get the normal vector of the gate plane
        normal = torch.linalg.cross(basis_0, basis_1)
        normal /= torch.linalg.norm(normal, dim=1, keepdim=True)
        
        if full_basis:
            return normal, basis_0, basis_1
        else:
            return normal

    def project_robot_pos(self, gate_points_pos) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the projected robot position on the gate plane."""
        # Get the vector from the gate to the robot
        gate2robot = -torch.mean(gate_points_pos, dim=1) # shape: (N, 3)
        
        # Get the normal vector and basis of the gate plane
        normal_gate, basis_0, basis_1 = self.get_gate_normal(gate_points_pos, full_basis=True)
        
        # Project the robot position on the gate plane
        x = torch.sum(gate2robot * normal_gate, dim=1) # shape: (N,)
        y = torch.sum(gate2robot * basis_0, dim=1) # shape: (N,)
        z = torch.sum(gate2robot * basis_1, dim=1) # shape: (N,)
        
        return x, y, z
    
    def is_traversing_gate(self, x_proj: torch.Tensor, last_x_proj: torch.Tensor, gate_points_pos: torch.Tensor, last_gate_points_pos: torch.Tensor) -> torch.Tensor:
        """Check if the robot is traversing the gate in the current step."""
        # Check if the robot has crossed the gate plane
        has_crossed = (x_proj > 0.0) & (last_x_proj < 0.0) # shape: (N,)
        if has_crossed.sum() == 0:
            return has_crossed

        # Linear interpolation
        alpha = x_proj[has_crossed] / (x_proj[has_crossed] - last_x_proj[has_crossed] + 1e-6)
        mid_gate_points_pos = alpha.view(-1, 1, 1) * last_gate_points_pos[has_crossed] + (1 - alpha).view(-1, 1, 1) * gate_points_pos[has_crossed]
        
        # Check if the robot is inside the gate plane
        _, y, z = self.project_robot_pos(mid_gate_points_pos)
        is_inside = (torch.abs(y) < self.cfg.gate_width / 2.0) & (torch.abs(z) < self.cfg.gate_height / 2.0)
        
        # Check if the robot is traversing the gate
        is_traversing = torch.zeros_like(has_crossed)
        is_traversing[has_crossed] = is_inside
        
        return is_traversing
    
    def is_reaching_goal(self, goal_pos) -> torch.Tensor:
        """Check if the robot is reaching the goal."""
        distance_to_goal = torch.linalg.norm(goal_pos, dim=1)
        reaching_threshold = (torch.ones_like(distance_to_goal) + self._keep_traversing.float()) * self.cfg.reaching_threshold
        is_reaching = self._traversed & (distance_to_goal < reaching_threshold) # traversed and close to the goal
        return is_reaching
    
    def update_gate_color(self):
        """Update the color of the gate based on the traversing status."""
        stage = omni.usd.get_context().get_stage()
        gate_prim_paths = self._gate.root_physx_view.prim_paths
        for i, gate_prim_paths in enumerate(gate_prim_paths):
            # get the shader of the gate
            shader_path = gate_prim_paths.replace("body", "Looks/OmniSurfaceLite/Shader")
            shader_prim = stage.GetPrimAtPath(shader_path)
            shader = UsdShade.Shader(shader_prim)
            # green if traversed, red if not traversed
            moved = self._gate_moved[i].item()
            traversed = self._traversed[i].item()
            collided = self._collided[i].item()
            color = (float(not traversed or moved), float(traversed),float(collided))
            # traversed collided    color
            #   False    False   (1.0, 0.0, 0.0) # red
            #   False    True    (1.0, 0.0, 1.0) # magenta
            #   True     False   (0.0, 1.0, 0.0) # green
            #   True     True    (0.0, 1.0, 1.0) # cyan
            shader.GetInput("diffuse_reflection_color").Set(color)
    
    def move_gate_and_goal(self, env_ids: torch.Tensor | None):
        """Move the gate and goal to another position."""
        if env_ids is None or env_ids.shape[0] == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        elif env_ids.shape[0] == 0:
            return
            
        # move gate behind the goal
        self.move_gate(env_ids)
        
        # move the goal
        self.move_goal(env_ids)
        
    def move_gate(self, env_ids: torch.Tensor | None):
        """Move the gate to another position based on the goal position."""
        if env_ids is None or env_ids.shape[0] == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        elif env_ids.shape[0] == 0:
            return
            
        # Get the current state
        robot_state = self._robot.data.root_state_w[env_ids]
        goal_pos = self._desired_pos_w[env_ids]
        
        # Move the gate in front of the robot and randomize it
        gate_state = self._generate_gate_state(env_ids) # shape: (N, 13)
        gate_state[:, :2] += goal_pos[:, :2]
        
        # Set the new state of the gate
        self._gate_points_pos_b[env_ids] = self.get_gate_points_pos(gate_state=gate_state, robot_state=robot_state, relative_to_robot=True)
        self._gate.write_root_pose_to_sim(gate_state[:, :7], env_ids)
        self._gate.write_root_velocity_to_sim(gate_state[:, 7:], env_ids)
        
    def move_goal(self, env_ids: torch.Tensor | None):
        """Move the goal to another position based on current position."""
        if env_ids is None or env_ids.shape[0] == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        elif env_ids.shape[0] == 0:
            return
            
        # Store the current state
        last_goal_pos = self._desired_pos_w[env_ids].clone() 
        
        # Move the goal in front of the robot
        goal_pos = self._generate_goal_pos(env_ids) # shape: (N, 3)
        goal_pos[:, :2] += last_goal_pos[:, :2]
        
        # Set the new state of the gate
        self._desired_pos_w[env_ids] = goal_pos

    def record_camera(self):
        """Record the camera data and save it as a video."""
        if not self.cfg.enable_camera:
            return
        
        # show side-by-side RGB and depth images
        max_depth = 6.0
        min_depth = 0.52

        # fetch and tile RGB
        rgb_images = self._camera.data.output["rgb"].clone()  # (N, H, W, 3)
        N, H, W, C = rgb_images.shape
        grid = int(math.sqrt(N))
        assert grid * grid == N, f"env_num ({N}) is not a perfect square"
        rgb_tiles = (
            rgb_images
            .view(grid, grid, H, W, C)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(grid * H, grid * W, C)
        )

        # fetch and tile depth
        depth_images = self._camera.data.output["depth"].clone()  # (N, H, W) or (N, H, W, 1)
        if depth_images.dim() == 3:
            depth_images = depth_images.unsqueeze(-1)  # make (N, H, W, 1)
        _, Hd, Wd, Cd = depth_images.shape
        depth_tiles = (
            depth_images
            .view(grid, grid, Hd, Wd, Cd)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(grid * Hd, grid * Wd, Cd)
        )
        # clamp & normalize depth
        depth_tiles = depth_tiles.clamp(min=min_depth, max=max_depth) / max_depth * 255
        # convert to 3-channel for display
        if depth_tiles.shape[2] == 1:
            depth_tiles = depth_tiles.repeat(1, 1, 3)

        # concatenate RGB (left) and depth (right)
        frame_t = torch.cat([rgb_tiles, depth_tiles], dim=1)  # shape (grid*H, grid*W*2, 3)

        # ensure output directory exists
        out_dir = "/home/longbin/.local/share/ov/pkg/IsaacLab-1.4.1/logs/videos"
        os.makedirs(out_dir, exist_ok=True)

        # prepare the frame
        frame = frame_t.cpu().numpy().astype("uint8")

        # initialize writer on first call
        if not hasattr(self, "video_writer"):
            video_path = os.path.join(out_dir, "run.mp4")
            self.video_writer = imageio.get_writer(video_path, fps=30, codec="libx264")

        # append frame to video
        self.video_writer.append_data(frame)
    
    def log_data(self):
        """Log the data to a CSV file."""
        # Data logging
        if self.cfg.log_data and self._log_file is not None:
            for env_idx in range(self.num_envs):
                self._log_writer.writerow([
                    env_idx, self._log_step,
                    self._robot.data.root_pos_w[env_idx, 0].item() - self._terrain.env_origins[env_idx, 0].item(),
                    self._robot.data.root_pos_w[env_idx, 1].item() - self._terrain.env_origins[env_idx, 1].item(),
                    self._robot.data.root_pos_w[env_idx, 2].item() - self._terrain.env_origins[env_idx, 2].item(),
                    self._robot.data.root_lin_vel_b[env_idx, 0].item(),
                    self._robot.data.root_lin_vel_b[env_idx, 1].item(),
                    self._robot.data.root_lin_vel_b[env_idx, 2].item(),
                    self._thrust[env_idx, 0, 2].item(),
                    self._moment[env_idx, 0, 0].item(),
                    self._moment[env_idx, 0, 1].item(),
                    self._moment[env_idx, 0, 2].item(),
                ])
            self._log_step += 1
    
    def close(self):
        """Close the csv writer"""
        try:
            self._log_file.close()
        except Exception as e:
            print(f"Error closing log file: {e}")
        super().close()