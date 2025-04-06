# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
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
    decimation = 2
    action_space = 4
    observation_space = 29
    state_space = 0
    debug_vis = True
    sim_dt = 1/100

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = RAYNOR_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 2.5
    moment_scale = 0.5
    v_max = 1.0
    
    # rectanglular gate
    rectanglular_gate: RigidObjectCfg = RECTANGLE_GATE_CFG.replace(prim_path="/World/envs/env_.*/Gate")

    # reward factors
    alpha = 1.0 # sensitivity of the reward for approaching the gate
    beta = 1.0 # sensitivity of the reward for traversing the gate
    gamma = 1.0 # sensitivity of the reward for reaching the goal
    
    # penalty factors
    lambda_jerk = 0.1 # penalty for jerk
    lambda_accel = 0.1 # penalty for acceleration
    lambda_velocity = 0.1 # penalty for velocity
    
    # traversal length
    traversal_length = 0.25 # length of the traversal
    traversal_width = 0.1 # width of the traversal


class RaynorEnv(DirectRLEnv):
    cfg: RaynorEnvCfg

    def __init__(self, cfg: RaynorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._last_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "approaching",
                "traversing",
                "reaching",
                # "aggressive_motion",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        # Add robot to the scene
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        # Add rectanglular gate to the scene
        self._gate = RigidObject(self.cfg.rectanglular_gate)
        self.scene.rigid_objects["gate"] = self._gate

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
        self._actions = actions.clone().clamp(-1.0, 1.0)
        desired_thrust = self._actions[:, 0]
        desired_moment = self._actions[:, 1:]
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (desired_thrust + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * desired_moment

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        self._gate_points_pos_b = self.get_gate_points_pos(relative_to_robot=True) # shape: (N, 4, 3)
        self._desired_pos_b = self._desired_pos_w - self._robot.data.root_pos_w # shape: (N, 3)
        obs = torch.cat(
            [
                self._gate_points_pos_b.reshape(-1, 12), # shape: (N, 12)
                self._desired_pos_b, # shape: (N, 3)
                self._robot.data.root_quat_w, # shape: (N, 4)
                self._robot.data.root_lin_vel_b, # shape: (N, 3)
                self._robot.data.root_ang_vel_b, # shape: (N, 3)
                self._actions, # shape: (N, 4)
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # reward for approaching 
        last_gate_center = torch.mean(self._gate_points_pos_b, dim=1) # shape: (N, 3)
        gate_center = torch.mean(self.get_gate_points_pos(relative_to_robot=True), dim=1) # shape: (N, 3)
        normal_gate = self.get_gate_normal() # shape: (N, 3)
        x_proj = torch.sum(-gate_center * normal_gate, dim=1) # shape: N
        reward_approaching = self.cfg.alpha * (torch.linalg.norm(last_gate_center, dim=1) - torch.linalg.norm(gate_center, dim=1))
        mask = (x_proj >= -self.cfg.traversal_length)
        reward_approaching[mask] = 0.0
        
        # reward for traversing
        last_x_proj = torch.sum(-last_gate_center * normal_gate, dim=1) # shape: N
        y_proj = torch.linalg.norm(gate_center + x_proj.unsqueeze(-1) * normal_gate, dim=1) # shape: N
        reward_traversing = self.cfg.beta * (x_proj - last_x_proj)
        mask = (torch.abs(x_proj) > self.cfg.traversal_length) | (y_proj > self.cfg.traversal_width)
        reward_traversing[mask] = 0.0
        
        # reward for reaching the goal
        last_goal_pos = self._desired_pos_b
        goal_pos = self._desired_pos_w - self._robot.data.root_pos_w
        reward_reaching = self.cfg.gamma * (torch.linalg.norm(last_goal_pos, dim=1) - torch.linalg.norm(goal_pos, dim=1))
        mask = (x_proj <= self.cfg.traversal_length)
        reward_reaching[mask] = 0.0
        
        
        # # penalty for aggressive motion
        # penalty_jerk = self.cfg.lambda_jerk * torch.sum(
        #     torch.abs(self._actions - self._last_actions) / self.cfg.sim.dt, dim=1
        # )
        # penalty_accel = self.cfg.lambda_accel * torch.sum(torch.abs(self._actions), dim=1)
        # v_drone = torch.linalg.norm(self._robot.data.root_lin_vel_w, dim=1)
        # penalty_velocity = torch.exp(self.cfg.lambda_velocity * (v_drone - self.cfg.v_max)) - 1
        # penalty_velocity[v_drone <= self.cfg.v_max] = 0.0
        # # penalty_aggressive_motion = penalty_jerk + penalty_accel + penalty_velocity
        # penalty_aggressive_motion = penalty_accel + penalty_velocity
        # self._last_actions = self._actions.clone() # update the last actions
        
        rewards = {
            "approaching": reward_approaching,
            "traversing": reward_traversing,
            "reaching": reward_reaching,
            # "aggressive_motion": -penalty_aggressive_motion,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 3.0)
        gate_move = torch.linalg.norm(self._gate.data.root_lin_vel_w, dim=1) > 0.5
        died = torch.logical_or(died, gate_move)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
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
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        self._gate.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, 0] = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(2.0, 3.0)
        self._desired_pos_w[env_ids, 1] = torch.zeros_like(self._desired_pos_w[env_ids, 1]).uniform_(-1.0, 1.0)
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        self._desired_pos_w[env_ids, :3] += self._terrain.env_origins[env_ids, :3]
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_robot_state = self._robot.data.default_root_state[env_ids]
        default_robot_state[:, :3] += self._terrain.env_origins[env_ids]
        
        self._robot.write_root_pose_to_sim(default_robot_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_robot_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Reset gate state
        default_gate_state = self._gate.data.default_root_state[env_ids]
        default_gate_state[:, 0] = torch.zeros_like(default_gate_state[:, 0]).uniform_(0.5, 1.5) # x-axis position
        default_gate_state[:, 1] = torch.zeros_like(default_gate_state[:, 1]).uniform_(-1.0, 1.0) # y-axis position
        default_gate_state[:, 2] = torch.zeros_like(default_gate_state[:, 2]).uniform_(0.5, 1.5) # z-axis position
        default_gate_state[:, :3] += self._terrain.env_origins[env_ids, :3]
        default_gate_state[:, 3:7] = quat_from_euler_xyz(   roll=torch.zeros_like(default_gate_state[:, 4]).uniform_(-torch.pi/2, torch.pi/2), 
                                                            pitch=torch.zeros_like(default_gate_state[:, 5]),
                                                            yaw=torch.zeros_like(default_gate_state[:, 6]))
        default_gate_state[:, 8] = torch.zeros_like(default_gate_state[:, 8]).uniform_(-0.5, 0.5) # linear velocity of y-axis
        default_gate_state[:, 10] = torch.zeros_like(default_gate_state[:, 10]).uniform_(-torch.pi/2, torch.pi/2) # angular velocity of x-axis
        self._gate.write_root_pose_to_sim(default_gate_state[:, :7], env_ids)
        self._gate.write_root_velocity_to_sim(default_gate_state[:, 7:], env_ids)

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
        
    def get_gate_points_pos(self, relative_to_robot: bool = False) -> torch.Tensor:
        """Get the diagonal points of the gate."""
        
        # Get the world positions of the gate
        gate_pos_w = self._gate.data.root_pos_w.squeeze()
        gate_quat_w = self._gate.data.root_quat_w.squeeze()
        
        # Hardcoded the diagonal points of the gate
        top_left_pos_b = torch.tensor([0.0, -0.2, 0.15], device=self.device)
        top_right_pos_b = torch.tensor([0.0, 0.2, 0.15], device=self.device)
        bottom_left_pos_b = torch.tensor([0.0, -0.2, -0.15], device=self.device)
        bottom_right_pos_b = torch.tensor([0.0, 0.2, -0.15], device=self.device)
        points_pos_b = torch.stack([top_left_pos_b, top_right_pos_b, bottom_left_pos_b, bottom_right_pos_b])
        
        # Get the world positions of the diagonal points
        points_pos_w = transform_points(points_pos_b, gate_pos_w, gate_quat_w)
        top_left_pos_w = points_pos_w[:, 0]
        top_right_pos_w = points_pos_w[:, 1]
        bottom_left_pos_w = points_pos_w[:, 2]
        bottom_right_pos_w = points_pos_w[:, 3]
        
        # Get the relative positions of the gate
        if relative_to_robot:
            robot_pos_w = self._robot.data.root_pos_w
            robot_quat_w = self._robot.data.root_quat_w
            top_left_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, top_left_pos_w)
            top_right_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, top_right_pos_w)
            bottom_left_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, bottom_left_pos_w)
            bottom_right_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, bottom_right_pos_w)
            return torch.stack([top_left_pos_b, top_right_pos_b, bottom_left_pos_b, bottom_right_pos_b], dim=1)
        else:
            return torch.stack([top_left_pos_w, top_right_pos_w, bottom_left_pos_w, bottom_right_pos_w], dim=1)

    def get_gate_normal(self) -> torch.Tensor:
        """Get the normal vector of the gate plane."""
        # Get the diagonal points of the gate
        top_left_pos = self._gate_points_pos_b[:, 0]
        top_right_pos = self._gate_points_pos_b[:, 1]
        bottom_left_pos = self._gate_points_pos_b[:, 2]
        
        # Get the basis vectors of the gate
        basis_0 = bottom_left_pos - top_left_pos
        basis_1 = top_right_pos - top_left_pos
        
        # Normalize the basis vectors
        basis_0 /= torch.linalg.norm(basis_0, dim=1, keepdim=True)
        basis_1 /= torch.linalg.norm(basis_1, dim=1, keepdim=True)
        
        # Get the normal vector of the gate plane
        normal = torch.linalg.cross(basis_0, basis_1)
        normal /= torch.linalg.norm(normal, dim=1, keepdim=True)
        
        return normal

    def get_projected_robot_pos(self) -> torch.Tensor:
        """Get the projected robot position on the gate plane."""
        # Get the vector from the gate to the robot
        robot_pos_w = self._robot.data.root_pos_w
        gate_pos_w = self._gate.data.root_pos_w
        gate2robot = robot_pos_w - gate_pos_w
        
        # Get the normal vector of the gate plane
        normal = self.get_gate_normal()
        
        # Project the robot position on the gate plane
        robot_pos_projected = robot_pos_w - torch.sum(gate2robot * normal, dim=1, keepdim=True) * normal
        
        return robot_pos_projected