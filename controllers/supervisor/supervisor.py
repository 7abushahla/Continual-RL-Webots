import os
import numpy as np
import gym
from gym import spaces
from deepbots.supervisor import CSVSupervisorEnv
from deepbots.supervisor.wrappers import KeyboardPrinter
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
import datetime
import json

# ====== Manual Experiment Configuration ======
MODE = "evaluation"         # "training" or "evaluation"
TASK = "maze"          # "obstacle", "line", "maze"
LOAD_FROM = "maze"         # None, "obstacle", "line", "maze"
CHECKPOINT_STEP = None  # e.g., 1000, 2000, ... or None

def normalize_to_range(value, min, max, new_min, new_max, clip=False):
    """Normalize a value from one range to another"""
    value = float(value)
    min = float(min)
    max = float(max)
    new_min = float(new_min)
    new_max = float(new_max)
    if clip:
        value = max if value > max else value
        value = min if value < min else value
    return (new_max - new_min) / (max - min) * (value - max) + new_max


def get_distance_from_target(robot, target):
    """Calculate Euclidean distance between robot and target"""
    robot_pos = robot.getPosition()
    target_pos = target.getPosition()
    return np.sqrt(sum((robot_pos[i] - target_pos[i])**2 for i in range(3)))


def get_angle_from_target(robot, target, is_abs=True):
    """Calculate angle between robot and target (Z-up convention, XY plane)"""
    robot_pos = robot.getPosition()
    target_pos = target.getPosition()
    
    # Calculate vector to target in XY plane
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    angle_to_target = np.arctan2(dy, dx)
    
    # Get robot's yaw (rotation around Z)
    robot_rot = robot.getOrientation()
    robot_yaw = np.arctan2(robot_rot[1], robot_rot[0])
    
    # Calculate relative angle
    relative_angle = angle_to_target - robot_yaw
    
    # Normalize angle to [-pi, pi]
    while relative_angle > np.pi:
        relative_angle -= 2 * np.pi
    while relative_angle < -np.pi:
        relative_angle += 2 * np.pi
        
    return np.abs(relative_angle) if is_abs else relative_angle


def get_heading_vector(robot):
    """Get the robot's heading vector (forward direction) in world coordinates."""
    orientation = robot.getOrientation()  # 3x3 rotation matrix, row-major
    # Webots: X is right, Y is up, Z is forward
    # Forward vector is the third column of the rotation matrix
    # orientation = [xx, xy, xz, yx, yy, yz, zx, zy, zz]
    # Forward (Z) = (xz, yz, zz)
    return np.array([orientation[2], orientation[5], orientation[8]])


class TrainingCallback(BaseCallback):
    """Custom callback for saving models and logging"""
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.last_save_step = 0

    def _on_step(self):
        # print(f"[DEBUG] TrainingCallback step: {self.n_calls}")
        # Save checkpoint at regular intervals
        if self.n_calls % self.check_freq == 0:
            # Save model and buffer with step count
            step_count = self.n_calls
            checkpoint_path = f"{self.save_path}/checkpoint_{step_count}"
            os.makedirs(checkpoint_path, exist_ok=True)
            model_path = f"{checkpoint_path}/model"
            self.model.save(model_path)
            print(f"\nSaved checkpoint at step {step_count} to {model_path}")
            if hasattr(self.model, 'replay_buffer'):
                buffer_path = f"{checkpoint_path}/replay_buffer.pkl"
                self.model.save_replay_buffer(buffer_path)
                print(f"Saved replay buffer at step {step_count} to {buffer_path}")
            self.last_save_step = step_count
        return True


class SuccessRateCallback(BaseCallback):
    def _on_rollout_end(self):
        infos = self.locals.get("infos", [])
        if infos:
            sr = np.mean([info.get("is_success", 0.0) for info in infos])
            self.logger.record("rollout/success_rate", sr)
    def _on_step(self):
        return True


class PerEpisodeRewardCallback(BaseCallback):
    """Logs the raw episodic return (reward per episode) to TensorBoard as 'custom/ep_rew_raw'."""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Check if a new episode has ended
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    ep_rew = info['episode']['r']
                    self.logger.record('custom/ep_rew_raw', ep_rew)
        return True


class UnifiedSupervisor(CSVSupervisorEnv):
    def __init__(self):
        # Call parent init first
        super().__init__(emitter_name='emitter', receiver_name='receiver')
        
        # Debug print all communication channels
        print("\n=== Communication Channels ===")
        # RL Supervisor channels
        self.emitter = self.getDevice('emitter')
        self.receiver = self.getDevice('receiver')
        print(f"[RL Supervisor] Main emitter channel = {self.emitter.getChannel()}")
        print(f"[RL Supervisor] Main receiver channel = {self.receiver.getChannel()}")
        
        # Initialize basic attributes that aren't in parent class
        self.steps = 0
        self.max_steps = 500
        self.message = []
        self.should_done = False
        self.is_solved = False
        
        # Initialize previous values
        self.previous_distance = None  # Changed to None for first step handling
        self.previous_angle = 0.0
        self.previous_prox_values = np.zeros(8)
        
        # Get robot node
        self.robot = self.getFromDef('robot')
        if self.robot is None:
            raise ValueError("Robot node not found in simulation")
            
        # Get target node (for obstacle and maze tasks)
        self.target = self.getFromDef('TARGET')
        
        # Get PATH group and its children (waypoints)
        self.path_group = self.getFromDef('PATH')
        self.path_nodes = []
        if self.path_group is not None:
            children_field = self.path_group.getField('children')
            if children_field is not None:
                for i in range(children_field.getCount()):
                    node = children_field.getMFNode(i)
                    if node is not None:
                        self.path_nodes.append(node.getPosition())
        print(f"Loaded {len(self.path_nodes)} path waypoints from PATH group.")
        
        # Store initial position and rotation (axis-angle for correct reset)
        self.initial_position = self.robot.getPosition()
        self.initial_rotation = self.robot.getField('rotation').getSFRotation()
        
        # Setup gym spaces
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(13,),  # 8 proximity + 3 ground sensors + distance + angle
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, 
            shape=(2,),   # Gas and wheel commands
            dtype=np.float32
        )
        
        # Initialize SAC (but don't create model yet)
        self._model = None
        self.next_path_node_idx = 0  # For sparse path rewards
        self.path_node_threshold = 0.15  # Same as target reach threshold
        self.dense_waypoint_weight = 1.0  # Weight for dense proximity to next waypoint
        self.line_episode_reward = 0.0

    def setup_sac(self, load_previous_task=None, checkpoint_step=None):
        """Initialize or load SAC model — now using a Monitor-wrapped env and explicit logger."""
        # ensure logs/ exists
        os.makedirs("logs", exist_ok=True)
        
        run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        tb_path = os.path.join("logs", "tb", f"{self.current_task}_{run_id}")  # Unique folder per task+run
        os.makedirs(tb_path, exist_ok=True)
        print(f"Logging to {tb_path}")
        monitor_path = os.path.join(tb_path, f"monitor_{self.current_task}.csv")

        # wrap this supervisor in a Monitor (logs reward, length, is_success)
        monitored_env = Monitor(self,
                                filename=monitor_path,
                                info_keywords=("is_success",))
        vec_env = DummyVecEnv([lambda: monitored_env])

        # Always create a fresh logger for this run
        new_logger = configure(tb_path, ["stdout", "tensorboard"])

        if load_previous_task:
            model_path  = f"logs/sac_model_{load_previous_task}_final.zip"
            buffer_path = f"logs/replay_buffer_{load_previous_task}_final.pkl"
            print(f"DEBUG: Model path: {os.path.abspath(model_path)} (exists: {os.path.exists(model_path)})")
            print(f"DEBUG: Buffer path: {os.path.abspath(buffer_path)} (exists: {os.path.exists(buffer_path)})")
            print(f"DEBUG: Observation space before loading: {self.observation_space}")
            if os.path.exists(model_path):
                print(f"\n=== Loading model from previous task: {load_previous_task} ===")
                self._model = SAC.load(model_path, env=vec_env, tensorboard_log=None)
                print(f"DEBUG: Model loaded: {model_path}")
                print(f"DEBUG: Model policy: {type(self._model.policy)}")
                print(f"DEBUG: Observation space after loading: {self.observation_space}")
                self._model.set_logger(new_logger)
                if os.path.exists(buffer_path):
                    self._model.load_replay_buffer(buffer_path)
                    self._model.replay_buffer.max_size = 2_000_000
            else:
                print(f"No previous model found for task: {load_previous_task}")
                self._create_new_model(vec_env, new_logger)

        elif checkpoint_step is not None:
            cp_dir      = f"logs/checkpoint_{checkpoint_step}"
            model_path  = f"{cp_dir}/model.zip"
            buffer_path = f"{cp_dir}/replay_buffer.pkl"
            if os.path.exists(model_path):
                print(f"\n=== Loading checkpoint from step {checkpoint_step} ===")
                self._model = SAC.load(model_path, env=vec_env, tensorboard_log=None)
                self._model.set_logger(new_logger)
                if os.path.exists(buffer_path):
                    self._model.load_replay_buffer(buffer_path)
            else:
                print(f"Checkpoint not found at step {checkpoint_step}, creating new model")
                self._create_new_model(vec_env, new_logger)

        else:
            print("\n=== Creating new model from scratch ===")
            self._create_new_model(vec_env, new_logger)

    def _create_new_model(self, env, logger):
        """Create a new SAC model on the given env and set logger."""
        print("Creating new SAC model...")
        self._model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            verbose=1,
            tensorboard_log=None
        )
        self._model.set_logger(logger)

    @property
    def model(self):
        return self._model

    def get_observations(self):
        """Get normalized sensor readings"""
        self.message = self.handle_receiver()
        
        if self.message is not None and len(self.message) >= 8:  # At least 8 proximity sensors
            # Normalize proximity sensors [0, 1023] -> [0, 1]
            prox_sensors = [normalize_to_range(float(self.message[i]), 0, 1023, 0, 1) for i in range(8)]
            
            # Get ground sensor values if available
            ground_sensors = [0.0, 0.0, 0.0]  # Default values
            if len(self.message) >= 11:  # If we have ground sensors
                ground_sensors = [normalize_to_range(float(self.message[i]), 0, 1000, 0, 1) for i in range(8, 11)]
            
            # Initialize target-related observations
            distance = 0.0
            angle = 0.0
            
            # Only get target information for obstacle task
            if self.current_task == "obstacle" and hasattr(self, 'target'):
                distance = get_distance_from_target(self.robot, self.target)
                angle = get_angle_from_target(self.robot, self.target, is_abs=False)
                
                # Normalize distance [0, 1.5] -> [0, 1]
                distance = normalize_to_range(distance, 0, 1.5, 0, 1)
                
                # Normalize angle [-pi, pi] -> [0, 1]
                angle = normalize_to_range(angle, -np.pi, np.pi, 0, 1)
            
            # Print debug info every 50 steps
            if self.steps % 50 == 0:
                print("\n=== Debug Info ===")
                if self.current_task == "obstacle":
                    print(f"Distance to target: {distance:.3f}")
                    print(f"Angle to target: {angle:.3f}")
                print(f"Proximity sensors: {[f'{s:.3f}' for s in prox_sensors]}")
                if len(self.message) >= 11:
                    print(f"Ground sensors: {[f'{s:.3f}' for s in ground_sensors]}")
                # Get current action if available
                if hasattr(self, '_model') and self._model is not None:
                    obs = np.array(prox_sensors + ground_sensors + [distance, angle], dtype=np.float32)
                    action, _ = self._model.predict(obs, deterministic=True)
                    print(f"Action: gas={action[1]:.3f}, wheel={action[0]:.3f}")
            
            return np.array(prox_sensors + ground_sensors + [distance, angle], dtype=np.float32)
        
        return np.zeros(13, dtype=np.float32)

    def get_default_observation(self):
        """Return default observation when no message is received"""
        return np.zeros(13, dtype=np.float32)

    def get_reward(self, action):
        """Calculate reward based on current task"""
        # Store the action for reward breakdown
        self._last_action = action
        
        reward = 0.0
        if self.current_task == "obstacle":
            reward = self._get_obstacle_reward(action)
        elif self.current_task == "line":
            reward = self._get_line_reward()
            self.line_episode_reward += reward  # Accumulate reward for line task
        elif self.current_task == "maze":
            reward = self._get_maze_reward(action)
        
        # Print progress every 50 steps (consistent with maze/obstacle)
        if self.steps % 50 == 0 and self.steps > 0:
            print(f"[REWARD] Task: {self.current_task}, Step: {self.steps}, Reward: {reward:.2f}, Episode total: {self.line_episode_reward:.2f}")
        
        # # Add a small time penalty every step
        # reward -= 0.01
        
        # Penalize being too close to obstacles (walls) using proximity sensors
        # if self.message is not None and len(self.message) >= 8:
        #     prox_values = [float(self.message[i]) for i in range(8)]
        #     # If any proximity sensor is above a threshold (e.g., 800), penalize
        #     if max(prox_values) > 800:
        #         reward -= 0.5  # Strong penalty for being very close to a wall
        #         if self.steps % 50 == 0:
        #             print("Penalty: -0.5 (Very close to wall, prox sensor > 800)")
        #     # If gliding along wall (moderately high proximity), smaller penalty
        #     elif max(prox_values) > 500:
        #         reward -= 0.2
        #         if self.steps % 50 == 0:
        #             print("Penalty: -0.2 (Gliding along wall, prox sensor > 500)")
        
        return reward

    def _get_obstacle_reward(self, action):
        # --- Setup and state ---
        # If episode was just reset, set previous and current target distance and angle
        if not hasattr(self, 'just_reset'):
            self.just_reset = (self.previous_distance is None)
        if self.just_reset:
            self.previous_tar_d = self.current_tar_d = get_distance_from_target(self.robot, self.target)
            self.previous_tar_a = self.current_tar_a = get_angle_from_target(self.robot, self.target, is_abs=False)
        else:
            self.previous_tar_d = getattr(self, 'previous_tar_d', get_distance_from_target(self.robot, self.target))
            self.current_tar_d = get_distance_from_target(self.robot, self.target)
            self.previous_tar_a = getattr(self, 'previous_tar_a', get_angle_from_target(self.robot, self.target, is_abs=False))
            self.current_tar_a = get_angle_from_target(self.robot, self.target, is_abs=False)

        # --- Distance to target reward ---
        # Base reward for current distance (higher reward when closer)
        dist_tar_reward = 1.0 - normalize_to_range(self.current_tar_d, 0.0, 1.5, 0.0, 1.0, clip=True)
        
        # Additional reward for making progress
        if self.previous_tar_d is not None:
            progress = self.previous_tar_d - self.current_tar_d
            if progress > 0:  # If we're getting closer
                dist_tar_reward += 2.0 * progress  # Reward proportional to progress made
            
        # Print distance reward debug info
        if self.steps % 50 == 0:
            print(f"\n=== Distance Reward Debug ===")
            print(f"Current distance: {self.current_tar_d:.3f}")
            print(f"Previous distance: {self.previous_tar_d:.3f}")
            if self.previous_tar_d is not None:
                print(f"Progress made: {progress:.3f}")
            print(f"Base distance reward: {1.0 - normalize_to_range(self.current_tar_d, 0.0, 1.5, 0.0, 1.0, clip=True):.3f}")
            print(f"Final distance reward: {dist_tar_reward:.3f}")

        # --- Reach target reward ---
        reach_tar_reward = 0.0
        if self.current_tar_d < 0.15:  # Increased threshold for reaching reward
            reach_tar_reward = 2.0 * (1.0 - self.current_tar_d/0.15)  # Scales from 0 to 2.0 based on closeness
            if abs(self.current_tar_a) < (np.pi / 4):  # Extra bonus if facing target while close
                reach_tar_reward *= 1.5

        # --- Angle to target reward ---
        # Use raw angles for reward calculation
        current_angle_abs = abs(self.current_tar_a)
        previous_angle_abs = abs(self.previous_tar_a)
        
        # Reward improvement in angle
        if current_angle_abs > (np.pi / 4):  # If angle is large (>45 degrees)
            if previous_angle_abs - current_angle_abs > 0.001:  # If angle is decreasing
                ang_tar_reward = 1.0
            elif current_angle_abs - previous_angle_abs > 0.001:  # If angle is increasing
                ang_tar_reward = -1.0
            else:
                ang_tar_reward = 0.0
        else:
            # When angle is small, give proportional reward
            ang_tar_reward = 1.0 - (current_angle_abs / (np.pi / 4))  # 1.0 at 0 degrees, 0.0 at 45 degrees
        
        # Print angle debug info
        if self.steps % 50 == 0:
            print(f"\n=== Angle Reward Debug ===")
            print(f"Current angle: {self.current_tar_a:.3f} rad ({self.current_tar_a * 180/np.pi:.1f}°)")
            print(f"Previous angle: {self.previous_tar_a:.3f} rad ({self.previous_tar_a * 180/np.pi:.1f}°)")
            print(f"Angle reward: {ang_tar_reward:.3f}")

        # --- Obstacle avoidance rewards (distance sensors) ---
        dist_sensors_reward = 0
        prox_values = np.array([float(x) for x in self.message[:8]]) if self.message is not None and len(self.message) >= 8 else np.zeros(8)
        d_min = (1023.0 - min(prox_values)) * 5e-5  # meters from nearest wall
        
        # Only penalize when really close to obstacles
        if d_min < 0.10:  # Decreased from 0.15 to be less conservative
            dist_sensors_reward = -1.0
        elif d_min > 0.15:  # Increased threshold for positive reward
            dist_sensors_reward = 1.0
        else:
            dist_sensors_reward = 0.0
        dist_sensors_reward *= dist_tar_reward

        # --- Collision reward (adjusted to be more forgiving) ---
        collision_reward = 0.0
        if np.any(prox_values > 80):  # Increased from 70 to allow closer approach
            collision_reward = -1.0

        # --- Smoothness reward (no angular velocity, so use wheel difference) ---
        if action is not None:
            wheel_diff = abs(action[0])
        else:
            wheel_diff = 0.0
        smoothness_reward = -abs(normalize_to_range(wheel_diff, 0.0, 1.0, -1.0, 1.0, clip=True))
        smoothness_reward *= dist_tar_reward

        # --- Speed reward (modified to only reward speed when approaching target) ---
        if action is not None:
            speed = max(0.0, action[1])
            # Get absolute angle to target (0 means facing target directly)
            abs_angle_to_target = abs(get_angle_from_target(self.robot, self.target, is_abs=True))
            
            # Only give speed reward if we're getting closer to target and properly facing it
            distance_decreasing = (self.previous_tar_d - self.current_tar_d) > 0.001  # Check if we're getting closer
            facing_target = abs_angle_to_target < (np.pi / 3)  # Within 60 degrees of target
            
            if distance_decreasing and facing_target:
                # When very close to target (<20cm), be more forgiving about speed
                if self.current_tar_d < 0.20:
                    speed_reward = 0.5  # Base reward for being close and aligned
                    if speed > 0.0:  # Add small bonus for careful forward motion
                        speed_reward += 0.5 * speed
                else:
                    # Normal speed reward for longer distances
                    speed_reward = 2.0 * (np.exp(speed) - 1.0) if speed > 0.3 else normalize_to_range(speed, 0.0, 0.3, 0.0, 0.5)
                # Scale speed reward by how well we're facing the target
                angle_factor = 1.0 - (abs_angle_to_target / (np.pi / 3))  # 1.0 when facing perfectly, 0.0 at 60 degrees
                speed_reward *= angle_factor
            else:
                # Only penalize significant speed in wrong direction
                if speed > 0.2:  # Increased threshold for penalty
                    speed_reward = -2.0 * speed
                else:
                    speed_reward = 0.0  # No penalty for low speeds
        else:
            speed = 0.0
            speed_reward = 0.0
        speed_reward *= dist_tar_reward
        
        # Add debug info for speed reward calculation
        if self.steps % 50 == 0:
            print(f"\n=== Speed Reward Debug ===")
            print(f"Speed: {speed:.3f}")
            print(f"Distance decreasing: {distance_decreasing}")
            print(f"Absolute angle to target: {abs_angle_to_target:.3f} rad ({abs_angle_to_target * 180 / np.pi:.1f} degrees)")
            print(f"Facing target: {facing_target}")
            if facing_target and distance_decreasing:
                print(f"Angle factor: {angle_factor:.3f}")
            print(f"Final speed reward: {speed_reward:.3f}")

        # --- Reward modification: zero angle reward if obstacle/collision ---
        if dist_sensors_reward != 0.0 or collision_reward != 0.0:
            ang_tar_reward = 0.0

        # --- Reward weights (rebalanced for more aggressive target pursuit) ---
        reward_weight_dict = {
            "dist_tar": 3.0,           # Increased from 2.0 to 3.0 to prioritize target approach
            "ang_tar": 1.5,            # Increased from 1.0 to 1.5 to encourage better target alignment
            "dist_sensors": 0.3,       # Decreased from 0.5 to 0.3 to be less afraid of obstacles
            "tar_reach": 1.0,          # Kept same
            "collision": 1.0,          # Kept same for safety
            "smoothness_weight": 0.5,  # Decreased from 1.0 to 0.5 to allow more dynamic movement
            "speed_weight": 2.0        # Increased from 1.0 to 2.0 to strongly encourage forward motion
        }
        weighted_dist_tar_reward = reward_weight_dict["dist_tar"] * dist_tar_reward
        weighted_ang_tar_reward = reward_weight_dict["ang_tar"] * ang_tar_reward
        weighted_dist_sensors_reward = reward_weight_dict["dist_sensors"] * dist_sensors_reward
        weighted_reach_tar_reward = reward_weight_dict["tar_reach"] * reach_tar_reward
        weighted_collision_reward = reward_weight_dict["collision"] * collision_reward
        weighted_smoothness_reward = reward_weight_dict["smoothness_weight"] * smoothness_reward
        weighted_speed_reward = reward_weight_dict["speed_weight"] * speed_reward

        # --- Proximity bonus - reward for being close to target ---
        proximity_bonus = 0.0
        if self.current_tar_d < 1.5:  # Start bonus at 1.5 meters
            if self.current_tar_d < 0.085:  # Extra close bonus
                proximity_bonus = 12.0  # Extra high bonus when really close
            elif self.current_tar_d < 0.10:
                proximity_bonus = 8.0
            elif self.current_tar_d < 0.20:
                proximity_bonus = 6.0
            elif self.current_tar_d < 0.30:
                proximity_bonus = 4.0
            elif self.current_tar_d < 0.40:
                proximity_bonus = 2.0
            elif self.current_tar_d < 0.75:
                proximity_bonus = 1.0
            elif self.current_tar_d < 1.0:
                proximity_bonus = 0.5
            else:  # Between 1.0 and 1.5 meters
                proximity_bonus = 0.2

        # --- Terminal events (adjusted thresholds) ---
        terminal_reward = 0.0
        if self.current_tar_d < 0.10:  # Increased from 0.075 to 0.10 to be more forgiving
            terminal_reward = 100.0
            self.is_solved = True
            print("\n=== TARGET REACHED! (Obstacle) ===")
        elif d_min < 0.012:  # Hard collision - kept same
            terminal_reward = -100.0

        # --- Reach target reward (separate from terminal) ---
        reach_tar_reward = 0.0
        if self.current_tar_d < 0.15:  # Increased threshold for reaching reward
            reach_tar_reward = 2.0 * (1.0 - self.current_tar_d/0.15)  # Scales from 0 to 2.0 based on closeness
            if abs(self.current_tar_a) < (np.pi / 4):  # Extra bonus if facing target while close
                reach_tar_reward *= 1.5

        # Add to total reward
        reward = (weighted_dist_tar_reward + weighted_ang_tar_reward + weighted_dist_sensors_reward +
                  weighted_collision_reward + weighted_reach_tar_reward + weighted_smoothness_reward +
                  weighted_speed_reward + proximity_bonus + terminal_reward)

        if self.just_reset:
            self.just_reset = False
            return 0.0
        else:
            # Print debug info every 50 steps
            if self.steps % 50 == 0 or self.is_solved:
                print("\n=== Reward Breakdown (Obstacle) ===")
                print(f"Action: gas={action[1] if action is not None else 0.0:.3f}, wheel={action[0] if action is not None else 0.0:.3f}")
                print(f"Distance to target: {self.current_tar_d:.3f}")
                print(f"Angle to target: {self.current_tar_a:.3f}")
                print(f"Distance to target reward: {dist_tar_reward:.3f}")
                print(f"Angle to target reward: {ang_tar_reward:.3f}")
                print(f"Proximity sensors: {[f'{s:.3f}' for s in prox_values]}")
                print(f"Obstacle avoidance reward: {dist_sensors_reward:.3f}")
                print(f"Collision reward: {collision_reward:.3f}")
                print(f"Smoothness reward: {smoothness_reward:.3f}")
                print(f"Speed reward: {speed_reward:.3f}")
                print(f"Target reached reward: {reach_tar_reward:.3f}")
                print(f"Proximity bonus: {proximity_bonus:.3f}")
                print(f"Terminal reward: {terminal_reward:.3f}")
                print(f"Total reward: {reward:.3f}")
            return reward
    
    def _get_line_reward(self):
        """Reward for line following task"""
        if self.message is None or len(self.message) != 11:
            return -0.1
            
        # Get raw ground sensor values
        left_raw = float(self.message[8])
        middle_raw = float(self.message[9])
        right_raw = float(self.message[10])
        
        # Normalize ground sensor values [0, 1000] -> [0, 1]
        left = normalize_to_range(left_raw, 0, 1000, 0, 1)
        middle = normalize_to_range(middle_raw, 0, 1000, 0, 1)
        right = normalize_to_range(right_raw, 0, 1000, 0, 1)
        
        # Get current action from model
        obs = self.get_observations()
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Print debug info every 50 steps
        if self.steps % 50 == 0:
            print("\n=== Debug Info ===")
            print(f"Raw Ground Sensor Values:")
            print(f"Left: {left_raw:.1f} -> {left:.3f}")
            print(f"Middle: {middle_raw:.1f} -> {middle:.3f}")
            print(f"Right: {right_raw:.1f} -> {right:.3f}")
            print(f"Action: [wheel: {action[0]:.3f}, gas: {action[1]:.3f}]")
            print("==================\n")
        
        # Calculate reward
        reward = 0.0
        
        # Strong positive reward when middle sensor is on black line
        if middle < 0.4:  # Middle sensor on black line
            reward = 1.0
            # Add small bonus for forward movement when on line
            if action[1] > 0.5:  # If gas is above 0.5
                reward += 0.2
            if self.steps % 50 == 0:
                print("Reward: +1.0 (Middle sensor on black line)")
                if action[1] > 0.5:
                    print("Bonus: +0.2 (Forward movement on line)")
        # Small positive reward if either side sensor is on black
        elif left < 0.4 or right < 0.4:
            reward = 0.2
            if self.steps % 50 == 0:
                print("Reward: +0.2 (Side sensor on black line)")
        # Negative reward when completely on white
        else:
            reward = -0.1
            if self.steps % 50 == 0:
                print("Reward: -0.1 (All sensors on white)")
        
        return reward
    
    
    def _get_maze_reward(self, action=None):
        """Reward for maze navigation task (copied from obstacle task)"""
        if self.message is None or len(self.message) < 8:
            return 0.0
        if action is None:
            action = self._last_action if hasattr(self, '_last_action') else (0, 0)
        
        # Get proximity sensor values
        prox_values = np.array([float(x) for x in self.message[:8]])
        
        # Calculate current distance to target
        current_distance = get_distance_from_target(self.robot, self.target) if self.target is not None else 0.0
        current_angle = get_angle_from_target(self.robot, self.target, is_abs=False) if self.target is not None else 0.0
        
        # Calculate d_min (distance to nearest wall) early so it is available for all logic
        d_min = (1023.0 - min(prox_values)) * 5e-5  # meters from nearest wall

        # Initialize reward components
        forward_progress = 0.0
        wall_repulsion = 0.0
        spin_penalty = 0.0
        idle_penalty = 0.0
        terminal_reward = 0.0
        proximity_bonus = 0.0
        forward_speed_bonus = 0.0
        orientation_progress = 0.0
        wall_push_penalty = 0.0
        
        # 1. Forward progress reward
        if self.previous_distance is None:
            self.previous_distance = current_distance
            self.previous_angle = abs(current_angle)
            return 0.0
        else:
            # Scale forward progress reward based on distance to target
            if current_distance < 0.30:  # When close to target
                forward_progress = 30.0 * (self.previous_distance - current_distance)
            else:
                forward_progress = 20.0 * (self.previous_distance - current_distance)
            # Scale up forward progress if not near a wall
            if d_min > 0.10:
                forward_progress *= 1.5
        
        # 2. Orientation progress - reward for facing target (gated by forward speed)
        forward_speed = max(0.0, action[1])
        if forward_speed > 0.05 and current_distance > 0.10:  # Only reward when moving forward and far enough
            orientation_progress = 6.0 * (self.previous_angle - abs(current_angle)) * forward_speed
        else:
            orientation_progress = 0.0
        self.previous_angle = abs(current_angle)
        
        # 3. Wall repulsion - only when really close
        if d_min < 0.15:
            wall_repulsion = -2.0 * np.exp(-d_min / 0.10)
            wall_repulsion = max(wall_repulsion, -1.0)  # Cap the penalty at -1.0
        else:
            wall_repulsion = 0.0
        
        # 3b. Wall pushing penalty - when trying to move forward while too close
        if forward_speed > 0.1:
            front_sensors = [1, 2, 3, 4, 5]
            front_prox = max(prox_values[i] for i in front_sensors)
            front_distance = (1023.0 - front_prox) * 5e-5
            if front_distance < 0.15:  # increased threshold
                wall_push_penalty = -2.0 * forward_speed * (1.0 - front_distance/0.15)
        else:
            wall_push_penalty = 0.0
        
        # Encourage backing up and turning when close to wall
        # d_min is in meters
        if d_min < 0.08:
            reverse_gas = min(0.0, action[1])  # negative if reversing
            abs_wheel = abs(action[0])
            if reverse_gas < -0.1:
                # Small reward for just reversing
                wall_push_penalty += 0.5 * abs(reverse_gas)
                # Extra reward for reversing and turning
                if abs_wheel > 0.3:
                    wall_push_penalty += 0.5 * abs(reverse_gas) * abs_wheel
        
        # Encourage exploration and turning when close to a wall
        if d_min < 0.08:
            abs_wheel = abs(action[0])
            if abs_wheel > 0.2:
                # Small bonus for turning when close to a wall
                wall_push_penalty += 0.2 * abs_wheel

        # Encourage forward movement in open corridors (not near a wall)
        if d_min > 0.20:
            forward_speed = max(0.0, action[1])
            if forward_speed > 0.1:
                # Small bonus for sustained forward movement in open space
                forward_speed_bonus += 0.5 * forward_speed
        
        # 4. Spin penalty
        spin_penalty = 0.0  # removed spin penalty
        
        # 5. Idle penalty
        if abs(action[0]) < 0.1 and abs(action[1]) < 0.1:
            idle_penalty = -0.2  # reduced from -0.5 to -0.2
        
        # 6. Proximity bonus - reward for being close to target
        if current_distance < 0.40:  # Only apply when within 40cm
            if current_distance < 0.10:  # Within 10cm
                proximity_bonus = 4.0  # increased from 2.0
            elif current_distance < 0.20:  # Within 20cm
                proximity_bonus = 3.0  # increased from 1.5
            elif current_distance < 0.30:  # Within 30cm
                proximity_bonus = 2.0  # increased from 1.0
            else:  # Within 40cm
                proximity_bonus = 1.0  # increased from 0.5
        
        # 7. Forward speed bonus - only if we really moved forward this step
        dist_delta = self.previous_distance - current_distance  # already computed
        forward_speed = max(0.0, action[1])
        if dist_delta > 0.005:  # moved ≥0.5 cm
            forward_speed_bonus = 4.0 * forward_speed  # same scale as before
        else:
            forward_speed_bonus = 0.0
        
        # 8. Terminal events
        if current_distance < 0.075:  # Goal reached
            terminal_reward = 100.0
            self.is_solved = True
            print("\n=== TARGET REACHED! ===")
        elif d_min < 0.015:  # Hard collision
            terminal_reward = -100.0
        
        # --- Penalty for staying near walls for too long ---
        if not hasattr(self, 'near_wall_steps'):
            self.near_wall_steps = 0
        if d_min < 0.15:
            self.near_wall_steps += 1
        else:
            self.near_wall_steps = 0
        near_wall_penalty = 0.0
        if self.near_wall_steps > 40:
            near_wall_penalty = -0.5
        
        # --- Corner Escape Bonus ---
        # If close to wall and turning away (large wheel value), give a bonus
        corner_escape_bonus = 0.0
        if d_min < 0.15:
            abs_wheel = abs(action[0])
            if abs_wheel > 0.5:
                corner_escape_bonus = 0.5 * abs_wheel
        
        # Add to total reward
        total_reward = (forward_progress + orientation_progress + wall_repulsion + 
                       spin_penalty + idle_penalty + proximity_bonus + 
                       forward_speed_bonus + wall_push_penalty + terminal_reward +
                       near_wall_penalty + corner_escape_bonus)
        
        # Clip total reward
        total_reward = np.clip(total_reward, -110.0, 110.0)
        
        # --- Penalize lack of progress (stuck detection) ---
        if not hasattr(self, 'maze_no_progress_steps'):
            self.maze_no_progress_steps = 0
            self.maze_last_distance = current_distance
        if abs(current_distance - self.maze_last_distance) < 0.01:
            self.maze_no_progress_steps += 1
        else:
            self.maze_no_progress_steps = 0
            self.maze_last_distance = current_distance
        
        # Print debug info every 50 steps
        if self.steps % 50 == 0 or self.is_solved:
            print("\n=== Reward Breakdown (Maze) ===")
            print(f"Action: gas={action[1]:.3f}, wheel={action[0]:.3f}")
            print(f"Forward progress: {forward_progress:.3f}")
            print(f"Orientation progress: {orientation_progress:.3f}")
            print(f"Wall repulsion: {wall_repulsion:.3f}")
            print(f"Wall push penalty: {wall_push_penalty:.3f}")
            print(f"Spin penalty: {spin_penalty:.3f}")
            print(f"Idle penalty: {idle_penalty:.3f}")
            print(f"Proximity bonus: {proximity_bonus:.3f}")
            print(f"Forward speed bonus: {forward_speed_bonus:.3f}")
            print(f"Near wall penalty: {near_wall_penalty:.3f}")
            print(f"Corner escape bonus: {corner_escape_bonus:.3f}")
            print(f"Terminal reward: {terminal_reward:.3f}")
            if self.target is not None:
                print(f"Angle to target: {current_angle:.3f}")
                print(f"Distance to target: {current_distance:.3f}")
            print(f"Minimum obstacle distance: {d_min:.3f}")
            print(f"Proximity sensors: {[f'{s:.3f}' for s in prox_values]}")
            print(f"Total reward: {total_reward:.3f}")
        
        return total_reward
    

    def is_done(self):
        """Check if episode is complete"""
        self.steps += 1
        
        # Check step limit
        if self.steps >= self.max_steps:
            return True
            
        # Check if target reached
        if self.current_task == "obstacle" and self.is_solved:
            print("\nEpisode ended: Target reached!")
            return True
        if self.current_task == "maze" and self.is_solved:
            print("\nEpisode ended: Maze target reached!")
            return True
            
        # Check if robot is stuck off the line (no progress logic, like maze)
        if self.current_task == "line":
            current_position = self.robot.getPosition()
            # Calculate distance from start
            dist_from_start = np.linalg.norm(np.array(current_position) - np.array(self.line_start_position))
            # If robot has moved away from start, mark as left
            if dist_from_start > 0.15:  # 15 cm away from start
                self.line_has_left_start = True
            # If robot has left start and comes back close, mark as solved
            # --- Success condition for line task: robot completes a full loop and returns to start position ---
            # This will end the episode and trigger a reset, just like the maze task when the goal is reached.
            if self.line_has_left_start and dist_from_start < 0.10:  # within 10 cm of start
                self.is_solved = True  # Mark as solved for this episode
                print("[LINE] Success: Completed full loop!")
                return True
            # Existing no progress logic
            if not hasattr(self, 'line_last_position'):
                self.line_last_position = current_position
                self.line_no_progress_steps = 0
            dist_moved = np.linalg.norm(np.array(current_position) - np.array(self.line_last_position))
            if dist_moved < 0.01:  # Threshold for "no progress"
                self.line_no_progress_steps += 1
            else:
                self.line_no_progress_steps = 0
                self.line_last_position = current_position
            if self.line_no_progress_steps > 50:  # 50 steps of no progress
                print("Episode ended: Robot not making progress on the line")
                return True
        
        # Check if robot is stuck off the line
        if self.current_task == "line":
            if self.message is not None and len(self.message) == 11:
                # Get ground sensor values
                left = float(self.message[8])
                middle = float(self.message[9])
                right = float(self.message[10])
                
                # If all sensors are on white for too long (100 steps), end episode
                if left > 700 and middle > 700 and right > 700:
                    if not hasattr(self, 'off_line_steps'):
                        self.off_line_steps = 0
                    self.off_line_steps += 1
                    if self.off_line_steps >= 100:
                        print("Episode ended: Robot stuck off the line")
                        return True
                else:
                    self.off_line_steps = 0
             
            
        # Check if robot is stuck in obstacle task
        elif self.current_task == "obstacle":
            if self.message is not None and len(self.message) >= 8:
                # Get proximity sensor values
                prox_values = np.array([float(x) for x in self.message[:8]])
                
                # If too close to obstacles for too long (90 steps), end episode
                if np.max(prox_values) > 500:  # Very close to obstacle
                    if not hasattr(self, 'stuck_steps'):
                        self.stuck_steps = 0
                    self.stuck_steps += 1
                    if self.stuck_steps >= 90:
                        print("Episode ended: Robot stuck too close to obstacles")
                        return True
                else:
                    self.stuck_steps = 0
                    
                # If not making progress towards target for too long (200 steps), end episode
                if self.target is not None:
                    current_distance = get_distance_from_target(self.robot, self.target)
                    if not hasattr(self, 'no_progress_steps'):
                        self.no_progress_steps = 0
                        self.last_distance = current_distance
                    elif abs(current_distance - self.last_distance) < 0.05:  # Increased threshold from 0.01
                        self.no_progress_steps += 1
                        if self.no_progress_steps >= 200:
                            print("Episode ended: Robot not making progress towards target")
                            return True
                    else:
                        self.no_progress_steps = 0
                        self.last_distance = current_distance
                    
        # Task-specific completion criteria
        if self.current_task == "maze":
            goal = self.getFromDef('goal')
            if goal is not None:
                distance = np.sqrt(np.sum(
                    np.square(np.array(self.robot.getPosition()) - np.array(goal.getPosition()))
                ))
                if distance < 0.05:
                    print("======== + Solved + ========")
                    return True
                    
        # Penalize lack of progress in maze
        if self.current_task == "maze":
            if hasattr(self, 'maze_no_progress_steps') and self.maze_no_progress_steps > 200:
                print("Episode ended: Robot not making progress towards target in maze")
                return True
                    
        # # At the end of the episode for the line task, print the total episode reward
        if self.current_task == "line" and (self.steps >= self.max_steps or self.line_no_progress_steps > 50):
            print(f"[LINE] Total episode reward: {self.line_episode_reward:.2f}")
            
        return False


    def reset(self):
        # Reset all internal episode state before anything else
        self.steps = 0
        self.should_done = False
        self.is_solved = False
        self.line_episode_reward = 0.0
        # Reset previous values
        self.previous_distance = None  # Changed to None for first step handling
        # Reset stuck counters
        if hasattr(self, 'stuck_steps'):
            self.stuck_steps = 0
        if hasattr(self, 'no_progress_steps'):
            self.no_progress_steps = 0
            self.last_distance = 0.0
        if hasattr(self, 'line_no_progress_steps'):
            self.line_no_progress_steps = 0
        if hasattr(self, 'line_last_position'):
            self.line_last_position = self.robot.getPosition()
        if hasattr(self, 'off_line_steps'):
            self.off_line_steps = 0
        if hasattr(self, 'maze_no_progress_steps'):
            self.maze_no_progress_steps = 0
        if hasattr(self, 'maze_last_distance'):
            self.maze_last_distance = self.previous_distance if self.previous_distance is not None else 0.0
        # Store the starting position for the line task loop detection
        self.line_start_position = self.robot.getPosition()
        self.line_has_left_start = False  # Track if robot has left the start area
        # Print episode reset
        print(f"\nEpisode reset - Current task: {self.current_task}")
        # Call parent reset
        return super().reset()
    
    
    def get_info(self):
        """Return additional info about the environment state"""
        return {
            "steps": self.steps,
            "task": self.current_task,
            "is_solved": self.is_solved
        }

    def set_task(self, task_name):
        """Switch current task"""
        self.current_task = task_name
        # print(f"Switched to task: {task_name}")  # REMOVE or comment out

    def train_task(self, task_name, total_timesteps=10000, load_previous_task=None, checkpoint_step=None):
        """Train on specified task
        
        Args:
            task_name: Name of task to train on
            total_timesteps: Total number of timesteps to train for
            load_previous_task: Name of previous task to load model from
            checkpoint_step: Step number of checkpoint to load from same task
        """
        print(f"\n=== Starting training for task: {task_name} ===")
        self.set_task(task_name)
        
        # Setup callback for saving checkpoints, logging success rate, and logging raw episodic reward
        callback = CallbackList([
            TrainingCallback(check_freq=2000, save_path="logs"),
            SuccessRateCallback(),
            PerEpisodeRewardCallback()
        ])
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=1
        )
        
        # Save final state
        model_path = f"logs/sac_model_{task_name}_final.zip"
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save replay buffer
        buffer_path = f"logs/replay_buffer_{task_name}_final.pkl"
        self.model.save_replay_buffer(buffer_path)
        print(f"Replay buffer saved to {buffer_path}")
        
        print(f"=== Completed training for task: {task_name} ===\n")
    
    def evaluate_task(self, task_name, n_episodes=50):
        """Evaluate performance on specified task and return mean/std/success rate."""
        self.set_task(task_name)
        total_rewards = []
        successes = 0
        for episode in range(n_episodes):
            obs = self.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)
            successes += int(info.get("is_success", False))
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        success_rate = successes / n_episodes
        print(f"Task: {task_name}")
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Success rate: {success_rate:.2%}")
        # Save results to JSON
        # Use self.current_task as the model/task being evaluated (LOAD_FROM)
        result_path = f"logs/result_{task_name}_{self.current_task}.json"
        with open(result_path, 'w') as f:
            json.dump({
                "mean_return": mean_reward,
                "std_return": std_reward,
                "success_rate": success_rate
            }, f)
        print(f"Saved evaluation results to {result_path}")
        return {
            "mean_return": mean_reward,
            "std_return": std_reward,
            "success_rate": success_rate
        }

    def stop_robot(self):
        """Stop the robot by setting its velocity to zero"""
        try:
            # Get motor devices through supervisor
            left_motor = self.getDevice('left wheel motor')
            right_motor = self.getDevice('right wheel motor')
            
            # Stop motors
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            print("\nRobot stopped successfully!")
        except Exception as e:
            print(f"\nWarning: Could not stop robot: {e}")
            print("This is non-critical and evaluation results are still valid")

    def step(self, action):
        """
        Perform one environment step:
        - delegates to the parent implementation
        - then adds `is_success` to the info dict
        """
        obs, reward, done, info = super().step(action)
        # add the success flag so Monitor & callbacks can see it
        info["is_success"] = self.is_solved
        return obs, reward, done, info


def main():
    """Main function with mode selection for training or evaluation"""
    # ===== Configuration =====
    # Use top-level config variables
    mode = MODE
    task = TASK
    load_previous_task = LOAD_FROM
    checkpoint_step = CHECKPOINT_STEP  # Use top-level variable
    
    # Common parameters
    steps_per_episode = 20000   # Maximum steps per episode
    
    # Training specific parameters
    total_timesteps = 50000    # 200 episodes worth of steps (200 * 500)
    print_interval = 200       # Print progress every 50 steps
    
    # Evaluation specific parameters
    n_episodes = 3  # Number of episodes to evaluate
    
    # ===== Setup =====
    # Create the environment
    supervisor = UnifiedSupervisor()
    supervisor.set_task(task)
    supervisor.max_steps = steps_per_episode
    
    # Set the initial task
    print(f"\n=== {mode.capitalize()} Configuration ===")
    print(f"Task: {task}")
    print(f"Observation space: {supervisor.observation_space.shape[0]} dimensions")
    print(f"Action space: {supervisor.action_space.shape[0]} dimensions")
    
    print("\nModel Information:")
    if load_previous_task:
        print(f"Loading model from previous task: {load_previous_task}")
    elif checkpoint_step is not None:
        print(f"Loading checkpoint from step {checkpoint_step}")
    else:
        print("Creating new model from scratch")
    
    print(f"\nSteps per episode: {steps_per_episode}")
    
    if mode == "training":
        print(f"Total timesteps: {total_timesteps}")
        print(f"Print interval: {print_interval}")
    else:
        print(f"Number of episodes: {n_episodes}")
    print("===========================\n")
    
    # Load or create model
    supervisor.setup_sac(load_previous_task, checkpoint_step)
    
    # Run selected mode
    if mode == "training":
        # Train on the specified task
        supervisor.train_task(task, total_timesteps=total_timesteps, 
                            load_previous_task=load_previous_task,
                            checkpoint_step=checkpoint_step)
        print("\nTraining completed successfully!")
        
        # Run evaluation after training
        print("\nRunning evaluation...")
        results = supervisor.evaluate_task(task, n_episodes=n_episodes)
        mean_reward = results["mean_return"]
        std_reward = results["std_return"]
        success_rate = results["success_rate"]
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Success rate: {success_rate:.2%}")
    else:
        # Run evaluation
        results = supervisor.evaluate_task(task, n_episodes=n_episodes)
        mean_reward = results["mean_return"]
        std_reward = results["std_return"]
        success_rate = results["success_rate"]
        print("\nEvaluation completed!")
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Success rate: {success_rate:.2%}")
    
    # Stop the robot
    supervisor.stop_robot()


if __name__ == "__main__":
    main() 