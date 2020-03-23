"""
This file implements gym environment of hexapod
"""

import math
import time

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
from pybullet_envs.hexapod import body
from pybullet_envs.hexapod.env import hexapod
from pybullet_envs.hexapod.env import hexapod_logging
from pybullet_envs.hexapod.env import hexapod_logging_pb2
from pybullet_envs.hexapod.env import motor
from pkg_resources import parse_version

NUM_MOTORS = 18
MOTOR_ANGLE_OBSERVATION_INDEX = 0
MOTOR_VELOCITY_OBSERVATION_INDEX = MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS
MOTOR_TORQUE_OBSERVATION_INDEX = MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS
BASE_ORIENTATION_OBSERVATION_INDEX = MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS
ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
RENDER_HEIGHT = 360
RENDER_WIDTH = 480
SENSOR_NOISE_STDDEV = hexapod.SENSOR_NOISE_STDDEV
DEFAULT_URDF_VERSION = "default"
NUM_SIMULATION_ITERATION_STEPS = 300

HEXAPOD_URDF_VERSION_MAP = {
  DEFAULT_URDF_VERSION : hexapod.Hexapod
}

def convert_to_list(obj):
  try:
    iter(obj)
    return obj
  except TypeError:
    return [obj]

def get_urdf_root_path():
  currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  parentdir = os.path.dirname(currentdir)
  urdf_root_path = os.path.join(parentdir,"body")
  return urdf_root_path

class HexapodGymEnv(gym.Env):
  """
  The gym enviornment for hexapod
  It simulte the locomotion of hexapod robot. The state space include angle, velociry and 
  torque of all eighteen motors and action include the desired angle for motor. The reward
  is based on how far the hexapod can walk in 1000 steps and energy consumption penalize
  the reward
  """
  metadata = {"render.modes" : ["human" , "rgb_array"], "video.frames_per_second" : 100}

  def __init__(self,
               urdf_root = get_urdf_root_path(),
               urdf_version = None,
               distance_weight = 1.0,
               energy_weight = 0.005,
               shake_weight = 0.0,
               drift_weight = 0.0,
               distance_limit = float("inf"),
               observation_noise_stdev = SENSOR_NOISE_STDDEV,
               self_collision_enabled = True,
               motor_velocity_limit = np.inf,
               pd_control_enabled = False,
               leg_model_enabled = False,
               accurate_motor_model_enabled = False,
               remove_default_joint_damping = False,
               motor_kp = 1.0,
               motor_kd = 0.02,
               control_latency = 0.0,
               pd_latency = 0.0,
               motor_overheat_protection = False,
               hard_reset = True,
               on_rack = False,
               render = False,
               num_steps_to_log = 1000,
               action_repeat = 1,
               control_time_step = None,
               env_randomizer = None,
               forward_reward_cap = float("inf"),
               reflection = True,
               log_path = None):
    """
    Initialize the hexapod gym environment
    Args:
    urdf_root : folder path for URDF file
    urdf_version : For now only DEFAULT_URDF_VERSION is used. Other version 
      will be available in future.
    distance_weight : The weight of distance term in reward
    energy_weight : The weight of energy term in reward
    shake_weight : The weight of vertical shakeness term in reward
    drift_weight : The weight of horizontal drift term in reawrd
    distance_limit : The maximum distance to terminate the episode
    obervation_noise_std_dev : The stddev of gaussian noise add into sensor values
    self_collision_enabled : Wheter to enable the self collision
    motor_velocity_limit : The velocity limit of each motor
    pd_control_enabled : Wheater to use PD controller for each motor
    leg_model_enabled : Whether to use leg model to reparameterized the action space
    accurate_motor_model_enabled : Wheter to use accurate motor model
    remove_default_joint_damping : Whether to remove joint damping
    motor_kp : proportional gain of the motor
    motor_kd: derivative gain of the motor
    control_latency : Latency between obervation is made and that reading is reported back to 
      Neural Network
    pd_latency : Latency between when the motor angle and velocity is observed and on the micro-
      controller and when the true state happens on the motor
    motor_overheat_protection : Wheter to enable the motor overheat protection
      It shutdown the motor if large torque is applied for large time
    hard_reset : Wheter to wipe the simulation and load everything when reset is called.
      If set to false, the reset will put hexapod to its initial position and intial configuration
    on_rack : For the debuging, wheter to put the hexapod on rack. To observe its walking gait
    render : Wheter to render the simulation
    num_steps_to_log : The max num of steps to log in proto file. The num of steps are more than 
      this steps but futher steps are not recorded
    action_repeat : The number of simulation steps before action is applied
    control_time_step : The time step between two control signal
    env_randomizer : An instance of (List of) EnvRandomizer(z). It randomize the physical property
      of hexapod, randomize the terrain during reset() or add perturbation forces during step()
    forward_reward_cap : The max value at which forward reward is capped at
    log_path : The path to log  
    """

    self._log_path = log_path
    self.logging = hexapod_logging.HexapodLoggig(self._log_path)
    if control_time_step is not None:
      self.control_time_step = control_time_step
      self._action_repeat = action_repeat
      self._time_step = control_time_step / action_repeat
    else:
      if accurate_motor_model_enabled or pd_control_enabled:
        self._time_step = 0.002
        self._action_repeat = 5
      else:
        self._time_step = 0.01
        self._action_repeat = 1
      self.control_time_step = self._time_step * self._action_repeat

    self._num_bullet_solver_iterations = int(NUM_SIMULATION_ITERATION_STEPS / self._action_repeat)
    self._urdf_root = urdf_root
    self._self_collision_enabled = self_collision_enabled
    self._motor_velocity_limit = motor_velocity_limit
    self._observation = []
    self._true_observation = []
    self._objectives = []
    self._objective_weights = [distance_weight, energy_weight, drift_weight, shake_weight]
    self._env_step_counter = 0
    self._num_steps_to_log = num_steps_to_log
    self._is_render = render
    self._last_base_position = [0,0,0]
    self._distance_weight = distance_weight
    self._energy_weight = energy_weight
    self._shake_weight = shake_weight
    self._drift_weight = drift_weight
    self._distance_limit = distance_limit
    self._observation_noise_stdev = observation_noise_stdev
    self._action_bound = 1./4
    self._pd_control_enabled = pd_control_enabled
    self._leg_model_enabled = leg_model_enabled
    self._accurate_moter_model_enabled = accurate_motor_model_enabled
    self._remove_default_joint_dampig = remove_default_joint_damping
    self._motor_kp = motor_kd
    self._motor_kd = motor_kd
    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack
    self._cam_dist = 1.0
    self._cam_pitch = -30
    self._cam_yaw = 0
    self._forward_reward_cap = forward_reward_cap
    self._hard_reset = True
    self._last_frame_time = 0.0
    self._control_latency = control_latency
    self._pd_latency = pd_latency
    self._urdf_version = urdf_version
    self._ground_id = None
    self._reflection = reflection
    self._env_randomizer = convert_to_list(env_randomizer) if env_randomizer else []
    self._episode_proto = hexapod_logging_pb2.HexapodEpisode()
    if self._is_render:
      self._pybullet_client = bc.BulletClient(connection_mode = pybullet.GUI)
    else:
      self._pybullet_client = bc.BulletClient()
    if self._urdf_version is None:
      self._urdf_version = DEFAULT_URDF_VERSION
    self._pybullet_client.setPhysicsEngineParameter(enableConeFriction = 0)
    self.seed()
    self.reset()
    observation_high = ( self._get_observation_upper_bound() + OBSERVATION_EPS )
    observation_low = ( self._get_observation_lower_bound() - OBSERVATION_EPS )
    action_dim = NUM_MOTORS
    action_high = np.array([self._action_bound] * action_dim)
    self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(observation_low, observation_high)
    self.viewer = None
    self._hard_reset = hard_reset

  def close(self):
    if self._env_step_counter > 0:
      self.logging.save_episode(self._episode_proto)
    self.hexapod.Terminate()

  def reset(self, initial_motor_angles = None, reset_duration = 1.0):
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)
    if self._env_step_counter > 0:
      self.logging.save_episode(self._episode_proto)
    self._episode_proto = hexapod_logging_pb2.HexapodEpisode()
    hexapod_logging.preallocate_episode_proto(self._episode_proto,self._num_steps_to_log)
    if self._hard_reset:
      self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(
        numSolverIterations = int(self._num_bullet_solver_iterations))
      self._pybullet_client.setTimeStep(self._time_step)
      self._ground_id = self._pybullet_client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
      if self._reflection:
        self._pybullet_client.changeVisualShape(self._ground_id, -1, rgbaColor = [1,1,1,0.8])
        self._pybullet_client.configureDebugVisualizer(
          self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, self._ground_id)
      self._pybullet_client.setGravity(0,0,-9.8)
      acc_motor = self._accurate_moter_model_enabled
      motor_protect = self._motor_overheat_protection
      if self._urdf_version not in HEXAPOD_URDF_VERSION_MAP:
        raise ValueError("%s is not correct type of URDF version" % self._urdf_version)
      else:
        self.hexapod = HEXAPOD_URDF_VERSION_MAP[self._urdf_version](
          pybullet_client = self._pybullet_client,
          action_repeat = self._action_repeat,
          urdf_root = self._urdf_root,
          time_step = self._time_step,
          self_collision_enabled = self._self_collision_enabled,
          motor_velocity_limit = self._motor_velocity_limit,
          motor_overheat_protection = self._motor_overheat_protection,
          pd_control_enabled = self._pd_control_enabled,
          accurate_motor_model_enabled = self._accurate_moter_model_enabled,
          remove_default_joint_damping = self._remove_default_joint_dampig,
          motor_kp = self._motor_kp,
          motor_kd = self._motor_kd,
          control_latency = self._control_latency,
          pd_latency = self._pd_latency,
          observation_noise_stdev = self._observation_noise_stdev,
          on_rack = self._on_rack)

    self.hexapod.Reset(reload_urdf=False, default_motor_angles=initial_motor_angles, reset_time=reset_duration)

    for env_randomizer in self._env_randomizer:
      env_randomizer.randomize_env(self)

    self._pybullet_client.setPhysicsEngineParameter(enableConeFriction = 0)
    self._env_step_counter = 0
    self._last_base_position = [0,0,0]
    self._objectives = []
    self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0,0,0])
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)
    return self._get_observation()

  def seed(self, seed = None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def render(self, mode="rgb_array", close = False):
    """
    Render the frame for current time step in simulation using PyBullet
    Args:
    mode : "rgb_array" or "human" mode. first one returns the rgb array of frame-
    Used to save video. Second mode generate frames and display them immediately.
    """
    if mode != "rgb_array":
      return np.array([])
    base_pos = self.hexapod.GetBasePosition()
    view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
      cameraTargetPosition = base_pos,
      distance = self._cam_dist,
      yaw = self._cam_yaw,
      pitch = self._cam_pitch,
      roll = 0,
      upAxisIndex = 2)
    proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
      fov = 60,
      aspect = float(RENDER_WIDTH) / RENDER_HEIGHT,
      nearVal = 0.1,
      farVal = 100)
    (_,_,px,_,_) = self._pybullet_client.getCameraImage(
      width = RENDER_WIDTH,
      height = RENDER_HEIGHT,
      renderer = self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
      viewMatrix = view_matrix,
      projectionMatrix = proj_matrix)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:,:,:3]
    return rgb_array

  def step(self, action):
    """
    Step forward the simultion for given action
    Args:
    action : A list of angle of eighteen motors

    Returns:
    observation : The angle,velocity and torque of all motors
    reward : reward for current state/action par
    done : Wheter the episode has ended
    info : A dictionary that stores diagnostic info

    Raises:
    ValueError : The action dimension is not same as the numbers of motors
    ValueError : The magnitude of action is  out of bound
    """
    self._last_base_position = self.hexapod.GetBasePosition()

    if self._is_render:
      time_spent = time.time() - self._last_frame_time
      self._last_frame_time = time.time()
      time_to_sleep = self.control_time_step - time_spent
      if time_to_sleep > 0:
        time.sleep(time_to_sleep)
      base_pos = self.hexapod.GetBasePosition()
      [yaw, pitch ,dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
      self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)

    for env_randomizer in self._env_randomizer:
      env_randomizer.randomize_step(self)

    action = self._transform_action_to_motor_command(action)
    self.hexapod.Step(action)
    reward = self._reward()
    done = self._termination()
    if self._log_path is not None:
      hexapod_logging.update_episode_proto(self._episode_proto, self.hexapod, action, self._env_step_counter)
    self._env_step_counter += 1
    if done:
      self.hexapod.Terminate()

    return np.array(self._get_observation()), reward, done, {}

  def add_env_randomizer(self, env_randomizer):
    self._env_randomizer.append(env_randomizer)

  def _get_observation_upper_bound(self):
    """
    Get the upper bound for observation
    see GetObservation() for more info
    """
    upper_bound = np.zeros(self._get_observation_dimension())
    num_motors = self.hexapod.num_motors
    upper_bound[0:num_motors] = math.pi / 4 #joint angle
    upper_bound[num_motors: 2 * num_motors] = (motor.MOTOR_SPEED_LIMIT) #joint velocity
    upper_bound[num_motors * 2 : 3 * num_motors] = (motor.OBSERVED_TORQUE_LIMIT) #joint torque
    upper_bound[num_motors * 3: 4 * num_motors] = 1. #Quaternion of base
    return upper_bound

  def _get_observation_lower_bound(self):
    """
    Get the lower bound for observation
    """
    lower_bound = self._get_observation_upper_bound()
    num_motors = self.hexapod.num_motors
    lower_bound[0:num_motors] = -1*(math.pi / 4)
    return lower_bound


  def _get_observation_dimension(self):
    """
    Get the length of observation list
    """
    return len(self._get_observation())

  def _get_observation(self):
    """
    Get the observation of env with noise and latency
    hexapod class contains history of true observation and based on latency 
    req observation is found, interpolation is used if necessory. Then gaussian
    noise is added based on self.observation_noise_stdev
    """

    observation = []
    observation.extend(self.hexapod.GetMotorAngles().tolist())
    observation.extend(self.hexapod.GetMotorVelocities().tolist())
    observation.extend(self.hexapod.GetMotorTorques().tolist())
    observation.extend(list(self.hexapod.GetBaseOrientation()))
    self._true_observation = observation
    return self._true_observation

  def _get_true_observation(self):
    """
    Get the true observation of env (values without sensor noise and latency)
    """

    observation = []
    observation.extend(self.hexapod.GetTrueMotorAngles().tolist())
    observation.extend(self.hexapod.GetTrueMotorVelocities().tolist())
    observation.extend(self.hexapod.GetTrueMotorTorques().tolist())
    observation.extned(list(self.hexapod.GetBaseOrientation()))
    self._observation = observation
    return self._observation

  def _transform_action_to_motor_command(self,action):
    if self._leg_model_enabled:
      for i,action_component in enumerate(action):
        if not (-self._action_bound - ACTION_EPS <= action_component <= self._action_bound + ACTION_EPS):
          raise ValueError("{}th action {} out of bounds.".format(i, action_component))
      action = self.hexapod.ConvertFromLegModel(action)
    return action

  def _reward(self):
    current_base_position = self.hexapod.GetBasePosition()
    forward_reward = current_base_position[0] - self._last_base_position[0]
    forward_reward = min(forward_reward, self._forward_reward_cap)

    drift_reward = -abs(current_base_position[1] - self._last_base_position[1])

    orientation = self.hexapod.GetBaseOrientation()
    rot_matrix = pybullet.getMatrixFromQuaternion(orientation)
    local_up_vec = rot_matrix[6:]
    shake_reward = -abs(np.dot(np.asarray([1,1,0]), np.asarray(local_up_vec)))
    energy_reward = -np.abs(np.dot(self.hexapod.GetMotorTorques(),self.hexapod.GetMotorVelocities()))*self._time_step

    objectives = [forward_reward, energy_reward, drift_reward, shake_reward]
    weighted_objectives = [o * w for o,w in zip(objectives,self._objective_weights)]
    reward = sum(weighted_objectives)
    self._objectives.append(objectives)
    return reward

  def _termination(self):
    position = self.hexapod.GetBasePosition()
    distance = math.sqrt(position[0]**2 + position[1]**2)
    return self._is_fallen() or distance > self._distance_limit

  def _is_fallen(self):
    """
    Determine wheter the hexapod has fallen or not
    """
    orientation = self.hexapod.GetBaseOrientation()
    rot_matrix = pybullet.getMatrixFromQuaternion(orientation)
    local_up_vec = rot_matrix[6:]
    pos = self.hexapod.GetBasePosition()
#NUMERICAL VALUE NEED TO BE CHANGED
    return np.dot(np.asarray([0,0,1]), np.asarray(local_up_vec)) < 0.85 or pos[1] < 0.13

  def get_objectives(self):
    return self._objectives

  @property
  def objective_weights(self):
    """
    Return the weigths of all objectives
    """
    return self._objective_weights

  def get_hexapod_motor_angles(self):
    """
    Return the np array of hexapod motor angles
    """
    return np.array(self._observation[MOTOR_ANGLE_OBSERVATION_INDEX : MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS])

  def get_hexapod_motor_velocities(self):
    """
    Return the np array of motor velocities
    """
    return np.array(self._observation[MOTOR_VELOCITY_OBSERVATION_INDEX : MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS])

  def get_hexapod_motor_torques(self):
    """
    Return the np arrat of motor torques
    """
    return np.array(self._observation[MOTOR_TORQUE_OBSERVATION_INDEX : MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS])

  def get_hexapod_base_orientation(self):
    """
    return base orientation in quaternion
    """
    return np.array(self._observation[BASE_ORIENTATION_OBSERVATION_INDEX:])

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step

  def set_time_step(self, control_step, simulation_step = 0.001):
    """
    Set the time step of enviornment
    Args:
    control_step : The time period between two control signal is applied
    simulation_step : The simulation time step in Pybullet. Default value is 0.001

    Raise:
    ValueError : When control_step is smaller than simulation_step
    """

    if control_step < simulation_step:
      raise ValueError("Control step should be larger than simulation step")
    self.control_time_step = control_step
    self._time_step = simulation_step
    self._action_repeat = int(round(self.control_time_step / self._time_step))
    self._num_bullet_solver_iterations = (NUM_SIMULATION_ITERATION_STEPS / self._action_repeat)
    self._pybullet_client.setPhysicsEngineParameter(numSolverIterations = self._num_bullet_solver_iterations)
    self._pybullet_client.setTimestep(self._time_step)
    self.hexapod.setTimesteps(action_repeat = self._action_repeat, simulation_step = self._time_step)

  @property
  def pybullet_client(self):
    return self._pybullet_client
  
  @property
  def ground_id(self):
    return self._ground_id
  
  @ground_id.setter
  def ground_id(self, ground_id):
    self._ground_id = ground_id

  @property
  def env_step_counter(self):
    return self._env_step_counter
  




  






