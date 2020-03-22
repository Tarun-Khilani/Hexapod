"""
This file contains the functionalities of the hexapod in PyBullet
"""

import os,inspect
import collections
import copy
import math
import re
from pybullet_envs.hexapod.env import motor
import numpy as np

INIT_POSITION = [0,0,0.2]
INIT_RACK_POSITION = [0,0,1]
INIT_ORIENTATION = [0,0,0,1]
OVERHEAT_SHUTDOWN_TORQUE = 2.45
OVERHEAT_SHUTDOWN_TIME = 1.0


_BASE_MOTOR_PATTERN = re.compile(r"base_motor\d*_\d*")
_MOTOR_MOTOR_PATTERN = re.compile(r"motor\d*_\d*_motor\d*_\d*")
_MOTOR_JOINT_PATTERN = re.compile(r"motor\d*_\d*_joint\d*")
_JOINT_LEG_PATTERN = re.compile(r"joint\d*_leg\d*")

SENSOR_NOISE_STDDEV = (0.0, 0.0, 0.0, 0.0, 0.0)
TWO_PI = 2 * math.pi

def MapToMinusPiToPi(angles):
  """Maps a list of angles to [-pi, pi].

  Args:
    angles: A list of angles in rad.
  Returns:
    A list of angle mapped to [-pi, pi].
  """
  mapped_angles = copy.deepcopy(angles)
  for i in range(len(angles)):
    mapped_angles[i] = math.fmod(angles[i], TWO_PI)
    if mapped_angles[i] >= math.pi:
      mapped_angles[i] -= TWO_PI
    elif mapped_angles[i] < -math.pi:
      mapped_angles[i] += TWO_PI
  return mapped_angles


class Hexapod(object):
  """
  This hexapod class simulates the hexapod robot
  """

  def __init__(self, 
    pybullet_client, 
    urdf_root="", 
    time_step = 0.01, 
    action_repeat = 1,
    self_collision_enabled = False, 
    motor_velocity_limit = np.inf,
    pd_control_enabled = False,
    accurate_motor_model_enabled = False,
    remove_default_joint_damping = False, 
    motor_kp = 1.0, 
    motor_kd = 0.02, 
    pd_latency = 0.0,
    control_latency = 0.0, 
    observation_noise_stdev = SENSOR_NOISE_STDDEV, 
    motor_overheat_protection = False, 
    on_rack = False):
    """
  Construct hexapod and reset it to intial state

  Args:
  pybullet_client : The instance of Bullet Client to manage simulation
  urdf_root : The path to the URDF folder
  time_steps : The time steps of the simulation
  action_repeat : the number of ApplyAction() for each control step
  self_collision_enabled : Wheater to enable self collision
  motor_velocity_limit : Limit of motor velocity
  accurate_motor_model_enabled : Wheater to enable accurate modeling of motor
  remove_default_joint_damping : Wheather to remove default joint damping
  motor_kp : Proportional gain constant of motor
  motor_kd : Derivative gain constant of motor
  pd_latency : The latency of observation in seconds used to calculate PD control. It is the
  latency between microcontroller and motor controller
  control_latency : The latency of observation in seconds used to calculate action. It is the
  latency from the motor controller, microcontroller to the host(Nvidia TX1).
  observation_noise_stdev : The standard deviation for the Gaussian noise in sensor module.
  It is array of for separate sensors in order of [motor_angle, motor_velocity, 
  motor_torque, base_roll_pitch_yaw, base_angular_velocity]
  motor_overheat_protection : Protection for the overheating of motor
  on_rack : Wheater to place the hexapod on rack. This is used to only debug the walking gait the base 
  is hanged in the mid air to observe the walking gait.
    """    
    self.num_motors = 18
    self.num_legs = 6
    self._pybullet_client = pybullet_client
    self._action_repeat = action_repeat
    self._urdf_root = urdf_root
    self._self_collision_enabled = self_collision_enabled
    self._motor_velocity_limit = motor_velocity_limit
    self._observed_motor_torques = np.zeros(self.num_motors)
    self._applied_motor_torques = np.zeros(self.num_motors)
    self._max_force = 3.5
    self._pd_latency = pd_latency
    self._control_latency = control_latency
    self._observation_noise_stdev = observation_noise_stdev
    self._accurate_motor_model_enabled = accurate_motor_model_enabled
    self._pd_control_enabled = pd_control_enabled
    self._remove_default_joint_damping = remove_default_joint_damping
    self._observation_history = collections.deque(maxlen = 100)
    self._control_observation = []
    self._base_link_id = [-1]
    self._base_motor_link_ids = []
    self._motor_motor_link_ids = []
    self._motor_joint_link_ids = []
    self._joint_leg_link_ids = []
    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack

    if self._pd_control_enabled:
      self._kp = 8
      self._kd = 0.3
    else:
      self._kp = 1
      self._kd = 1

    if self._accurate_motor_model_enabled:
      self._kp = motor_kp
      self._kd = motor_kd
      self._motor_model = motor.MotorModel(kp=self._kp, kd=self._kd)

    self.time_step = time_step
    self._step_counter = 0

    self.Reset(reset_time = -1)

#This function is used by hexapod_gym_env.py
  def Terminate(self):
    pass

#This function is required by hexapod_logging.py
  def GetTimeSinceReset(self):
    return self._step_counter*self.time_step

#This function is used by hexapod_gym_env.py
  def Step(self, action):
    for _ in range(self._action_repeat):
      self.ApplyAction(action)
      self._pybullet_client.stepSimulation()
      self.ReceiveObservation()
      self._step_counter += 1

  def ReceiveObservation(self):
    """
    Receive observation from sensor noise
    This function is called once per step and update observation history
    """
    self._observation_history.appendleft(self._GetTrueObservation())
    self._control_observation = self._GetControlObservation()

  def _GetTrueObservation(self):
    observation = []
    observation.extend(self.GetTrueMotorAngles())
    observation.extend(self.GetTrueMotorVelocities())
    observation.extend(self.GetTrueMotorTorques())
    observation.extend(self.GetTrueBaseOrientation())
    observation.extend(self.GetTrueBaseRollPitchYawRate())
    return observation

  def _GetControlObservation(self):
    control_delayed_observation = self._GetDelayedObservation(self._control_latency)
    return control_delayed_observation

  def ApplyAction(self, motor_commands, motor_kps = None, motor_kds = None):
    """
    Set the Desired motor angles of hexapod
    If pd_control_enabled is true, a torque is calculated according to difference between 
    desired angle and current angle, current motor velocity. This torque is applied to
    motor.
    Args:
    motor_commands : The eighteen desired angles of motor
    motor_kps : Proportional gain constant of motor
    motor_kds : Derivative gain constant of motor
    """
    if self._motor_velocity_limit < np.inf:
      current_motor_angle = self.GetTrueMotorAngles()
      motor_command_max = current_motor_angle + self.time_step*self._motor_velocity_limit
      motor_command_min = current_motor_angle + self.time_step*self._motor_velocity_limit
      motor_commands = np.clip(motor_commands, motor_command_max, motor_command_min)

    if motor_kps is None:
      motor_kps = np.full(18,self._kp)
    if motor_kds is None:
      motor_kds = np.full(18,self._kd)

    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      q, qdot = self._GetPDObservation()
      qdot_true = self.GetTrueMotorVelocities()
      if self._accurate_motor_model_enabled:
        actual_torque, observed_torque = self._motor_model.convert_to_torque(
          motor_commands, q, qdot, qdot_true, motor_kps, motor_kds)
        if self._motor_overheat_protection:
          for i in range(self.num_motors):
            if abs(actual_torque[i]) > OVERHEAT_SHUTDOWN_TORQUE:
              self._overheat_counter[i] += 1
            else:
              self._overheat_counter[i] += 1
            if self._overheat_counter[i] > (OVERHEAT_SHUTDOWN_TIME / self.time_step):
              self._motor_enabled_list[i] = False

        self._observed_motor_torques = observed_torque
        self._applied_motor_torques = self._observed_motor_torques

        for motor_id, motor_torque, motor_enable in zip(self._motor_id_list, 
          self._applied_motor_torques, self._motor_enabled_list):
          if motor_enable:
            self._SetMotorTorqueById(motor_id, motor_torque)
          else:
            self._SetMotorTorqueById(motor_id,0)
          
      else:
        torque_commands = -1 * motor_kps * (q - motor_commands) - motor_kds * qdot
        self._observed_motor_torques = torque_commands
        self._applied_motor_torques = self._observed_motor_torques

        for motor_id, motor_torque in zip(self._motor_id_list, self._applied_motor_torques):
          self._SetMotorTorqueById(motor_id, motor_torque)

    else:
      for motor_id, motor_command in zip(self._motor_id_list, motor_commands):
        self._SetDesiredMotorAngleById(motor_id, motor_command)

  def GetTrueMotorAngles(self):
    """
    Return the current sixteen motor angles
    """
    motor_angles = [
    self._pybullet_client.getJointState(self.hexapod, motor_id)[0] for motor_id in self._motor_id_list
    ]
    return motor_angles

  def GetTrueMotorVelocities(self):
    """
    Get the true velocities of all eighteen motors
    """
    motor_velocities = [
    self._pybullet_client.getJointState(self.hexapod, motor_id)[1] for motor_id in self._motor_id_list
    ]
    return motor_velocities

  def GetTrueMotorTorques(self):
    """
    Get the torque applied on eighteen motors
    """
    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      return self._observed_motor_torques
    else:
      motor_torques = [
      self._pybullet_client.getJointState(self.hexapod, motor_id)[3] for motor_id in self._motor_id_list
      ]
    return motor_torques

  def GetTrueBaseOrientation(self):
    """
    Get the orientation of hexapod base in quaternion
    """
    _, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.hexapod))
    return orientation

  def GetTrueBaseRollPitchYawRate(self):
    """
    Get the rate of change of orientation of the base of hexapod , rate of change of (roll,pitch,yaw)
    """
    vel = self._pybullet_client.getBaseVelocity(self.hexapod)
    return np.asarray([vel[1][0], vel[1][1], vel[1][2]])

  def _SetMotorTorqueById(self,motor_id, motor_torque):
    self._pybullet_client.setJointMotorControl2(bodyIndex = self.hexapod,
                                                jointIndex = motor_id,
                                                controlMode = self._pybullet_client.TORQUE_CONTROL,
                                                force = motor_torque)


  def _GetPDObservation(self):
    pd_delayed_observation = self._GetDelayedObservation(self._pd_latency)
    q = pd_delayed_observation[0:self.num_motors]
    qdot = pd_delayed_observation[self.num_motors:self.num_motors*2]
    return (np.array(q), np.array(qdot))

  def _GetDelayedObservation(self,latency):
    """
    Return Delayed observation
    Args:
    latency : The latency of delayed obersvation
    Returns:
    observation : Observation observed latency seconds ago
    """
    if latency <= 0 or len(self._observation_history) == 1:
      observation = self._observation_history[0]
    else:
      n_steps_ago = int(latency/self.time_step)
      if n_steps_ago + 1 >= len(self._observation_history):
        return self._observation_history[-1]
      remaining_latency = latency - n_steps_ago * self.time_step
      blend_alpha = remaining_latency / self.time_step
      observation = ( (1.0 - blend_alpha)*np.array(self._observation_history[n_steps_ago]) + blend_alpha * np.array(
        self._observation_history[n_steps_ago+1]) )
    return observation

#This function is used by hexapod_gym_env.py
  def Reset(self, reload_urdf = True, default_motor_angles = None, reset_time = 3.0):
    """
    Reset the hexapod to its intial state
    Args:
    reload_urdf : Wheater to reload the URDF file. If not, the hexapod is returned to its initial position
    default_motor_angles : Default motor motor angles. If None the hexapod will hold its default pose for 100 
    time steps
    reset_time : The duration to hold the default motor angles

    """

    if self._on_rack:
      init_position = INIT_RACK_POSITION
    else:
      init_position = INIT_POSITION
    if reload_urdf:
      if self._self_collision_enabled:
        self.hexapod = self._pybullet_client.loadURDF("%s/hexapod.urdf" %self._urdf_root,
          init_position, useFixedBase = self._on_rack,
          flags = self._pybullet_client.URDF_USE_SELF_COLLISION)

      else:
        self.hexapod = self.pybullet_client.loadURDF("%s/hexapod.urdf" %self._urdf_root,
          init_position, useFixedBase = self._on_rack)

      self._BuildJointNameToIdDict()
      self._BuildURDFIds()
      if self._remove_default_joint_damping:
        self._RemoveDefaultJointDamping()
      self._BuildMotorIdList()
      self._RecordMassInfoFromURDF()
      self._RecordInertiaInfoFromURDF()
      self._ResetPose()

    else:
      self._pybullet_client.resetBasePositionAndOrientation(self.hexapod, init_position, INIT_ORIENTATION)
      self._pybullet_client.resetBaseVelocity(self.hexapod, [0,0,0], [0,0,0])
      self._ResetPose()

    self._overheat_counter = np.zeros(self.num_motors)
    self._motor_enabled_list = [True] * self.num_motors
    self._step_counter = 0

#NO IDEA WHY I M DOING THIS
    self._observation_history.clear()
    if reset_time > 0.0:
      self.ReceiveObservation()
      for _ in range(100):
        self.ApplyAction([math.pi / 2] * self.num_motors)
        self._pybullet_client.stepSimulation()
        self.ReceiveObservation()
      if default_motor_angles is not None:
        num_steps_to_reset = int(reset_time / self.time_step)
        for _ in range(num_steps_to_reset):
          self.ApplyAction(default_motor_angles)
          self._pybullet_client.stepSimulation()
          self.ReceiveObservation()
    self.ReceiveObservation() 

  def _BuildJointNameToIdDict(self):
    """
    Create a dictionary mapping for joint name to joint id
    """
    num_joints = self._pybullet_client.getNumJoints(self.hexapod)
    self._joint_name_to_id = {}
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.hexapod,i)
      self._joint_name_to_id[ joint_info[1].decode("UTF-8") ] = joint_info[0]

  def _BuildURDFIds(self):
    """
    Build the link ids from its name in URDF file
    """
    num_joints = self._pybullet_client.getNumJoints(self.hexapod)
    self._base_link_id = [-1]
    self._base_motor_link_ids = []
    self._motor_motor_link_ids = []
    self._motor_joint_link_ids = []
    self._joint_leg_link_ids = []
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.hexapod,i)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = self._joint_name_to_id[joint_name]

      if _BASE_MOTOR_PATTERN.match(joint_name):
        self._base_motor_link_ids.append(joint_id)

      elif _MOTOR_MOTOR_PATTERN.match(joint_name):
        self._motor_motor_link_ids.append(joint_id)

      elif _MOTOR_JOINT_PATTERN.match(joint_name):
        self._motor_joint_link_ids.append(joint_id)

      else:
        self._joint_leg_link_ids.append(joint_id)

      self._base_motor_link_ids.sort()
      self._motor_motor_link_ids.sort()
      self._motor_joint_link_ids.sort()
      self._joint_leg_link_ids.sort()

  def _RemoveDefaultJointDamping(self):
    """
    Remove Default Joint Damping from every Joints
    """
    num_joints = self._pybullet_client.getNumJoints(self.hexapod)
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.hexapod,i)
      self._pybullet_client.changeDynamics(joint_info[0], -1, linearDamping = 0, angularDamping = 0)

  def _BuildMotorIdList(self):
    """
    Create List of all motor joints in hexapod
    """ 
    self._motor_id_list = self._base_motor_link_ids + self._motor_joint_link_ids + self._joint_leg_link_ids
    self._motor_id_list.sort()

  def _RecordMassInfoFromURDF(self):
    """
    Record mass of each link from its joint id
    """
    
    self._base_motor_mass_urdf = []
    for i in self._base_motor_link_ids:
      self._base_motor_mass_urdf.append(self._pybullet_client.getDynamicsInfo(self.hexapod,i)[0])

    self._motor_motor_mass_urdf = []
    for i in self._motor_motor_link_ids:
      self._motor_motor_mass_urdf.append(self._pybullet_client.getDynamicsInfo(self.hexapod,i)[0])

    self._motor_joint_mass_urdf = []
    for i in self._motor_joint_link_ids:
      self._motor_joint_mass_urdf.append(self._pybullet_client.getDynamicsInfo(self.hexapod,i)[0])

    self._joint_leg_masss_urdf = []
    for i in self._joint_leg_link_ids:
      self._joint_leg_masss_urdf.append(self._pybullet_client.getDynamicsInfo(self.hexapod,i)[0])


  def _RecordInertiaInfoFromURDF(self):
    """
    Record Inertia of each link from its joint Id
    """
    
    self._base_motor_inertia_urdf = []
    for i in self._base_motor_link_ids:
      self._base_motor_inertia_urdf.append(self._pybullet_client.getDynamicsInfo(self.hexapod,i)[2])

    self._motor_motor_inertia_urdf = []
    for i in self._motor_motor_link_ids:
      self._motor_motor_inertia_urdf.append(self._pybullet_client.getDynamicsInfo(self.hexapod,i)[2])

    self._motor_joint_inertia_urdf = []
    for i in self._motor_joint_link_ids:
      self._motor_joint_inertia_urdf.append(self._pybullet_client.getDynamicsInfo(self.hexapod,i)[2])

    self._joint_leg_inertia_urdf = []
    for i in self._joint_leg_link_ids:
      self._joint_leg_inertia_urdf.append(self._pybullet_client.getDynamicsInfo(self.hexapod,i)[2])

  def _ResetPose(self):
    """
    Reset the pose of hexapod
    """
    for i in self._base_motor_link_ids:
      self._ResetPoseForLeg(i)

  def _ResetPoseForLeg(self,i):
    """
    Reste Pose of Leg
    """
    knee_friction_force = 0
    self._pybullet_client.resetJointState(self.hexapod, i, 0, targetVelocity = 0)
    self._pybullet_client.resetJointState(self.hexapod, i+2, 0, targetVelocity = 0)
    self._pybullet_client.resetJointState(self.hexapod, i+3, 0, targetVelocity = 0)

    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      self._pybullet_client.setJointMotorControl2(
        bodyIndex = self.hexapod,
        jointIndex = i,
        controlMode = self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity = 0,
        force = knee_friction_force)

      self._pybullet_client.setJointMotorControl2(
        bodyIndex = self.hexapod,
        jointIndex = i+2,
        controlMode = self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity = 0,
        force = knee_friction_force)

      self._pybullet_client.setJointMotorControl2(
        bodyIndex = self.hexapod,
        jointIndex = i+3,
        controlMode = self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity = 0,
        force = knee_friction_force)

    else:
      self._SetDesiredMotorAngleById(i,0)
      self._SetDesiredMotorAngleById(i+2,0)
      self._SetDesiredMotorAngleById(i+3,0)

  def _SetDesiredMotorAngleByName(self,motor_name,desired_angle):
    """
    Set desired angle of motor specified by its name
    """
    self._SetDesiredMotorAngleById(self._joint_name_to_id(motor_name), desired_angle)

  def _SetDesiredMotorAngleById(self,motor_id,desired_angle):
    """
    Set desired angle of motor specified by its ID
    """
    self._pybullet_client.setJointMotorControl2(
      bodyIndex = self.hexapod,
      jointIndex = motor_id,
      controlMode = self._pybullet_client.POSITION_CONTROL,
      targetPosition = desired_angle,
      positionGain = self._kp,
      velocityGain = self._kd,
      force = self._max_force)



#-------------------------------------------------------------------------------
#---------------------------------Extra functions-------------------------------
#-------------------------------------------------------------------------------

#This function is required by hexapod_logging.py
  def IsObservationValid(self):
    """
    Return wheater the observation is valid or not for current time step
    In simulation, observations are always valid. In real life, communication error between
    Jetson TX1 and microcontroller can send invalid observations
    """
    return True

  def _AddSensorNoise(self, sensor_values, noise_stdev):
    if noise_stdev <= 0:
      return sensor_values
    observation = sensor_values + np.random.normal(scale = noise_stdev, size = sensor_values.shape)
    return observation

#This function is used by hexapod_logging.py, hexapod_gym_env.py
  def GetBasePosition(self):
    """
    Get the base position of the hexapod
    """
    position, _ = self._pybullet_client.getBasePositionAndOrientation(self.hexapod)
    return position

  def GetTrueBaseRollPitchYaw(self):
    """
    Get true base orientation in euler angles tuple (roll, pitch, yaw) in the world frame
    """
    orientation = self.GetTrueBaseOrientation()
    roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(orientation)
    return np.asarray(roll_pitch_yaw)

#This function is used by hexapod_logging.py
  def GetBaseRollPitchYaw(self):
    """
    Return the base orienation in euler angles with sensor noise and latency
    """
    delayed_orientation = np.array(
      self._control_observation[3 * self.num_motors : 3 * self.num_motors + 4])
    delayed_roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(delayed_orientation)
    roll_pitch_yaw = self._AddSensorNoise(delayed_roll_pitch_yaw, self._observation_noise_stdev[3])
    return roll_pitch_yaw

#This function is required by hexpod_logging.py, hexapod_gym_env.py
  def GetMotorAngles(self):
    """
    Get the eighteen motor angles with sensor noise and latency mapped to [-pi,pi]
    """
    motor_angles = self._AddSensorNoise(np.array(self._control_observation[0:self.num_motors]),
      self._observation_noise_stdev[0])
    return MapToMinusPiToPi(motor_angles)

#This function is required by hexapod_logging.py, hexapod_gym_env.py
  def GetMotorVelocities(self):
    """
    Get the eighteen motor velocities with sensor noise and latency
    """
    motor_velocities = self._AddSensorNoise(
      np.array(self._control_observation[self.num_motors: 2 * self.num_motors]),
      self._observation_noise_stdev[1]
      )
    return motor_velocities

#This function is required by hexapod_logging.py, hexapod_gym_env.py
  def GetMotorTorques(self):
    """
    Get all motor torques with sensor noise and latency
    """
    return self._AddSensorNoise(
      np.array(self._control_observation[2 * self.num_motors : 3 * self.num_motors]),
      self._observation_noise_stdev[2]
      )

#This function is used by hexapod_gym_env.py
  def GetBaseOrientation(self):
    """
    Get the base orientation in quaternion with sensor noise and latency
    """
    return self._pybullet_client.getQuaternionFromEuler(self.GetBaseRollPitchYaw())

#This function is used by hexapod_logging.py
  def GetBaseRollPitchYawRate(self):
    """
    Get the rate of change of orientation of base of hexapod in euler angle tupe (roll,pitch,yaw)
    with sensor noise and latency
    """
    return self._AddSensorNoise(
      np.array(self._control_observation[3 * self.num_motors + 4:3 * self.num_motors + 7 ]),
      self._observation_noise_stdev[4]
      )

  def GetActionDimension(self):
    """
    Get the length of action list
    """
    return self.num_motors

  def SetFootFriction(self, foot_friction):
    """
    Set the lateral friction coefficient of foot for all feet
    """
    for link_id in self._joint_leg_link_ids:
      self._pybullet_client.changeDynamics(self.hexapod, link_id, lateralFriction = foot_friction)

  def SetFootRestitution(self, foot_restitution):
    """
    Set the coefficient of restitution (bounciness) of foot for all feet
    """
    for link_id in self._joint_leg_link_ids:
      self._pybullet_client.changeDynamics(self.hexapod, link_id, restitution = foot_restitution)

  def SetJointFriction(self, joint_frictions):
    for link_id, friction in zip(self._joint_leg_link_ids, joint_frictions):
      self._pybullet_client.setJointMotorControl2(
        bodyIndex = self.hexapod,
        jointIndex = link_id,
        controlMode = self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity = 0,
        force = friction
        )

  def GetNumKneeJoints(self):
    return len(self._joint_leg_link_ids)

  def SetBatteryVoltage(self, voltage):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_voltage(voltage)

  def SetMotorViscousDamping(self, viscous_damping):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_viscous_damping(viscous_damping)

  def SetControlLatency(self, latency):
    """
    Set the latency for the control loop
    latency: time difference between sending the action from Jetson TX1 and 
    observing the observation from microcontroller
    """
    self._control_latency = latency

  def GetControlLatency(self):
    """
    Get the control latency
    """
    return self._control_latency

  def SetMotorGains(self, kp, kd):
    """
    Set the motor gains kp and kd
    """
    self._kp = kp
    self._kd = kd
    if self._accurate_motor_model_enabled:
      self._motor_model.set_motor_gains(kp, kd)

  def GetMotorGains(self):
    """
    Get the motor gains kp and kd
    """
    return self._kp, self._kd

#This function is used by hexapod_gym_env.py
  def SetTimeSteps(self, action_repeat, simulation_step):
    """
    Set the time step of control and simulation
    Args:
    action_repeat : the number simulation steps the same action is repeated
    simulation_step : The simulation time step
    """
    self._action_repeat = action_repeat
    self.time_step = simulation_step

  @property
  def base_link_id(self):
    return self._base_link_id

  def ConvertFromLegModel(self, actions):
    """
    Convert the angles of leg model to real motor angles
    """
    motor_angle = copy.deepcopy(actions)
    motor_angle = motor_angle * math.pi
    return motor_angle
  



#-----------------------------------------------------------------------------------
#--------------------------THIS PART NEED TO BE CHANGED-----------------------------
#-----------------------------------------------------------------------------------
'''

  def GetBaseMassesFromURDF(self):
    """Get the mass of the base from the URDF file."""
    return self._base_mass_urdf

  def GetBaseInertiasFromURDF(self):
    """Get the inertia of the base from the URDF file."""
    return self._base_inertia_urdf

  def GetLegMassesFromURDF(self):
    """Get the mass of the legs from the URDF file."""
    return self._leg_masses_urdf

  def GetLegInertiasFromURDF(self):
    """Get the inertia of the legs from the URDF file."""
    return self._leg_inertia_urdf

  def SetBaseMasses(self, base_mass):
    """Set the mass of minitaur's base.

    Args:
      base_mass: A list of masses of each body link in CHASIS_LINK_IDS. The
        length of this list should be the same as the length of CHASIS_LINK_IDS.
    Raises:
      ValueError: It is raised when the length of base_mass is not the same as
        the length of self._chassis_link_ids.
    """
    if len(base_mass) != len(self._chassis_link_ids):
      raise ValueError("The length of base_mass {} and self._chassis_link_ids {} are not "
                       "the same.".format(len(base_mass), len(self._chassis_link_ids)))
    for chassis_id, chassis_mass in zip(self._chassis_link_ids, base_mass):
      self._pybullet_client.changeDynamics(self.quadruped, chassis_id, mass=chassis_mass)

  def SetLegMasses(self, leg_masses):
    """Set the mass of the legs.

    A leg includes leg_link and motor. 4 legs contain 16 links (4 links each)
    and 8 motors. First 16 numbers correspond to link masses, last 8 correspond
    to motor masses (24 total).

    Args:
      leg_masses: The leg and motor masses for all the leg links and motors.

    Raises:
      ValueError: It is raised when the length of masses is not equal to number
        of links + motors.
    """
    if len(leg_masses) != len(self._leg_link_ids) + len(self._motor_link_ids):
      raise ValueError("The number of values passed to SetLegMasses are "
                       "different than number of leg links and motors.")
    for leg_id, leg_mass in zip(self._leg_link_ids, leg_masses):
      self._pybullet_client.changeDynamics(self.quadruped, leg_id, mass=leg_mass)
    motor_masses = leg_masses[len(self._leg_link_ids):]
    for link_id, motor_mass in zip(self._motor_link_ids, motor_masses):
      self._pybullet_client.changeDynamics(self.quadruped, link_id, mass=motor_mass)

  def SetBaseInertias(self, base_inertias):
    """Set the inertias of minitaur's base.

    Args:
      base_inertias: A list of inertias of each body link in CHASIS_LINK_IDS.
        The length of this list should be the same as the length of
        CHASIS_LINK_IDS.
    Raises:
      ValueError: It is raised when the length of base_inertias is not the same
        as the length of self._chassis_link_ids and base_inertias contains
        negative values.
    """
    if len(base_inertias) != len(self._chassis_link_ids):
      raise ValueError("The length of base_inertias {} and self._chassis_link_ids {} are "
                       "not the same.".format(len(base_inertias), len(self._chassis_link_ids)))
    for chassis_id, chassis_inertia in zip(self._chassis_link_ids, base_inertias):
      for inertia_value in chassis_inertia:
        if (np.asarray(inertia_value) < 0).any():
          raise ValueError("Values in inertia matrix should be non-negative.")
      self._pybullet_client.changeDynamics(self.quadruped,
                                           chassis_id,
                                           localInertiaDiagonal=chassis_inertia)

  def SetLegInertias(self, leg_inertias):
    """Set the inertias of the legs.

    A leg includes leg_link and motor. 4 legs contain 16 links (4 links each)
    and 8 motors. First 16 numbers correspond to link inertia, last 8 correspond
    to motor inertia (24 total).

    Args:
      leg_inertias: The leg and motor inertias for all the leg links and motors.

    Raises:
      ValueError: It is raised when the length of inertias is not equal to
      the number of links + motors or leg_inertias contains negative values.
    """

    if len(leg_inertias) != len(self._leg_link_ids) + len(self._motor_link_ids):
      raise ValueError("The number of values passed to SetLegMasses are "
                       "different than number of leg links and motors.")
    for leg_id, leg_inertia in zip(self._leg_link_ids, leg_inertias):
      for inertia_value in leg_inertias:
        if (np.asarray(inertia_value) < 0).any():
          raise ValueError("Values in inertia matrix should be non-negative.")
      self._pybullet_client.changeDynamics(self.quadruped,
                                           leg_id,
                                           localInertiaDiagonal=leg_inertia)

    motor_inertias = leg_inertias[len(self._leg_link_ids):]
    for link_id, motor_inertia in zip(self._motor_link_ids, motor_inertias):
      for inertia_value in motor_inertias:
        if (np.asarray(inertia_value) < 0).any():
          raise ValueError("Values in inertia matrix should be non-negative.")
      self._pybullet_client.changeDynamics(self.quadruped,
                                           link_id,
                                           localInertiaDiagonal=motor_inertia)
'''