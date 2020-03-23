"""
A protobuf based logging system for hexapod.
It log the information regarding time since reset, base position, orientation, 
motor information (joint angle, velocity, torque) in protocol buffers. See 
hexapod_logging.proto for more information. The episode_proto is updated per
time step by the enviornment and is saved into disk.
"""
import datetime, os, time
import tensorflow as tf
from pybullet_envs.hexapod.env import hexapod_logging_pb2

NUM_MOTORS = 18

def _update_base_state(base_state, values):
  base_state.x = values[0]
  base_state.y = values[1]
  base_state.z = values[2]

def preallocate_episode_proto(episode_proto, max_num_steps):
  """
  Preallocate max memory for episode proto.
  Dynamically allocating the memory for protobuf as it expands 
  and causes memory delay which is not desirable for control.
  Args:
  episode_proto : The proto contains state/action for current episode
  max_num_steps : The max num of steps that will recorded in proto
  """
  for _ in range(max_num_steps):
    step_log = episode_proto.state_action.add()
    step_log.info_valid = False
    step_log.time.seconds = 0
    step_log.time.nanos = 0
    for _ in range(NUM_MOTORS):
      motor_state = step_log.motor_states.add()
      motor_state.angle = 0
      motor_state.velocity = 0
      motor_state.torque = 0
      motor_state.action = 0
    _update_base_state(step_log.base_position, [0,0,0])
    _update_base_state(step_log.base_orientation, [0,0,0])
    _update_base_state(step_log.base_angular_vel, [0,0,0])

def update_episode_proto(episode_proto, hexapod, action, step):
  """
  Update the episode_proto by appending the state/action of the hexapod
  Args:
  episode_proto : The proto file containing the state/action for current episode
  hexapod : the hexapod instance. (env.hexapod)
  action : The action applied at current time step. The action is 18 element numpy 
  float array.
  step : The current time step
  """
  max_num_steps = len(episode_proto.state_action)
  if step >= max_num_steps:
    tf.logging.warning("{}th step is not logged since only {}steps were preallocated".format(step, max_num_steps))
    return
  step_log = episode_proto.state_action[step]
  step_log.info_valid = hexapod.IsObservationValid()
  time_in_seconds = hexapod.GetTimeSinceReset()
  step_log.time.seconds = int(time_in_seconds)
  step_log.time.nanos = int((time_in_seconds - int(time_in_seconds)) * 1e9)

  motor_angles = hexapod.GetMotorAngles()
  motor_velocities = hexapod.GetMotorVelocities()
  motor_torques = hexapod.GetMotorTorques()
  for i in range(hexapod.num_motors):
    step_log.motor_states[i].angle = motor_angles[i]
    step_log.motor_states[i].velocity = motor_velocities[i]
    step_log.motor_states[i].torque = motor_torques[i]
    step_log.motor_states[i].action = action[i]

  _update_base_state(step_log.base_position, hexapod.GetBasePosition())
  _update_base_state(step_log.base_orientation, hexapod.GetBaseRollPitchYaw())
  _update_base_state(step_log.base_angular_vel, hexapod.GetBaseRollPitchYawRate())


class HexapodLoggig(object):
  """
  A logging system that records state/action during simulation
  """
  def __init__(self, log_path = None):
    self._log_path = log_path

  def save_episode(self, episode_proto):
    """
    Save episode_proto to self._log_path.
    Here name of log file is time stamp and directory is self._log_path
    for example, self._log_path = "/usr/local/bin"
    then actual log file is "usr/local/bin/yyyy-mm-dd-hh:mm:ss"

    Args:
    episode_proto : The proto file containing state/action for current
    episode and need to be saved to the disk
    Returns:
    The full log path of file
    """
    if not self._log_path or not episode_proto.state_action:
      return self._log_path
    if not tf.gfile.Exists(self._log_path):
      tf.gfile.MakeDirs(self._log_path)

    ts = time.time()
    time_stamp = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d-%H%M%S")
    log_path = os.path.join(self._log_path, "hexapod_log{}".format(time_stamp))
    with tf.gfile.Open(log_path,"w") as f:
      f.write(episode_proto.SerializeToString())
    return log_path

  def restore_episode(self, log_path):
    """
    Restore the episode_proto from given log_path
    Args:
    log_path : full path of log file
    Returns:
    The hexapod episode_proto
    """
    with tf.gfile.Open(log_path,"rb") as f:
      content = f.read()
      episode_proto = hexapod_logging_pb2.HexapodEpisode()
      episode_proto.ParseFromString(content)
      return episode_proto