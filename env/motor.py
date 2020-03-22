'''This file implements accurate servo motor model'''

import numpy as np

VOLTAGE_CLIPPING = 50
OBSERVED_TORQUE_LIMIT = 5.7
MOTOR_VOLTAGE = 16.0
MOTOR_RESISTANCE = 0.186
MOTOR_TORQUE_CONSTANT = 0.0954
MOTOR_VISCOUS_DAMPING = 0
NUM_MOTORS = 18
MOTOR_SPEED_LIMIT = MOTOR_VOLTAGE / (MOTOR_VISCOUS_DAMPING + MOTOR_TORQUE_CONSTANT)

class MotorModel(object):
  """
  The accurate motor model based on the physics of DC servo motor.
  The motor has position control mode: a desired motor angle is provided
  and a torque is calculated based on the internal model of the motor.

  The internal motor model takese following consideration: pd gain,
  viscous damping, back EMF voltage and current-torque profile.
  """

  def __init__(self,kp = 1.2 ,kd = 0):
    self._kp = kp
    self._kd = kd
    self._resistance = MOTOR_RESISTANCE
    self._voltage = MOTOR_VOLTAGE
    self._torque_constant = MOTOR_TORQUE_CONSTANT
    self._viscouse_damping = MOTOR_VISCOUS_DAMPING
    self._current_table = [0, 10, 20, 30, 40, 50, 60]
    self._torque_table = [0, 1, 1.9, 2.45, 3.0, 3.25, 3.5]

  def set_motor_gains(self,kp,kd):
    self._kp = kp
    self._kd = kd

  def set_voltage(self, voltage):
    self._voltage = voltage

  def get_voltage(self):
    return self._voltage

  def set_viscous_damping(self,viscous_damping):
    self._viscouse_damping = viscous_damping

  def get_viscous_damping(self):
    return self._viscouse_damping

  def convert_to_torque(self, motor_commands, motor_angle, motor_velocity, true_motor_velocity, kp = None, kd = None):
    """
    Convert motor commands to torque.
    Args: 
    motor_commands: desired motor angles
    current_motor_angle: the motor angle at current time step
    current_motor_velocity: the motor velocity at current time step

    Returns:
    actual_torque: the actual torque applied to motor
    observed_torque: the torque observed by the sensor

    """
    if kp is None:
      kp = np.full(NUM_MOTORS, self._kp)
    if kd is None:
      kd = np.full(NUM_MOTORS, self._kp)

    pwm = -1 * kp * (motor_angle - motor_commands) - kd * motor_velocity
    pwm = np.clip(pwm,-1.0 ,1.0)

    observed_torque = np.clip(self._torque_constant * (np.asarray(pwm) * self._voltage / self._resistance)
      , -OBSERVED_TORQUE_LIMIT, OBSERVED_TORQUE_LIMIT)

    voltage_net = np.clip(np.asarray(pwm)*self._voltage - (self._torque_constant + self._viscouse_damping)*np.asarray(true_motor_velocity),
      -VOLTAGE_CLIPPING, VOLTAGE_CLIPPING)

    current = voltage_net / self._resistance
    current_sign = np.sign(current)
    current_magnitude = np.absolute(current)

    actual_torque = np.interp( current_magnitude, self._current_table, self._torque_table)
    actual_torque = np.multiply(actual_torque, current_sign)

    return actual_torque, observed_torque



