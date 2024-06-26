# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: hexapod_logging.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pybullet_envs.hexapod.env import timestamp_pb2 as timestamp__pb2
from pybullet_envs.hexapod.env import vector_pb2 as vector__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='hexapod_logging.proto',
  package='robotics.reinforcement_learning.hexapod.envs',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x15hexapod_logging.proto\x12,robotics.reinforcement_learning.hexapod.envs\x1a\x0ftimestamp.proto\x1a\x0cvector.proto\"h\n\x0eHexapodEpisode\x12V\n\x0cstate_action\x18\x01 \x03(\x0b\x32@.robotics.reinforcement_learning.hexapod.envs.HexapodStateAction\"T\n\x11HexapodMotorState\x12\r\n\x05\x61ngle\x18\x01 \x01(\x01\x12\x10\n\x08velocity\x18\x02 \x01(\x01\x12\x0e\n\x06torque\x18\x03 \x01(\x01\x12\x0e\n\x06\x61\x63tion\x18\x04 \x01(\x01\"\xcb\x02\n\x12HexapodStateAction\x12\x12\n\ninfo_valid\x18\x06 \x01(\x08\x12(\n\x04time\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x32\n\rbase_position\x18\x02 \x01(\x0b\x32\x1b.robotics.messages.Vector3d\x12\x35\n\x10\x62\x61se_orientation\x18\x03 \x01(\x0b\x32\x1b.robotics.messages.Vector3d\x12\x35\n\x10\x62\x61se_angular_vel\x18\x04 \x01(\x0b\x32\x1b.robotics.messages.Vector3d\x12U\n\x0cmotor_states\x18\x05 \x03(\x0b\x32?.robotics.reinforcement_learning.hexapod.envs.HexapodMotorStateb\x06proto3'
  ,
  dependencies=[timestamp__pb2.DESCRIPTOR,vector__pb2.DESCRIPTOR,])




_HEXAPODEPISODE = _descriptor.Descriptor(
  name='HexapodEpisode',
  full_name='robotics.reinforcement_learning.hexapod.envs.HexapodEpisode',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='state_action', full_name='robotics.reinforcement_learning.hexapod.envs.HexapodEpisode.state_action', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=102,
  serialized_end=206,
)


_HEXAPODMOTORSTATE = _descriptor.Descriptor(
  name='HexapodMotorState',
  full_name='robotics.reinforcement_learning.hexapod.envs.HexapodMotorState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='angle', full_name='robotics.reinforcement_learning.hexapod.envs.HexapodMotorState.angle', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='velocity', full_name='robotics.reinforcement_learning.hexapod.envs.HexapodMotorState.velocity', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='torque', full_name='robotics.reinforcement_learning.hexapod.envs.HexapodMotorState.torque', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='action', full_name='robotics.reinforcement_learning.hexapod.envs.HexapodMotorState.action', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=208,
  serialized_end=292,
)


_HEXAPODSTATEACTION = _descriptor.Descriptor(
  name='HexapodStateAction',
  full_name='robotics.reinforcement_learning.hexapod.envs.HexapodStateAction',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='info_valid', full_name='robotics.reinforcement_learning.hexapod.envs.HexapodStateAction.info_valid', index=0,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='time', full_name='robotics.reinforcement_learning.hexapod.envs.HexapodStateAction.time', index=1,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='base_position', full_name='robotics.reinforcement_learning.hexapod.envs.HexapodStateAction.base_position', index=2,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='base_orientation', full_name='robotics.reinforcement_learning.hexapod.envs.HexapodStateAction.base_orientation', index=3,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='base_angular_vel', full_name='robotics.reinforcement_learning.hexapod.envs.HexapodStateAction.base_angular_vel', index=4,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='motor_states', full_name='robotics.reinforcement_learning.hexapod.envs.HexapodStateAction.motor_states', index=5,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=295,
  serialized_end=626,
)

_HEXAPODEPISODE.fields_by_name['state_action'].message_type = _HEXAPODSTATEACTION
_HEXAPODSTATEACTION.fields_by_name['time'].message_type = timestamp__pb2._TIMESTAMP
_HEXAPODSTATEACTION.fields_by_name['base_position'].message_type = vector__pb2._VECTOR3D
_HEXAPODSTATEACTION.fields_by_name['base_orientation'].message_type = vector__pb2._VECTOR3D
_HEXAPODSTATEACTION.fields_by_name['base_angular_vel'].message_type = vector__pb2._VECTOR3D
_HEXAPODSTATEACTION.fields_by_name['motor_states'].message_type = _HEXAPODMOTORSTATE
DESCRIPTOR.message_types_by_name['HexapodEpisode'] = _HEXAPODEPISODE
DESCRIPTOR.message_types_by_name['HexapodMotorState'] = _HEXAPODMOTORSTATE
DESCRIPTOR.message_types_by_name['HexapodStateAction'] = _HEXAPODSTATEACTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

HexapodEpisode = _reflection.GeneratedProtocolMessageType('HexapodEpisode', (_message.Message,), {
  'DESCRIPTOR' : _HEXAPODEPISODE,
  '__module__' : 'hexapod_logging_pb2'
  # @@protoc_insertion_point(class_scope:robotics.reinforcement_learning.hexapod.envs.HexapodEpisode)
  })
_sym_db.RegisterMessage(HexapodEpisode)

HexapodMotorState = _reflection.GeneratedProtocolMessageType('HexapodMotorState', (_message.Message,), {
  'DESCRIPTOR' : _HEXAPODMOTORSTATE,
  '__module__' : 'hexapod_logging_pb2'
  # @@protoc_insertion_point(class_scope:robotics.reinforcement_learning.hexapod.envs.HexapodMotorState)
  })
_sym_db.RegisterMessage(HexapodMotorState)

HexapodStateAction = _reflection.GeneratedProtocolMessageType('HexapodStateAction', (_message.Message,), {
  'DESCRIPTOR' : _HEXAPODSTATEACTION,
  '__module__' : 'hexapod_logging_pb2'
  # @@protoc_insertion_point(class_scope:robotics.reinforcement_learning.hexapod.envs.HexapodStateAction)
  })
_sym_db.RegisterMessage(HexapodStateAction)


# @@protoc_insertion_point(module_scope)
