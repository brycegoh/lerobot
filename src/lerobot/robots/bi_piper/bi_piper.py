#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import cached_property
import torch

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError

from lerobot.robots.robot import Robot
from .config_bi_piper import BiPiperConfig
from piper_control import piper_interface
from piper_control import piper_init
from piper_control import piper_connect

logger = logging.getLogger(__name__)

class BiPiper(Robot):
    """
    Bimanual Piper Robot using piper_sdk for CAN communication
    """

    config_class = BiPiperConfig
    name = "bi_piper"

    def __init__(self, config: BiPiperConfig):
        super().__init__(config)
        self.config = config

        # Use PiperInterface wrapper from piper_control
        try:
            self.PiperInterface = piper_interface.PiperInterface
        except Exception as e:
            raise ImportError(
                "piper_control is not available. Ensure it's installed in the environment"
            ) from e

        # Initialize left and right arm interfaces
        self.left_arm = None
        self.right_arm = None
        print(f"CONFIG: {config}")
        self.cameras = make_cameras_from_configs(config.cameras) if config.cameras is not None else None
        if self.cameras is None:
            raise ValueError("cameras is none")

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Define the motor features for both arms"""
        motor_features = {}
        # Left arm: joint_1 to joint_6 and gripper
        for i in range(1, 7):
            motor_features[f"left_joint_{i}.pos"] = float
        motor_features["left_gripper.pos"] = float

        # Right arm: joint_1 to joint_6 and gripper
        for i in range(1, 7):
            motor_features[f"right_joint_{i}.pos"] = float
        motor_features["right_gripper.pos"] = float

        return motor_features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        cam_ft = {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
        return cam_ft 

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        left_connected = self.left_arm is not None
        right_connected = self.right_arm is not None
        cameras_connected = all(cam.is_connected for cam in self.cameras.values())
        return left_connected and right_connected and cameras_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to both Piper arms via CAN ports
        """
        
        # Print out the CAN ports that are available to connect.
        print(piper_connect.find_ports())

        # Activate all the ports so that you can connect to any arms connected to your
        # machine.
        piper_connect.activate()

        # Connect left arm
        try:
            logger.info(f"Connecting to left arm on CAN port: {self.config.left_arm_can_port}")
            self.left_arm = self.PiperInterface(self.config.left_arm_can_port)
            # Ensure arm enabled and in CAN JOINT mode
            piper_init.reset_arm(
                self.left_arm,
                arm_controller=piper_interface.ArmController.POSITION_VELOCITY,
                move_mode=piper_interface.MoveMode.JOINT,
            )
        except Exception as e:
            logger.error(f"Failed to connect to left arm: {e}")
            raise

        # Connect right arm
        try:
            logger.info(f"Connecting to right arm on CAN port: {self.config.right_arm_can_port}")
            self.right_arm = self.PiperInterface(self.config.right_arm_can_port)
            piper_init.reset_arm(
                self.right_arm,
                arm_controller=piper_interface.ArmController.POSITION_VELOCITY,
                move_mode=piper_interface.MoveMode.JOINT,
            )
        except Exception as e:
            logger.error(f"Failed to connect to right arm: {e}")
            raise


        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        logger.info("BiPiper robot connected successfully")

        

    @property
    def is_calibrated(self) -> bool:
        """BiPiper robots are assumed to be always calibrated"""
        return True

    def calibrate(self) -> None:
        """BiPiper robots don't require manual calibration"""
        logger.info("BiPiper robot calibration - no action needed, assumed calibrated")
        pass

    def configure(self) -> None:
        """Configure the BiPiper robot - no specific configuration needed"""
        logger.info("BiPiper robot configuration - no action needed")
        pass

    def disconnect(self) -> None:
        """
        Disconnect from both Piper arms and cameras
        """
        # try:
        #     if self.left_arm is not None:
        #         try:
        #             self.left_arm.DisconnectPort()
        #             logger.info("Left arm disconnected from CAN port")
        #         except Exception as e:
        #             logger.warning(f"Error disconnecting left arm: {e}")
        #         self.left_arm = None

        #     if self.right_arm is not None:
        #         try:
        #             self.right_arm.DisconnectPort()
        #             logger.info("Right arm disconnected from CAN port")
        #         except Exception as e:
        #             logger.warning(f"Error disconnecting right arm: {e}")
        #         self.right_arm = None

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info("BiPiper robot disconnected successfully")

    def get_observation(self) -> dict:
        """
        Capture current joint positions and camera images
        """
        if not self.is_connected:
            raise RuntimeError("BiPiper robot is not connected")

        observation = {}

        try:
            # Get left arm joint positions
            left_joint_msgs = self.left_arm.piper.GetArmJointMsgs()
            left_gripper_msgs = self.left_arm.piper.GetArmGripperMsgs()

            # Parse joint positions for left arm
            # Based on the format: ArmMsgFeedBackJointStates with Joint 1-6 values
            self._parse_joint_messages(left_joint_msgs, "left", observation)

            # Parse gripper position for left arm
            # Based on the format: ArmMsgFeedBackGripper with grippers_angle
            self._parse_gripper_messages(left_gripper_msgs, "left", observation)

            # Get right arm joint positions
            right_joint_msgs = self.right_arm.piper.GetArmJointMsgs()
            right_gripper_msgs = self.right_arm.piper.GetArmGripperMsgs()

            # Parse joint positions for right arm
            self._parse_joint_messages(right_joint_msgs, "right", observation)

            # Parse gripper position for right arm
            self._parse_gripper_messages(right_gripper_msgs, "right", observation)

            # Capture camera images
            for cam_name, cam in self.cameras.items():
                observation[cam_name] = cam.async_read()

        except Exception as e:
            logger.error(f"Error capturing observation: {e}")
            raise

        return observation

    def _parse_joint_messages(self, joint_msgs, arm_prefix: str, observation: dict) -> None:
        """
        Parse joint messages from piper SDK format.
        Expected format includes Joint 1-6 values in the message.
        """
        try:
            # Convert message to string to parse it
            msg_str = str(joint_msgs)

            # Extract joint values using string parsing
            # Looking for patterns like "Joint 1:value", "Joint 2:value", etc.
            for i in range(1, 7):
                joint_key = f"{arm_prefix}_joint_{i}.pos"

                # Look for "Joint {i}:" pattern in the message
                pattern = f"Joint {i}:"
                start_idx = msg_str.find(pattern)

                if start_idx != -1:
                    # Find the value after "Joint {i}:"
                    value_start = start_idx + len(pattern)
                    # Find the end of the value (next newline or end of string)
                    value_end = msg_str.find("\n", value_start)
                    if value_end == -1:
                        value_end = len(msg_str)

                    value_str = msg_str[value_start:value_end].strip()
                    try:
                        observation[joint_key] = float(value_str)
                    except ValueError:
                        logger.warning(f"Could not parse joint {i} value: {value_str}")
                        observation[joint_key] = 0.0
                else:
                    logger.warning(f"Joint {i} not found in message")
                    observation[joint_key] = 0.0

        except Exception as e:
            logger.error(f"Error parsing joint messages: {e}")
            # Fallback: set all joints to 0
            for i in range(1, 7):
                observation[f"{arm_prefix}_joint_{i}.pos"] = 0.0

    def _parse_gripper_messages(self, gripper_msgs, arm_prefix: str, observation: dict) -> None:
        """
        Parse gripper messages from piper SDK format.
        Expected format: ArmGripper object with gripper_state.grippers_angle attribute
        grippers_angle is in 0.001mm units, needs conversion to mm.
        """
        try:
            # Access gripper_state.grippers_angle - convert from 0.001mm to mm
            angle_raw = gripper_msgs.gripper_state.grippers_angle
            angle_mm = float(angle_raw) / 1000.0
            observation[f"{arm_prefix}_gripper.pos"] = angle_mm
            logger.debug(f"{arm_prefix} gripper position: {angle_mm}mm (raw: {angle_raw})")

        except Exception as e:
            logger.error(f"Error parsing {arm_prefix} gripper messages: {e}")
            logger.error(f"Gripper message type: {type(gripper_msgs)}")
            logger.error(f"Gripper message content: {gripper_msgs}")
            observation[f"{arm_prefix}_gripper.pos"] = 0.0

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.action_features)}
        return action

    def _action_dict_to_action_list(self, action: dict[str, float]) -> list[float]:
        action_list = [action[key] for key in self.action_features]
        return action_list

    def send_action(self, action: list[float] | dict[str, any]) -> None:
        """Write the predicted actions from policy to the motors"""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )

        # if is dict, convert using _action_dict_to_action_tensor
        if isinstance(action, dict):
            action = self._action_dict_to_action_list(action)

        if len(action) != 14:
            raise ValueError(f"Expected 14-dim action, got {len(action)}")

        # send to motors, torch to list
        target_joints = action
        # data is first left then right
        left_joints = target_joints[:7] # left arm: 6 joints + 1 gripper
        right_joints = target_joints[7:] # right arm: 6 joints + 1 gripper

        self._write_to_motors(left_joints, self.left_arm)
        self._write_to_motors(right_joints, self.right_arm)

        return action

    def _write_to_motors(self, joint_positions: list, piper_arm: piper_interface.PiperInterface):
        """
        Write target joint positions to a single Piper arm
        
        Args:
            joint_positions: List of 7 values [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
            piper_arm: C_PiperInterface_V2 instance for the arm to control
        """
        if len(joint_positions) != 7:
            raise ValueError(f"Expected 7 joint values (6 joints + 1 gripper), got {len(joint_positions)}")
        
        try:
            # print(f"Target joint values: {target_joint}")
            # Extract joint positions (first 6 values)
            joint_1 = int(round(joint_positions[0]))
            joint_2 = int(round(joint_positions[1]))
            joint_3 = int(round(joint_positions[2]))
            joint_4 = int(round(joint_positions[3]))
            joint_5 = int(round(joint_positions[4]))
            joint_6 = int(round(joint_positions[5]))
            gripper_position = joint_positions[6]
            
            # Joint positions expected in radians by PiperInterface
            # piper_arm.command_joint_positions(joint_positions)
            piper_arm.piper.JointCtrl(joint_1, joint_2, joint_3, joint_4, joint_5, joint_6)
            # Gripper expects meters position
            # piper_arm.command_gripper(position=float(gripper_position))
            gripper_angle = int(round(gripper_position * 1000.0))
            piper_arm.piper.GripperCtrl(gripper_angle, 2000, 0x01, 0)

            # print(f"Sent joint commands")

                        
        except Exception as e:
            logger.error(f"Error writing to motors: {e}")
            logger.error(f"Target joint values: {joint_positions}")
            raise