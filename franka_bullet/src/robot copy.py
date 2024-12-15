import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from typing import List, Tuple
import numpy as np
import pybullet as p
import time
import os

from utils import rot2quat, PandaConfig


def log_joint_positions(robot_id: int, joint_ids: List[int]):
    """Log current joint positions"""
    states = p.getJointStates(robot_id, joint_ids)
    positions = [state[0] for state in states]
    for idx, pos in enumerate(positions):
        print(f"Joint {idx}: {pos:.4f} rad ({pos * 180 / np.pi:.1f} deg)")
    return positions


class PandaRobot:
    """Panda Robot simulation with grasping functionality"""

    def __init__(self, config: PandaConfig):
        """Initialize the Panda robot simulation"""
        self.config = config
        self.t = 0.0
        self.test_object = None

        print("\nInitializing PandaRobot simulation...")
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=0.8,
            cameraYaw=30,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0.2],
        )

        self.initialize_simulation()

    def initialize_simulation(self, gravity: bool = False) -> None:
        """Initialize simulation, robot, and test object"""
        p.resetSimulation()
        p.setTimeStep(self.config.stepsize)
        p.setRealTimeSimulation(self.config.realtime)
        p.setGravity(0, 0, -9.81 if gravity else 0)
        print(f"Gravity: {'enabled' if gravity else 'disabled'}")

        # Load models
        p.setAdditionalSearchPath("../models")
        self.plane = p.loadURDF("plane/plane.urdf", useFixedBase=True)
        self.robot = p.loadURDF(
            "panda/panda_gripper.urdf", useFixedBase=True, basePosition=[0.5, 0, 0]
        )
        print("Models loaded successfully")

        # Setup joint IDs and limits
        self.arm_joint_ids = list(range(7))
        self.gripper_joint_ids = [7, 8]
        self.ee_id = 8

        # Get joint limits
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        for i in self.arm_joint_ids:
            joint_info = p.getJointInfo(self.robot, i)
            self.joint_limits_lower.append(joint_info[8])
            self.joint_limits_upper.append(joint_info[9])

        print("Joint limits initialized:")
        for i, (lower, upper) in enumerate(
            zip(self.joint_limits_lower, self.joint_limits_upper)
        ):
            print(f"Joint {i}: [{lower:.2f}, {upper:.2f}] rad")

        # Initialize robot to middle of joint ranges
        self.t = 0.0
        initial_poses = [
            (lower + upper) / 2.0
            for lower, upper in zip(self.joint_limits_lower, self.joint_limits_upper)
        ]

        print("\nSetting initial joint positions (middle of ranges):")
        for i, pose in enumerate(initial_poses):
            print(f"Joint {i}: {pose:.4f} rad ({pose * 180 / np.pi:.1f} deg)")

        # Set initial joint positions
        for joint_id, pose in zip(self.arm_joint_ids, initial_poses):
            p.resetJointState(self.robot, joint_id, pose)

        # Set initial gripper position
        for joint_id in self.gripper_joint_ids:
            p.resetJointState(self.robot, joint_id, self.config.max_grip_aperture)

        print("\nVerifying initial position:")
        log_joint_positions(self.robot, self.arm_joint_ids)

        # Load test object if filepath provided
        if self.config.object_filepath is None or not os.path.exists(
            self.config.object_filepath
        ):
            print("No test object provided")
            raise FileNotFoundError("Test object mesh not found")

        # Create visual and collision shapes
        vis_shape = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=self.config.object_filepath,
            meshScale=[self.config.mesh_scale] * 3,
            rgbaColor=[0.8, 0.8, 0.8, 1],
        )

        col_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=self.config.object_filepath,
            meshScale=[self.config.mesh_scale] * 3,
        )

        # Create the multibody using config parameters
        self.test_object = p.createMultiBody(
            baseMass=self.config.object_mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=self.config.object_com,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        )

        # Set dynamics properties
        p.changeDynamics(
            self.test_object,
            -1,
            lateralFriction=self.config.object_friction,
            spinningFriction=0.05,
            rollingFriction=0.05,
            restitution=0.2,
            contactStiffness=5000,
            contactDamping=50,
            maxJointVelocity=100,
        )

        # Set inertia from config
        p.changeDynamics(
            self.test_object,
            -1,
            mass=self.config.object_mass,
            lateralFriction=self.config.object_friction,
            spinningFriction=0.05,
            rollingFriction=0.05,
            restitution=0.2,
            contactStiffness=5000,
            contactDamping=50,
            maxJointVelocity=100,
            localInertiaDiagonal=np.diag(self.config.object_inertia),
        )

        for _ in range(50):
            self.step()

    def step(self) -> None:
        """Step the simulation forward"""
        self.t += self.config.stepsize
        p.stepSimulation()
        if self.config.realtime:
            time.sleep(self.config.stepsize)

    def get_joint_states(self) -> Tuple[List[float], List[float]]:
        """Get current joint positions and velocities for arm joints"""
        states = p.getJointStates(self.robot, self.arm_joint_ids)
        positions = [state[0] for state in states]
        velocities = [state[1] for state in states]
        return positions, velocities

    def set_joint_positions(self, positions: List[float]) -> None:
        """Set target joint positions for arm joints using position control"""
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.arm_joint_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=positions,
            forces=self.config.max_torque,
            positionGains=self.config.position_control_gain_p,
            velocityGains=self.config.position_control_gain_d,
        )

    def SE3_to_joint_positions(self, SE3: np.ndarray) -> List[float]:
        """Convert SE3 pose to joint positions using inverse kinematics"""
        print("\nCalculating inverse kinematics...")

        SE3_copy = SE3.copy()

        SE3_copy[:3, 3] = SE3_copy[:3, 3] - np.array([0.5, 0, 0])

        position = SE3_copy[:3, 3]
        rotation = SE3_copy[:3, :3]
        quaternion = rot2quat(rotation)

        print(f"Target position: {position}")
        print(f"Target quaternion: {quaternion}")

        current_poses = self.get_joint_states()[0]

        try:
            joint_positions = p.calculateInverseKinematics(
                bodyUniqueId=self.robot,
                endEffectorLinkIndex=self.ee_id,
                targetPosition=position,
                targetOrientation=quaternion,
                lowerLimits=self.joint_limits_lower,
                upperLimits=self.joint_limits_upper,
                jointRanges=[
                    u - l
                    for u, l in zip(self.joint_limits_upper, self.joint_limits_lower)
                ],
                restPoses=current_poses,
                maxNumIterations=100,
                residualThreshold=1e-5,
            )

            return list(joint_positions[:7])
        except Exception as e:
            print(f"IK failed: {e}")
            raise

    def move_to_pose(self, SE3: np.ndarray, duration: float = 2.0) -> None:
        """Move to target SE3 pose with interpolation"""
        print("\nExecuting move to pose...")
        try:
            start_positions = self.get_joint_states()[0]
            target_positions = self.SE3_to_joint_positions(SE3)

            print("start_positions: ", start_positions)
            print("target_positions: ", target_positions)
            self.set_joint_positions(target_positions)
            self.step()

            print("last joint positions: ", self.get_joint_states()[0])

            # Verify final position
            final_pos, final_orn = p.getLinkState(self.robot, self.ee_id)[:2]

            print("\nFinal joint positions:")
            log_joint_positions(self.robot, self.arm_joint_ids)

        except Exception as e:
            print(f"Failed to move to pose: {e}")
            raise

    def close_gripper(self) -> bool:
        """Close gripper until force threshold or full closure"""
        print("\nClosing gripper...")
        max_steps = 1000
        step_count = 0
        total_force = 0  # Initialize outside the loop

        while step_count < max_steps:
            finger_states = p.getJointStates(self.robot, self.gripper_joint_ids)
            positions = [state[0] for state in finger_states]

            if all(abs(pos) >= self.config.max_grip_aperture for pos in positions):
                print("Gripper fully closed without detecting object")
                return False

            if self.test_object is not None:
                contact_points = p.getContactPoints(
                    bodyA=self.robot, bodyB=self.test_object
                )
                total_force = sum(pt[9] for pt in contact_points if pt[9] > 0)

                if total_force >= self.config.grip_force_threshold:
                    print(f"Object grasped with force: {total_force:.2f}N")
                    return True

            p.setJointMotorControlArray(
                bodyUniqueId=self.robot,
                jointIndices=self.gripper_joint_ids,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[-self.config.finger_velocity] * 2,
                forces=[self.config.max_torque[0]] * 2,
            )

            self.step()
            step_count += 1

        print("Gripper closure timed out")
        return False

    def check_grasp(self) -> bool:
        """Check if object is still in contact with both fingers"""
        if self.test_object is None:
            return False

        contact_points = p.getContactPoints(bodyA=self.robot, bodyB=self.test_object)
        finger_contacts = [False, False]

        for contact in contact_points:
            if contact[3] in self.gripper_joint_ids:
                finger_contacts[self.gripper_joint_ids.index(contact[3])] = True

        print("\nGrasp check details:")
        print(f"Left finger contact: {finger_contacts[0]}")
        print(f"Right finger contact: {finger_contacts[1]}")

        contact_status = all(finger_contacts)
        print(f"Overall grasp status: {contact_status}")
        return contact_status

    def test_grasp(self, SE3: np.ndarray) -> bool:
        """Execute and test a grasp at the given SE3 pose"""
        print("\nTest grasp execution...")
        try:
            # Move to pre-grasp position
            # pre_grasp_SE3 = SE3.copy()
            # pre_grasp_SE3[2, 3] += 0.1  # Offset in z direction
            # print("\nMoving to pre-grasp position...")
            # print(f"Pre-grasp position: {pre_grasp_SE3[:3, 3]}")
            # self.move_to_pose(pre_grasp_SE3)

            # Move to grasp position
            print("\nMoving to grasp position...")
            print(f"Grasp position: {SE3[:3, 3]}")
            self.move_to_pose(SE3)

            # Close gripper
            if not self.close_gripper():
                print("Grasp failed: Unable to establish firm grip")
                return False

            # Enable gravity for testing
            print("\nEnabling gravity for grasp test...")
            p.setGravity(0, 0, -9.81)

            # Let the system settle
            print("Letting system settle...")
            for i in range(100):
                self.step()
                if i % 20 == 0:  # Print every 20 steps
                    if self.test_object:
                        pos, orn = self.get_object_pose()

            # Final grasp check
            success = self.check_grasp()
            print(f"\nGrasp test {'successful' if success else 'failed'}")

            if success:
                # Get and print final object pose
                final_pos, final_orn = self.get_object_pose()
                print("\nFinal object state:")
                print(f"Position: {final_pos}")
                print(f"Orientation: {final_orn}")

            return success

        except Exception as e:
            print(f"Grasp testing failed: {e}")
            return False

    def get_object_pose(self) -> Tuple[List[float], List[float]]:
        """Get current pose of the test object"""
        if self.test_object is None:
            raise ValueError("No test object loaded")
        pos, orn = p.getBasePositionAndOrientation(self.test_object)
        return list(pos), list(orn)
