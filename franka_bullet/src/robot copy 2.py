import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from typing import List, Tuple
import numpy as np
import pybullet as p
import time
import os

from utils import rot2quat, PandaConfig


def log_joint_states(
    robot_id: int, joint_ids: List[int], prefix: str = ""
) -> Tuple[List[float], List[float], List[float]]:
    """Get joint positions, velocities and torques with logging"""
    states = p.getJointStates(robot_id, joint_ids)
    positions = [state[0] for state in states]
    velocities = [state[1] for state in states]
    torques = [state[3] for state in states]

    print(f"\n{prefix}Joint States:")
    for idx, (pos, vel, torque) in enumerate(zip(positions, velocities, torques)):
        print(
            f"Joint {idx}: pos={pos:.4f} rad ({pos * 180 / np.pi:.1f}°), vel={vel:.4f} rad/s, torque={torque:.4f} Nm"
        )

    return positions, velocities, torques


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
        log_joint_states(self.robot, self.arm_joint_ids, "Initial ")

        # Load test object
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

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get end-effector position and orientation"""
        pos, orn = p.getLinkState(self.robot, self.ee_id)[:2]
        return np.array(pos), np.array(orn)

    def SE3_to_joint_positions(self, SE3: np.ndarray) -> List[float]:
        """Convert SE3 pose to joint positions using inverse kinematics"""
        print("\nCalculating inverse kinematics...")

        T = np.eye(4)
        T[0, 3] = -0.5

        # Multiply with original SE3
        SE3 = T @ SE3  # Using matrix multiplication

        position = SE3[:3, 3]
        rotation = SE3[:3, :3]
        quaternion = rot2quat(rotation)

        print(f"Target position: {position}")
        print(f"Target quaternion: {quaternion}")

        current_positions = log_joint_states(
            self.robot, self.arm_joint_ids, "Current "
        )[0]

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
                restPoses=current_positions,
                maxNumIterations=100,
                residualThreshold=1e-5,
            )

            # Log IK solution quality
            print("\nChecking IK solution quality:")
            target_pos = np.array(position)
            target_quat = np.array(quaternion)

            # Set joints to IK solution and get resulting EE pose
            temp_positions = [
                p.getJointState(self.robot, i)[0] for i in self.arm_joint_ids
            ]
            for i, pos in zip(self.arm_joint_ids, joint_positions):
                p.resetJointState(self.robot, i, pos)
            achieved_pos, achieved_quat = self.get_ee_pose()

            # Restore original joint positions
            for i, pos in zip(self.arm_joint_ids, temp_positions):
                p.resetJointState(self.robot, i, pos)

            # Calculate and log errors
            pos_error = np.linalg.norm(target_pos - achieved_pos)
            quat_error = np.arccos(np.abs(np.dot(target_quat, achieved_quat)))
            print(f"Position error: {pos_error:.6f} m")
            print(
                f"Orientation error: {quat_error:.6f} rad ({quat_error * 180 / np.pi:.2f}°)"
            )

            return list(joint_positions[:7])
        except Exception as e:
            print(f"IK failed: {e}")
            raise

    def move_to_pose(self, SE3: np.ndarray, duration: float = 2.0) -> None:
        """Move to target SE3 pose"""
        print("\nExecuting move to pose...")
        try:
            start_positions = log_joint_states(
                self.robot, self.arm_joint_ids, "Start "
            )[0]
            target_positions = self.SE3_to_joint_positions(SE3)

            print("\nTarget joint positions:")
            for i, pos in enumerate(target_positions):
                print(f"Joint {i}: {pos:.4f} rad ({pos * 180 / np.pi:.1f}°)")

            self.set_joint_positions(target_positions)
            self.step()

            # Log final position and compare with target
            final_positions, final_velocities, final_torques = log_joint_states(
                self.robot, self.arm_joint_ids, "Final "
            )

            # Calculate and log position errors
            print("\nJoint position errors:")
            for i, (target, actual) in enumerate(
                zip(target_positions, final_positions)
            ):
                error = target - actual
                print(f"Joint {i}: {error:.4f} rad ({error * 180 / np.pi:.1f}°)")

            # Get and log final end-effector pose
            final_pos, final_orn = self.get_ee_pose()
            print("\nFinal end-effector pose:")
            print(f"Position: {final_pos}")
            print(f"Orientation: {final_orn}")

        except Exception as e:
            print(f"Failed to move to pose: {e}")
            raise

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

    def close_gripper(self) -> bool:
        """Close gripper until force threshold or full closure"""
        print("\nClosing gripper...")
        max_steps = 1000
        step_count = 0

        # Log initial gripper state
        initial_states = log_joint_states(
            self.robot, self.gripper_joint_ids, "Initial gripper "
        )

        while step_count < max_steps:
            # Get current gripper state
            positions, velocities, torques = log_joint_states(
                self.robot, self.gripper_joint_ids, "Current gripper "
            )

            # Check if gripper is fully closed
            if all(abs(pos) >= self.config.max_grip_aperture for pos in positions):
                print("Gripper fully closed without detecting object")
                return False

            # Check contact forces if object exists
            if self.test_object is not None:
                contact_points = p.getContactPoints(
                    bodyA=self.robot, bodyB=self.test_object
                )

                # Log contact information
                print("\nContact information:")
                total_force = 0
                for pt in contact_points:
                    if pt[9] > 0:  # Normal force
                        print(f"Contact force: {pt[9]:.2f}N at position: {pt[5]}")
                        total_force += pt[9]

                if total_force >= self.config.grip_force_threshold:
                    print(f"\nObject grasped successfully")
                    print(f"Total contact force: {total_force:.2f}N")
                    # Log final gripper state
                    log_joint_states(
                        self.robot, self.gripper_joint_ids, "Final gripper "
                    )
                    return True

            # Command gripper to close
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot,
                jointIndices=self.gripper_joint_ids,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[-self.config.finger_velocity] * 2,
                forces=[self.config.max_torque[0]] * 2,
            )

            self.step()
            step_count += 1

            # Log progress every 100 steps
            if step_count % 100 == 0:
                print(f"\nGripper closure progress - Step {step_count}/{max_steps}")
                log_joint_states(self.robot, self.gripper_joint_ids, "Progress ")

        print("\nGripper closure timed out")
        return False

    def check_grasp(self) -> bool:
        """Check if object is still in contact with both fingers"""
        if self.test_object is None:
            return False

        # Get and log object state
        pos, orn = self.get_object_pose()
        print("\nObject state:")
        print(f"Position: {pos}")
        print(f"Orientation: {orn}")

        # Get contact information
        contact_points = p.getContactPoints(bodyA=self.robot, bodyB=self.test_object)
        finger_contacts = [False, False]
        contact_forces = [0, 0]

        # Analyze each contact point
        print("\nContact analysis:")
        for contact in contact_points:
            if contact[3] in self.gripper_joint_ids:
                idx = self.gripper_joint_ids.index(contact[3])
                finger_contacts[idx] = True
                contact_forces[idx] += contact[9]
                print(
                    f"Finger {idx}: Force = {contact_forces[idx]:.2f}N at position {contact[5]}"
                )

        print("\nGrasp status summary:")
        print(
            f"Left finger contact: {finger_contacts[0]} (Force: {contact_forces[0]:.2f}N)"
        )
        print(
            f"Right finger contact: {finger_contacts[1]} (Force: {contact_forces[1]:.2f}N)"
        )

        contact_status = all(finger_contacts)
        print(f"Overall grasp status: {'Stable' if contact_status else 'Unstable'}")
        return contact_status

    def test_grasp(self, SE3: np.ndarray) -> bool:
        """Execute and test a grasp at the given SE3 pose"""
        print("\nTest grasp execution...")
        try:
            # Log initial object pose
            initial_pos, initial_orn = self.get_object_pose()
            print("\nInitial object pose:")
            print(f"Position: {initial_pos}")
            print(f"Orientation: {initial_orn}")

            # Move to grasp position
            print("\nMoving to grasp position...")
            print(f"Target grasp position: {SE3[:3, 3]}")
            self.move_to_pose(SE3)

            # Close gripper
            if not self.close_gripper():
                print("Grasp failed: Unable to establish firm grip")
                return False

            # Enable gravity for testing
            print("\nEnabling gravity for grasp test...")
            p.setGravity(0, 0, -9.81)

            # Let the system settle and monitor object
            print("Letting system settle...")
            for i in range(100):
                self.step()
                if i % 20 == 0:  # Log every 20 steps
                    pos, orn = self.get_object_pose()
                    print(f"\nObject state at step {i}:")
                    print(f"Position: {pos}")
                    print(f"Orientation: {orn}")

            # Final grasp check
            success = self.check_grasp()
            print(f"\nGrasp test {'successful' if success else 'failed'}")

            if success:
                # Compare final pose with initial pose
                final_pos, final_orn = self.get_object_pose()
                print("\nObject pose comparison:")
                print("Position change:", np.array(final_pos) - np.array(initial_pos))
                print("Final orientation:", final_orn)

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
