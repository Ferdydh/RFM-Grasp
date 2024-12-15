import numpy as np
import pybullet as p
import time


class PandaBase:
    def __init__(self, stepsize=5e-4, realtime=0):
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime

        # Control parameters for the 7 arm joints
        self.position_control_gain_p = [0.01] * 7
        self.position_control_gain_d = [1.0] * 7
        self.max_torque = [100.0] * 7

        # Connect to pybullet
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=30,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0.5],
        )

        # Reset simulation with no gravity initially
        self.reset_simulation(gravity=False)

    def reset_simulation(self, gravity=False):
        """Reset simulation with option to enable/disable gravity"""
        p.resetSimulation()
        p.setTimeStep(self.stepsize)
        p.setRealTimeSimulation(self.realtime)
        p.setGravity(0, 0, -9.81 if gravity else 0)

        # Load models
        p.setAdditionalSearchPath("../models")
        self.plane = p.loadURDF("plane/plane.urdf", useFixedBase=True)
        self.robot = p.loadURDF("panda/panda_gripper.urdf", useFixedBase=True)

        # The first 7 joints are the arm joints
        self.arm_joint_ids = list(range(7))

        # Get gripper joint IDs (assuming they're the last two joints)
        self.gripper_joint_ids = [7, 8]  # Adjust these indices based on your URDF

        # Get joint limits
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        for i in self.arm_joint_ids:
            joint_info = p.getJointInfo(self.robot, i)
            self.joint_limits_lower.append(joint_info[8])
            self.joint_limits_upper.append(joint_info[9])

        self.ee_id = 7  # End effector link ID

        self.reset()

    def reset(self):
        """Reset robot to initial configuration"""
        self.t = 0.0
        initial_poses = [0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785]  # Home position

        # Reset arm joints
        for joint_id, pose in zip(self.arm_joint_ids, initial_poses):
            p.resetJointState(self.robot, joint_id, pose)

        # Reset gripper joints
        for joint_id in self.gripper_joint_ids:
            p.resetJointState(self.robot, joint_id, 0)

    def step(self):
        """Step the simulation forward"""
        self.t += self.stepsize
        p.stepSimulation()
        if self.realtime:
            time.sleep(self.stepsize)

    def get_joint_states(self):
        """Get current joint positions and velocities for arm joints"""
        states = p.getJointStates(self.robot, self.arm_joint_ids)
        positions = [state[0] for state in states]
        velocities = [state[1] for state in states]
        return positions, velocities

    def set_joint_positions(self, positions):
        """Set target joint positions for arm joints using position control"""
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.arm_joint_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=positions,
            forces=self.max_torque,
            positionGains=self.position_control_gain_p,
            velocityGains=self.position_control_gain_d,
        )

    @staticmethod
    def rot2quat(R):
        """Convert 3x3 rotation matrix to quaternion [x,y,z,w]"""
        trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / S
                qx = 0.25 * S
                qy = (R[0, 1] + R[1, 0]) / S
                qz = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / S
                qx = (R[0, 1] + R[1, 0]) / S
                qy = 0.25 * S
                qz = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / S
                qx = (R[0, 2] + R[2, 0]) / S
                qy = (R[1, 2] + R[2, 1]) / S
                qz = 0.25 * S
        return [qx, qy, qz, qw]
    


    def SE3_to_joint_positions(self, SE3):
        """Convert SE3 pose to joint positions using inverse kinematics"""
        position = SE3[:3, 3]
        rotation = SE3[:3, :3]
        quaternion = self.rot2quat(rotation)

        current_poses = self.get_joint_states()[0]

        try:
            joint_positions = p.calculateInverseKinematics(
                bodyUniqueId=self.robot,
                endEffectorLinkIndex=self.ee_id,
                targetPosition=position,
                targetOrientation=quaternion,
                # lowerLimits=self.joint_limits_lower,
                # upperLimits=self.joint_limits_upper,
                # jointRanges=[
                #     u - l
                #     for u, l in zip(self.joint_limits_upper, self.joint_limits_lower)
                # ],
                restPoses=current_poses,
                maxNumIterations=10000,
                residualThreshold=1e-7,
            )
            return list(joint_positions[:7])  # Return only arm joint positions
        except p.error as e:
            print(f"IK failed: {e}")
            print(f"Target position: {position}")
            print(f"Target orientation (quaternion): {quaternion}")
            raise

    def move_to_pose(self, SE3, duration=2.0):
        """Move to target SE3 pose with interpolation"""
        try:
            start_positions = self.get_joint_states()[0]
            target_positions = self.SE3_to_joint_positions(SE3)
            print("Target positions:", target_positions)

            steps = int(duration / self.stepsize)
            for i in range(steps):
                alpha = (i + 1) / steps
                interpolated_positions = [
                    start_pos + alpha * (target_pos - start_pos)
                    for start_pos, target_pos in zip(start_positions, target_positions)
                ]
                self.set_joint_positions(interpolated_positions)
                self.step()
                if i == steps - 1:
                    link_state = p.getLinkState(self.robot, self.ee_id)
                    pos = link_state[4]  # Get the position
                    orn = link_state[5]  # Get the orientation (quaternion)
                    rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
                    current_SE3 = np.eye(4)
                    current_SE3[:3, :3] = rot_matrix
                    current_SE3[:3, 3] = pos
                    print("Current SE3:", current_SE3)
                    print("Target pose", SE3)
                    print("Current joint pose",self.get_joint_states()[0])
        # Convert link state to SE3

        except Exception as e:
            print(f"Failed to move to pose: {e}")
            raise

    def move_to_pose(self, target_SE3, duration=2.0):
        """Move to target SE3 pose with Cartesian interpolation."""
        try:
            # Get start pose
            link_state = p.getLinkState(self.robot, self.ee_id)
            start_pos = np.array(link_state[4])
            start_orn = np.array(link_state[5])  # quaternion
            start_rot = np.array(p.getMatrixFromQuaternion(start_orn)).reshape(3, 3)

            # Create start SE3
            start_SE3 = np.eye(4)
            start_SE3[:3, :3] = start_rot
            start_SE3[:3, 3] = start_pos

            # Extract target position and rotation
            target_pos = target_SE3[:3, 3]
            target_rot = target_SE3[:3, :3]

            # Convert rotations to quaternions for SLERP
            start_quat = self.rot2quat(start_rot)
            target_quat = self.rot2quat(target_rot)

            steps = int(duration / self.stepsize)
            for i in range(steps):
                alpha = (i + 1) / steps

                # Interpolate position linearly
                pos = start_pos + alpha * (target_pos - start_pos)

                # Interpolate rotation using SLERP
                orn = p.getQuaternionSlerp(start_quat, target_quat, alpha)
                rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

                # Create interpolated SE3
                interpolated_SE3 = np.eye(4)
                interpolated_SE3[:3, :3] = rot
                interpolated_SE3[:3, 3] = pos

                # Convert to joint positions using IK
                target_joints = self.SE3_to_joint_positions(interpolated_SE3)
                self.set_joint_positions(target_joints)
                self.step()

                if i == steps - 1:
                    # Print final pose
                    final_state = p.getLinkState(self.robot, self.ee_id)
                    final_pos = final_state[4]
                    final_orn = final_state[5]
                    final_rot = np.array(p.getMatrixFromQuaternion(final_orn)).reshape(3, 3)
                    final_SE3 = np.eye(4)
                    final_SE3[:3, :3] = final_rot
                    final_SE3[:3, 3] = final_pos
                    print("Final SE3:", final_SE3)
                    print("Target SE3:", target_SE3)

        except Exception as e:
            print(f"Failed to move to pose: {e}")
            raise

