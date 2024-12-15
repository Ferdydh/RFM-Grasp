import numpy as np
import pybullet as p
import time
import os
from panda_base import PandaBase


class PandaGraspTester(PandaBase):
    def __init__(self, object_filepath, mesh_scale=1.0, stepsize=1e-3, obj_pos=[0,0,0],start_pos=None,realtime=0):
        # Set this here because of the complicated logic between reset() and load_test_object()
        # So this is a hacky way to set the max_grip_aperture to be used in reset()
        # We can make this code better by combinining both classes PandaBase and PandaGraspTester
        # But let's just keep it for now
        self.max_grip_aperture = 0.08  # 8cm maximum grip aperture
        self.start_pos = None#start_pos
        super().__init__(stepsize, realtime)
        # Modified gripper parameters
        self.grip_force_threshold = 5.0  # N
        self.finger_velocity = 0.05  # m/s for closing

        # Object parameters
        self.object_filepath = object_filepath
        self.mesh_scale = mesh_scale
        #self.obj_pos = np.array(obj_pos)

        
        
        # Load test object
        self.load_test_object()

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
        #the joint 7 is the hand joint which is a fixed joint
        # Get gripper joint IDs
        self.gripper_joint_ids = [8, 9]

        # Get joint limits
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        for i in self.arm_joint_ids:
            joint_info = p.getJointInfo(self.robot, i)
            self.joint_limits_lower.append(joint_info[8])
            self.joint_limits_upper.append(joint_info[9])

        self.ee_id = 7  # End effector link ID of palm

        self.reset()

    def reset(self):
        """Reset robot to initial configuration"""
        self.t = 0.0
        if self.start_pos is not None:
            initial_poses = self.SE3_to_joint_positions(self.start_pos)
        else:
            initial_poses = [0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785]
            initial_gripper_poses = [self.max_grip_aperture/2]*2
            print(initial_gripper_poses)

        # Reset arm joints
        for joint_id, pose in zip(self.arm_joint_ids, initial_poses):
            p.resetJointState(self.robot, joint_id, pose)

        # Reset gripper joints
        for joint_id,gripper_pose in zip(self.gripper_joint_ids,initial_gripper_poses):
            p.resetJointState(self.robot, joint_id, gripper_pose)

    def load_test_object(self):
        """Load the test object mesh"""
        print(f"\nAttempting to load mesh from: {self.object_filepath}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Does file exist? {os.path.exists(self.object_filepath)}")

        if not os.path.exists(self.object_filepath):
            raise FileNotFoundError(
                f"Test object mesh not found at: {self.object_filepath}"
            )

        try:
            # Create visual shape
            vis_shape = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=self.object_filepath,
                meshScale=[1]*3,#[self.mesh_scale] * 3,
                rgbaColor=[0.8, 0.8, 0.8, 1],
            )
            print(f"Visual shape ID: {vis_shape}")

            # Create collision shape
            col_shape = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=self.object_filepath,
                meshScale=[1]*3,#[self.mesh_scale] * 3,
            )
            print(f"Collision shape ID: {col_shape}")

            # TODO: Adjust this so the grasp is successful
            base_pos = [-0.2, -0.05, 0.10]  # Matches the SE3 target position
            base_orn = p.getQuaternionFromEuler([0, 0, 0])  # No rotation

            # Create the multibody with the new position
            self.test_object = p.createMultiBody(
                baseMass=2,  # Light mass
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                #basePosition=self.obj_pos,
                #baseOrientation=base_orn,
            )
            print(f"Object ID: {self.test_object}")
            #print(f"Object position: {self.obj_pos}")

            # Set appropriate friction coefficients
            p.changeDynamics(
                self.test_object,
                -1,
                lateralFriction=1,
                spinningFriction=1,#0.05,
                rollingFriction=1,#0.05,
                restitution=0.2,
                contactStiffness=5000,
                contactDamping=50,
                maxJointVelocity=100,
            )

            # Let the object settle briefly
            for _ in range(50):
                p.stepSimulation()

        except p.error as e:
            print(f"PyBullet error during object loading: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error during object loading: {e}")
            raise

    def step(self):
        """Step the simulation forward"""
        self.t += self.stepsize
        p.stepSimulation()
        if self.realtime:
            time.sleep(self.stepsize)

    def close_gripper(self):
        """Close gripper until force threshold or full closure"""
        max_steps = 1000  # Prevent infinite loop
        step_count = 0

        while step_count < max_steps:
            # Get current finger positions
            finger_states = p.getJointStates(self.robot, self.gripper_joint_ids)
            positions = [state[0] for state in finger_states]

            # Check if fingers are fully closed
            if all(abs(pos) >= self.max_grip_aperture for pos in positions):
                return False  # Failed to grasp object

            # Get contact forces
            contact_points = p.getContactPoints(
                bodyA=self.robot, bodyB=self.test_object
            )
            total_force = sum(pt[9] for pt in contact_points if pt[9] > 0)

            if total_force >= self.grip_force_threshold:
                return True  # Successfully grasped object

            # Continue closing fingers
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot,
                jointIndices=self.gripper_joint_ids,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[-self.finger_velocity, -self.finger_velocity],
                forces=[self.max_torque[0]] * 2,
            )

            self.step()
            step_count += 1

        return False  # Timeout without achieving grasp

    def shake_test(self, approach_distance=0.1, rotation_angle=np.pi / 4):
        """Perform shaking motion test"""
        # Get current end effector pose
        ee_state = p.getLinkState(self.robot, self.ee_id)
        initial_pos = ee_state[0]
        initial_orn = ee_state[1]

        # Up and down motion along approach direction
        for direction in [1, -1, 1]:  # Up, down, back to center
            target_pos = (
                initial_pos[0],
                initial_pos[1],
                initial_pos[2] + direction * approach_distance,
            )
            self._move_ee_to_pose(target_pos, initial_orn)

            # Check grasp stability after each movement
            if not self.check_grasp():
                return False

        # Rotation around prismatic joint axes
        for angle in [rotation_angle, -2 * rotation_angle, rotation_angle]:
            current_orn = p.getEulerFromQuaternion(initial_orn)
            target_orn = p.getQuaternionFromEuler(
                [current_orn[0], current_orn[1], current_orn[2] + angle]
            )
            self._move_ee_to_pose(initial_pos, target_orn)

            # Check grasp stability after each movement
            if not self.check_grasp():
                return False

        return True

    def _move_ee_to_pose(self, position, orientation, duration=1.0):
        """Helper method to move end effector to a specific pose"""
        steps = int(duration / self.stepsize)
        for _ in range(steps):
            # Use IK to get joint positions
            joint_positions = p.calculateInverseKinematics(
                self.robot, self.ee_id, position, orientation
            )

            # Set joint positions for arm
            self.set_joint_positions(joint_positions[:7])
            self.step()



    def check_grasp(self):
        """Check if object is still in contact with both fingers"""
        contact_points = p.getContactPoints(bodyA=self.robot, bodyB=self.test_object)
        finger_contacts = [False, False]

        for contact in contact_points:
            if contact[3] in self.gripper_joint_ids:
                finger_contacts[self.gripper_joint_ids.index(contact[3])] = True

        return all(finger_contacts)

    def test_grasp(self, SE3):
        """Execute complete grasp testing sequence"""
        print("Starting grasp test sequence...")

        try:
            # 0. Move to SE3 pose
            print("Moving to grasp pose...")
            self.move_to_pose(SE3)

            # 1. Close fingers
            print("Closing gripper...")
            if not self.close_gripper():
                print("FAIL: Could not establish firm grasp")
                return False

            # Enable gravity for the shake test
            p.setGravity(0, 0, -9.81)
            print("Enabled gravity for stability testing")

            # Let the system settle
            for _ in range(100):
                self.step()

            # 2. Perform shake test
            print("Performing shake test...")
            if not self.shake_test():
                print("FAIL: Object lost during shake test")
                return False

            print("PASS: Grasp maintained throughout testing")
            return True

        except Exception as e:
            print(f"Error during grasp test: {e}")
            return False

    def get_object_pose(self):
        """Get current pose of the test object"""
        pos, orn = p.getBasePositionAndOrientation(self.test_object)
        return pos, orn

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
