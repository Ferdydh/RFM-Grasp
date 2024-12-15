from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


def rot2quat(R: np.ndarray) -> List[float]:
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


@dataclass
class PandaConfig:
    """Configuration for Panda Robot simulation and control

    Contains parameters for:
    1. Simulation settings (timestep, realtime)
    2. Robot control (gains, torque limits)
    3. Gripper control (aperture, force threshold)
    4. Object properties (from grasp file)
    """

    # Simulation parameters
    stepsize: float = 1e-3  # Simulation timestep in seconds
    realtime: int = 1  # Whether to run in realtime (1) or as fast as possible (0)

    # Robot control parameters
    position_control_gain_p: List[float] = field(
        default_factory=lambda: [0.01] * 7
    )  # P gains for joint position control
    position_control_gain_d: List[float] = field(
        default_factory=lambda: [1.0] * 7
    )  # D gains for joint position control
    max_torque: List[float] = field(
        default_factory=lambda: [100.0] * 7
    )  # Maximum joint torques in Nm

    # Gripper control parameters
    max_grip_aperture: float = 0.08  # Maximum gripper opening in meters
    grip_force_threshold: float = 5.0  # Force threshold for grasp detection in N
    finger_velocity: float = 0.05  # Gripper closing velocity in m/s

    # Object parameters (set from grasp file)
    object_filepath: Optional[str] = None  # Path to object mesh file
    mesh_scale: float = 1.0  # Scaling factor for the mesh
    object_mass: float = 0.1  # Mass in kg
    object_friction: float = 0.8  # Friction coefficient
    # object_com: np.ndarray = field(  # Center of mass [x, y, z]
    #     default_factory=lambda: np.zeros(3)
    # )
    object_inertia: np.ndarray = field(  # 3x3 inertia matrix
        default_factory=lambda: np.eye(3)
    )

    @classmethod
    def from_grasp_file(cls, grasp_data: dict):
        """Create PandaConfig from grasp file data"""
        return cls(
            object_filepath=grasp_data["object/file"][()],
            mesh_scale=grasp_data["object/scale"][()],
            object_mass=grasp_data["object/mass"][()],
            object_friction=grasp_data["object/friction"][()],
            # object_com=grasp_data["object/com"][()],
            object_inertia=grasp_data["object/inertia"][()],
        )

    @classmethod
    def get_default(cls):
        """Return a PandaConfig instance with default values"""
        return cls()
