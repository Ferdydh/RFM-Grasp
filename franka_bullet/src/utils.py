import glob
import h5py
import numpy as np
import pybullet as p
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from acronym_tools import load_mesh
from typing import Dict, List, Tuple, Any
from numpy.typing import NDArray

def extract_object_info(h5_file_path: str) -> Dict[str, Any]:
    """Extract object properties from H5 file"""
    object_info = {}
    with h5py.File(h5_file_path, 'r') as h5file:
        obj_group = h5file['object']
        for scalar_key in ['density', 'friction', 'mass', 'scale', 'volume']:
            object_info[scalar_key] = float(obj_group[scalar_key][()])
        object_info['com'] = obj_group['com'][()].tolist()
        object_info['inertia'] = obj_group['inertia'][()].tolist()
        object_info['file'] = str(obj_group['file'][()])
    return object_info

def find_object_matches() -> List[Tuple[str, str]]:
    """Find matching object and grasp files."""
    objects = glob.glob("../models/*.obj")
    grasps = glob.glob("../models/*.h5")
    return [(obj, grasp) for obj in objects for grasp in grasps 
            if obj.split(".")[-2].split("/")[-1] in grasp]

def process_mesh(h5_path: str, mesh_scale_fg: float) -> str:
    """Process and export mesh, return mesh filename."""
    obj_mesh = load_mesh(filename=h5_path, mesh_root_dir="../models", scale=mesh_scale_fg)
    mesh_fname = "../models/exported_mesh.obj"
    obj_mesh.export(mesh_fname, file_type="obj")
    return mesh_fname

def load_grasp_data(h5_path: str) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Load grasp transform and success data."""
    with h5py.File(h5_path, 'r') as h5file:
        grasp_T = h5file['grasps']['transforms'][2,:,:]
        success = h5file["grasps/qualities/flex/object_in_gripper"]
    return grasp_T, success

def calculate_transformations(robot: Any, grasp_T: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate all necessary transformations."""
    pos, orn = p.getLinkState(robot.robot, robot.ee_id)[:2]
    world2ee_T = pt.transform_from_pq(np.concatenate([pos, pr.quaternion_wxyz_from_xyzw(orn)]))
    rotgrasp2grasp_T = pt.transform_from(
        pr.matrix_from_axis_angle([0, 0, 1, -np.pi / 2]), [0, 0, 0]
    )
    
    obj2grasp_T = grasp_T @ rotgrasp2grasp_T
    world2ctr_T = world2ee_T @ np.linalg.inv(obj2grasp_T)
    
    return world2ctr_T

def set_object_pose(robot: Any, transform: NDArray[np.float64]) -> None:
    """Set object position and orientation."""
    pq = pt.pq_from_transform(transform)
    p.resetBasePositionAndOrientation(
        robot.test_object, pq[:3], pr.quaternion_xyzw_from_wxyz(pq[3:])
    )
    p.resetBaseVelocity(robot.test_object, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
