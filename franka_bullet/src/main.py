import time
from grasp_tester import PandaGraspTester
from utils import (
    find_object_matches,
    extract_object_info,
    process_mesh,
    load_grasp_data,
    calculate_transformations,
    set_object_pose,
)

def main():
    # Find matches and extract information
    matches = find_object_matches()
    h5_path = matches[0][1]
    object_info = extract_object_info(h5_path)
    mesh_scale_fg = object_info["scale"]#scale from h5 file, mesh scale from the name more detailed
    mesh_scale = float(matches[0][1].split("_")[-1].split(".h5")[-2])

    # Process mesh and load grasp data
    mesh_fname = process_mesh(h5_path, mesh_scale_fg)
    grasp_T, success = load_grasp_data(h5_path)

    # Initialize robot
    robot = PandaGraspTester(
        object_filepath=mesh_fname,
        mesh_scale=mesh_scale,
        realtime=1,
    )

    # Calculate and apply transformations
    world2ctr_T = calculate_transformations(robot, grasp_T)
    set_object_pose(robot, world2ctr_T)

    # Trimesh debugging section does not work with pybullet at the same time
    # from acronym_tools import create_gripper_marker
    # import trimesh
    # mesh = trimesh.load(mesh_fname)
    # grasp = [
    #     create_gripper_marker(
    #         color=[0, 255, 0], tube_radius=0.003
    #     ).apply_transform(obj2rotgrasp_T)
    # ]
    # trimesh.Scene([mesh] + grasp).show()

    # Run simulation
    print("Letting simulation settle...")
    for _ in range(100):
        robot.step()
        if robot.realtime:
            time.sleep(robot.stepsize)

    robot.close_gripper()

    while True:
        robot.step()
        if robot.realtime:
            time.sleep(robot.stepsize)

if __name__ == "__main__":
    main()