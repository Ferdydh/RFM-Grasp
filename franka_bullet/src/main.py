import numpy as np
from grasp_tester import PandaGraspTester
import time
import os


def main():
    # Define test object path
    object_filepath = os.path.join("../models", "basket.obj")

    mesh_scale = 0.0093761727
    SE3 = np.array(
        [
            [0.1479, -0.9865, 0.0702, 0.4750],
            [-0.9865, -0.1522, -0.0604, 0.0168],
            [0.0702, -0.0604, -0.9957, 0.2541],
            [0.0000, 0.0000, 0.0000, 1],
        ]
    )

    # Initialize robot with test object - much larger scale
    robot = PandaGraspTester(
        object_filepath=object_filepath,
        mesh_scale=mesh_scale,  # Increased from 0.001 to 0.05
        realtime=1,
    )

    # Let simulation settle
    print("Letting simulation settle...")
    for _ in range(100):
        robot.step()
        if robot.realtime:
            time.sleep(robot.stepsize)

    # Get current object pose
    obj_pos, obj_orn = robot.get_object_pose()
    print(f"\nObject position: {obj_pos}")
    print(f"Object orientation: {obj_orn}")

    # Run grasp test
    success = robot.test_grasp(SE3)
    print(f"\nFinal Result: Grasp test {'succeeded' if success else 'failed'}")

    # Keep simulation running
    print("\nKeeping simulation alive for visualization...")
    while True:
        robot.step()
        if robot.realtime:
            time.sleep(robot.stepsize)


if __name__ == "__main__":
    main()
