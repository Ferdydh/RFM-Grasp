import h5py
import numpy as np
import time
import os

from robot import PandaConfig, PandaRobot


def main():
    data_path = "../../data/"
    # grasp_path = (
    #     "grasps/3Shelves_667bec36787219ba959d688c13bc4d6e_0.0015840344930117929.h5"
    # )

    # grasp_path = "grasps/Bear_345b945db322120d4cccbece4754c7cf_0.013295782781187739.h5"
    grasp_path = "grasps/Desk_a02b052927a00a58568f5d1b69b4d09f_0.0015754220237984534.h5"
    # grasp_path = (
    #     "grasps/Bucket_997f1218df2b1909ce0e7190fd762087_0.009372396151638597.h5"
    # )
    # grasp_path = (
    #     "grasps/Chaise_a9ec4ed3925dc1482db431502a680805_0.0018421146734938327.h5"
    # )

    with h5py.File(
        os.path.join(data_path, grasp_path),
        "r",
    ) as h5file:
        config = PandaConfig.from_grasp_file(h5file)
        transforms = h5file["grasps"]["transforms"][:]
        grasp_success = h5file["grasps"]["qualities"]["flex"]["object_in_gripper"][:]
        transforms = transforms[grasp_success == 1]

        SE3 = transforms[0]

    config.object_filepath = os.path.join(
        data_path, config.object_filepath.decode("utf-8")
    )

    # print(config)
    print("SE3: ", SE3)

    ####################################################################################################

    robot = PandaRobot(config)

    # Let simulation settle
    print("Letting simulation settle...")
    for _ in range(100):
        robot.step()

    # Get current object pose
    # obj_pos, obj_orn = robot.get_object_pose()
    # print(f"\nObject position: {obj_pos}")
    # print(f"Object orientation: {obj_orn}")

    # Run grasp test
    success = robot.test_grasp(SE3)
    print(f"\nFinal Result: Grasp test {'succeeded' if success else 'failed'}")

    # Keep simulation running
    print("\nKeeping simulation alive for visualization...")
    while True:
        robot.step()


if __name__ == "__main__":
    main()
