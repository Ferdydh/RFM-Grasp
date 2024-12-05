from data.grasp_dataset import GraspDataset


def visualize(experiment: str = "visualize"):
    """Visualize an existing grasp from the dataset."""
    test = GraspDataset(
        data_root="data/",
        split="test",
        grasp_files=["Xbox360_14e5dba73b283dc7fe0939859a0b15ea_0.0005312646125977.h5"],
        num_samples=5,
    )
