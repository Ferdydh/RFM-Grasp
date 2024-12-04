import typer
from pathlib import Path
from typing import Optional
import logging
import time

from data.data_manager import DataManager, DataSelector
from data.visualizer import GraspVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize(
    path: Path = typer.Option(
        "data/",
        help="Path to data file",
    ),
    object_id: Optional[str] = typer.Option(
        "56f2cfa7d89ef32a5eef6c5d029c7274",
        help="Specific object ID to visualize. Overrides config if provided.",
    ),
    num_grasps: int = typer.Option(
        20,
        help="Number of grasps to show per category (successful/failed).",
    ),
) -> None:
    """Visualize grasps for objects in the dataset."""

    start_time = time.time()

    logger.info(f"Starting visualization with path: {path}, object_id: {object_id}")

    # Create data selector
    selector = DataSelector(
        object_id=None,
        grasp_id="0.003873753800979896",
        item_name=None,
    )

    logger.info("Initializing visualizer...")
    # Initialize visualizer
    visualizer = GraspVisualizer(
        data_root=str(path),
        selectors=selector,
    )

    logger.info(
        f"Visualizer initialization took {time.time() - start_time:.2f} seconds"
    )

    # Check if any meshes were found
    mesh_paths = visualizer.data_manager.get_all_mesh_paths()
    logger.info(f"Found {len(mesh_paths)} meshes")
    if len(mesh_paths) == 0:
        logger.error("No meshes found! Check the data path and object ID")
        return

    logger.info("Starting visualization...")
    # Show visualization
    try:
        visualizer.show_grasps(num_grasps=num_grasps)
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        raise

    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
