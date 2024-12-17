import trimesh
import wandb


def scene_to_wandb_image(scene: trimesh.Scene) -> wandb.Image:
    """
    Log a colored front view of a trimesh scene using PyVista's off-screen rendering.
    Returns a low-res wandb.Image for basic visualization.
    """
    import pyvista as pv
    import numpy as np

    # Convert trimesh scene to PyVista
    plotter = pv.Plotter(off_screen=True)

    # Add each mesh from the scene
    for geometry in scene.geometry.values():
        if hasattr(geometry, "vertices") and hasattr(geometry, "faces"):
            # Convert trimesh to PyVista
            mesh = pv.PolyData(
                geometry.vertices,
                np.hstack([[3] + face.tolist() for face in geometry.faces]),
            )

            # Handle color
            if hasattr(geometry, "visual") and hasattr(geometry.visual, "face_colors"):
                face_colors = geometry.visual.face_colors
                if face_colors is not None:
                    # Convert RGBA to RGB if needed
                    if face_colors.shape[1] == 4:
                        face_colors = face_colors[:, :3]
                    mesh.cell_data["colors"] = face_colors
                    plotter.add_mesh(mesh, scalars="colors", rgb=True)
            else:
                # Default color if no colors specified
                plotter.add_mesh(mesh, color="lightgray")

    # # Set a very low resolution
    plotter.window_size = [1024, 1024]

    # Get the image array
    img_array = plotter.screenshot(return_img=True)

    # Close the plotter
    plotter.close()

    return wandb.Image(img_array)
