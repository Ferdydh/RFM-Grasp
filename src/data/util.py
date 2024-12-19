import logging
from typing import Tuple
import mesh2sdf
import numpy as np
import trimesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def enforce_trimesh(mesh) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Scene):
        # Get all meshes from the scene and preserve their transformations
        meshes = []
        for geometry_name, geometry in mesh.geometry.items():
            # Get the transform for this geometry
            transform_matrix = mesh.graph.get(geometry_name)[0]

            # Apply the transform to the vertices
            transformed_vertices = trimesh.transform_points(
                geometry.vertices, transform_matrix
            )

            # Create new mesh with transformed vertices
            transformed_mesh = trimesh.Trimesh(
                vertices=transformed_vertices, faces=geometry.faces
            )
            meshes.append(transformed_mesh)

        # Concatenate all transformed meshes
        return trimesh.util.concatenate(meshes)
    elif isinstance(mesh, trimesh.Trimesh):
        return mesh
    else:
        raise ValueError(f"Unsupported mesh type: {type(mesh)}")


def process_mesh_to_sdf(
    mesh: trimesh.Trimesh, size: int = 32
) -> Tuple[
    np.ndarray,  # sdf
    float,  # normalization_scale
    np.ndarray,  # centroid = list of 3 values
]:
    """Process a mesh to SDF with consistent scaling and centering."""
    centroid = mesh.centroid
    mesh.vertices = mesh.vertices - centroid
    normalization_scale = np.max(np.abs(mesh.vertices))
    mesh.vertices = mesh.vertices / normalization_scale

    mesh.vertices = mesh.vertices.astype(np.float32)
    mesh.faces = mesh.faces.astype(np.uint32)

    # First compute raw SDF
    raw_sdf = mesh2sdf.compute(mesh.vertices, mesh.faces, size)
    abs_sdf = np.abs(raw_sdf)

    # Choose a level value within the range of the absolute SDF
    level = (abs_sdf.min() + abs_sdf.max()) / 2

    # Compute final SDF with appropriate level
    sdf = mesh2sdf.compute(mesh.vertices, mesh.faces, size, fix=True, level=level)

    return sdf, normalization_scale, centroid
