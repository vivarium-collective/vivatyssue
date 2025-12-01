from cmath import sqrt
import numpy as np
import pandas as pd
import math

from .sheet_geometry import SheetGeometry
from .utils import rotation_matrix, rotation_matrices


class VesselGeometry(SheetGeometry):
    
    @staticmethod     
    def update_boundary_index(sheet):
    # Reset boundary flags
        sheet.vert_df['boundary'] = 0
        sheet.edge_df['boundary'] = 0
        sheet.face_df['boundary'] = 0

        # Update opposite edges
        sheet.get_opposite()

        # Identify boundary edges

    # Boolean mask for boundary edges
        boundary_edges = sheet.edge_df['opposite'].eq(-1)
        sheet.edge_df.loc[boundary_edges, 'boundary'] = 1

        # Set boundary vertices
        boundary_verts = sheet.edge_df.loc[boundary_edges, 'trgt'].unique()
        sheet.vert_df.loc[boundary_verts, 'boundary'] = 1

        # Set boundary faces
        boundary_faces = sheet.edge_df.loc[boundary_edges, 'face']
        sheet.face_df.loc[boundary_faces.dropna().unique().astype(int), 'boundary'] = 1

    @staticmethod
    def update_tangents(sheet):
        # Extract xy coordinates as numpy array, in current index order
        verts = sheet.vert_df[sheet.coords].to_numpy(copy=False)

        # Build xyz vectors with z = 0
        vert_coords = np.column_stack((verts, np.zeros(len(verts))))

        # Tangent = cross((x,y,0), (0,0,1)) = (y, -x, 0)
        tangent = np.empty_like(vert_coords)
        tangent[:, 0] = vert_coords[:, 1]  # tx =  y
        tangent[:, 1] = -vert_coords[:, 0]  # ty = -x
        tangent[:, 2] = 0  # tz =  0

        # Normalize
        lengths = np.linalg.norm(tangent, axis=1)
        tangent /= lengths[:, None]

        # Assign back to vert_df (pandas aligns on index)
        sheet.vert_df["tx"] = tangent[:, 0]
        sheet.vert_df["ty"] = tangent[:, 1]
        sheet.vert_df["tz"] = tangent[:, 2]

    @classmethod
    def update_all(cls, sheet):
        super().update_all(sheet)
        cls.update_tangents(sheet)
        cls.update_boundary_index(sheet)         

def face_svd_(faces):

    rel_pos = faces[["rx", "ry", "rz"]]
    _, _, rotation = np.linalg.svd(rel_pos.astype(float), full_matrices=False)
    return rotation
