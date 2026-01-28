from cmath import sqrt
import numpy as np
import pandas as pd
import math

from .sheet_geometry import SheetGeometry
from .utils import rotation_matrix, rotation_matrices


class VesselGeometry(SheetGeometry):

    @staticmethod
    def update_tangents(sheet):
        # Extract coordinates as NumPy arrays
        x = sheet.vert_df['x'].to_numpy()
        y = sheet.vert_df['y'].to_numpy()

        # Analytical Cross Product
        tx = y
        ty = -x

        # Calculate Length (Vectorized)
        length = np.hypot(tx, ty)

        # Normalize
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_length = 1.0 / length

        # 5. Assign directly to DataFrame
        # This assumes sheet.coords = ['x', 'y', 'z'], matching the original output naming
        sheet.vert_df['tx'] = tx * inv_length
        sheet.vert_df['ty'] = ty * inv_length
        sheet.vert_df['tz'] = 0.0  # Z-component is always 0 in this projection

    @staticmethod
    def update_vert_distance(sheet):
        distances = np.sqrt(sheet.vert_df['x'].to_numpy()**2 + sheet.vert_df['y'].to_numpy()**2)
        sheet.vert_df['distance_origin'] = distances
        sheet.vert_df["ox"] = sheet.vert_df["x"]/distances
        sheet.vert_df["oy"] = sheet.vert_df["y"]/distances

    @classmethod
    def update_all(cls, sheet):
        super().update_all(sheet)
        cls.update_tangents(sheet)
        cls.update_vert_distance(sheet)

def face_svd_(faces):

    rel_pos = faces[["rx", "ry", "rz"]]
    _, _, rotation = np.linalg.svd(rel_pos.astype(float), full_matrices=False)
    return rotation
