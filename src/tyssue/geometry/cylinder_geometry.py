from cmath import sqrt
import numpy as np
import pandas as pd
import math

from .sheet_geometry import SheetGeometry

class CylinderGeometryInit(SheetGeometry):

    @staticmethod
    def update_boundary_index(sheet):

        sheet.vert_df['boundary'] = 0
        sheet.edge_df['boundary'] = 0

        sheet.get_opposite()

        sheet.edge_df.loc[sheet.edge_df['opposite'] == -1, 'boundary'] = 1
        boundary_verts = sheet.edge_df.loc[sheet.edge_df['opposite'] == -1, 'trgt'].to_numpy()

        sheet.vert_df.loc[boundary_verts, "boundary"] = 1

    @staticmethod
    def update_tangents(sheet):

        vert_coords = sheet.vert_df[sheet.coords]
        vert_coords.loc[:, "z"] = 0
        vert_coords = vert_coords.values
        normal = np.column_stack((np.zeros(sheet.Nv), np.zeros(sheet.Nv), np.ones(sheet.Nv)))

        tangent = np.cross(vert_coords, normal)
        tangent = pd.DataFrame(tangent)

        tangent.columns = ["t" + u for u in sheet.coords]

        length = pd.DataFrame(tangent.eval("sqrt(tx**2 + ty**2 +tz**2)"), columns=['length'])
        tangent["length"] = length["length"]

        tangent = tangent[['tx', 'ty', 'tz']].div(length.length, axis=0)

        for u in sheet.coords:
            sheet.vert_df["t" + u] = tangent["t" + u]

    @staticmethod
    def update_face_tangents(sheet):

        face_coords = sheet.face_df[sheet.coords]
        face_coords["z"] = 0
        face_coords = sheet.face_df[sheet.coords].values
        normal = np.column_stack((np.zeros(sheet.Nf), np.zeros(sheet.Nf), np.ones(sheet.Nf)))

        tangent = np.cross(face_coords, normal)
        tangent = pd.DataFrame(tangent)

        tangent.columns = ["t" + u for u in sheet.coords]

        length = pd.DataFrame(tangent.eval("sqrt(tx**2 + ty**2 +tz**2)"), columns=['length'])
        tangent["length"] = length["length"]

        tangent = tangent[['tx', 'ty', 'tz']].div(length.length, axis=0)

        for u in sheet.coords:
            sheet.face_df["t" + u] = tangent["t" + u]

    @staticmethod
    def update_face_distance(sheet):
        sheet.face_df['distance_z_axis'] = sheet.face_df.eval(
            "sqrt(x** 2 + y** 2)"
        )

    @staticmethod
    def update_vert_distance(sheet):
        sheet.vert_df['distance_z_axis'] = sheet.vert_df.eval(
            "sqrt(x** 2 + y** 2)"
        )

    @staticmethod
    def update_vert_deviation(sheet):
        if "dev_length" not in sheet.vert_df.columns:
            sheet.vert_df["dev_length"] = np.nan
        if "dx" not in sheet.vert_df.columns:
            sheet.vert_df["dx"] = np.nan
        if "dy" not in sheet.vert_df.columns:
            sheet.vert_df["dy"] = np.nan
        if "dz" not in sheet.vert_df.columns:
            sheet.vert_df["dz"] = np.nan

        edge_np = sheet.edge_df.to_numpy()
        edge_dict = dict(zip(sheet.edge_df.columns,
                             list(range(0, len(sheet.edge_df.columns)))))
        vert_np = sheet.vert_df.to_numpy()
        vert_dict = dict(zip(sheet.vert_df.columns,
                             list(range(0, len(sheet.vert_df.columns)))))

        grad1 = np.nan
        grad2 = np.nan
        gradt = np.nan
        lenth = []

        for i in sheet.vert_df.index:
            mask = (i == edge_np[:, edge_dict["srce"]])
            neighbor_verts = edge_np[mask, edge_dict["trgt"]].tolist()
            neighbor_verts = vert_np[neighbor_verts][:, [vert_dict["x"], vert_dict["y"], vert_dict["z"]]]
            vert_coords = vert_np[i, [vert_dict["x"], vert_dict["y"], vert_dict["z"]]]

            if len(neighbor_verts) >= 3:
                center = (neighbor_verts[0] + neighbor_verts[1] + neighbor_verts[2]) / 3

                grad = np.array(vert_coords) - np.array(center)

                length = np.linalg.norm(grad)

                grad = grad / length

            else:
                grad = np.array([0, 0, 0])

            if i == 0:
                grad1 = grad

            if i == 1:
                grad2 = grad
                gradt = np.vstack((grad1, grad2))

            if i >= 2:
                gradt = np.vstack((gradt, grad))

            lenth.append(length)

        gradt = pd.DataFrame(gradt, columns=['dx', 'dy', 'dz'])
        sheet.vert_df[['dx', 'dy', 'dz']] = gradt
        sheet.vert_df['dev_length'] = lenth

    @staticmethod
    def update_vert_deviation2(sheet):
        if "dev_length" not in sheet.vert_df.columns:
            sheet.vert_df["dev_length"] = np.nan
        if "dx" not in sheet.vert_df.columns:
            sheet.vert_df["dx"] = np.nan
        if "dy" not in sheet.vert_df.columns:
            sheet.vert_df["dy"] = np.nan
        if "dz" not in sheet.vert_df.columns:
            sheet.vert_df["dz"] = np.nan

        for i in sheet.vert_df.index:
            vert = sheet.vert_df.loc[i, ["x", "y", "z"]].to_numpy()
            neighbors = sheet.edge_df.loc[sheet.edge_df["srce"] == 6, ["tx", "ty", "tz"]].to_numpy()
            center = neighbors.sum(axis=0) / len(neighbors)
            grad = vert - center
            length = np.linalg.norm(grad)
            sheet.vert_df[['dx', 'dy', 'dz']] = grad
            sheet.vert_df['dev_length'] = length

    @staticmethod
    def update_lumen_vol(sheet):
        lumen_pos_faces = sheet.edge_df[["f" + c for c in sheet.coords]].to_numpy()
        lumen_sub_vol = (
                np.sum((lumen_pos_faces) * sheet.edge_df[sheet.ncoords].to_numpy(), axis=1)
                / 6
        )
        lumen_volume_gross = sum(lumen_sub_vol)

        top_verts = sheet.vert_df.loc[(sheet.vert_df["boundary"] == 1) & (sheet.vert_df["z"] > 0)]
        top_radius = top_verts["distance_z_axis"].values.mean()
        top_height = top_verts["z"].values.mean()
        top_volume = (1 / 3) * math.pi * top_radius ** 2 * top_height

        bot_verts = sheet.vert_df.loc[(sheet.vert_df["boundary"] == 1) & (sheet.vert_df["z"] < 0)]
        bot_radius = bot_verts["distance_z_axis"].values.mean()
        bot_height = -bot_verts["z"].values.mean()
        bot_volume = (1 / 3) * math.pi * bot_radius ** 2 * bot_height

        sheet.settings["lumen_vol"] = top_volume + bot_volume + lumen_volume_gross

    @staticmethod
    def update_vol_cell(sheet):
        sheet.settings["vol_cell"] = sheet.settings["lumen_vol"]/len(sheet.face_df)

    @staticmethod
    def update_boundary_radius(sheet):
        sheet.vert_df[["cx", "cy", "cz"]] = np.nan
        sheet.vert_df.loc[(sheet.vert_df["boundary"] == 1) & (sheet.vert_df["z"] <= 0), ["cx", "cy", "cz"]] = (
                    sheet.vert_df.loc[(sheet.vert_df["boundary"] == 1) & (sheet.vert_df["z"] <= 0), ["x", "y", "z"]] -
                    sheet.settings["bot_center"]).to_numpy()
        sheet.vert_df.loc[(sheet.vert_df["boundary"] == 1) & (sheet.vert_df["z"] >= 0), ["cx", "cy", "cz"]] = (
                    sheet.vert_df.loc[(sheet.vert_df["boundary"] == 1) & (sheet.vert_df["z"] >= 0), ["x", "y", "z"]] -
                    sheet.settings["top_center"]).to_numpy()
        sheet.vert_df["bound_rad"] = sheet.vert_df.eval("(cx**2 + cy**2 + cz**2) ** 0.5")
        sheet.vert_df[["cx", "cy", "cz"]] = sheet.vert_df[["cx", "cy", "cz"]].div(sheet.vert_df["bound_rad"], axis=0)
        sheet.vert_df.fillna(0, inplace=True)

    @classmethod
    def update_all(cls, sheet):
        super().update_all(sheet)
        cls.update_boundary_index(sheet)
        cls.update_tangents(sheet)
        # cls.update_face_tangents(sheet)
        # cls.update_face_distance(sheet)
        cls.update_vert_distance(sheet)
        cls.update_vert_deviation(sheet)
        cls.update_lumen_vol(sheet)
        # cls.update_vol_cell(sheet)
        # cls.update_boundary_radius(sheet)


class CylinderGeometry(CylinderGeometryInit):

    @classmethod
    def update_all(cls, sheet):
        super().update_all(sheet)
        # cls.update_preflumen_volume(sheet)

    @staticmethod
    def update_preflumen_volume(sheet):
        sheet.settings["lumen_prefered_vol"] = sheet.settings["vol_cell"] * len(sheet.face_df)