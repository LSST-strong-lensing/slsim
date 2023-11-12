import pandas as pd
import numpy as np
from astropy.table import Table


def sq_deg_to_sq_rad(sq_deg):
    # Function to convert square degrees to square radians

    return sq_deg * (np.pi / 180) ** 2


def rad_to_arcsec(rad):
    # Function to convert radians to arcseconds

    return rad * (180 / np.pi) * 3600


def radius_from_area(area):
    # Function to calculate the radius from the area

    return np.sqrt(area / np.pi)


class ReadMS(object):
    def __init__(
            self,
            file_path=None,
            selecting_area=0.00082,
            z_source=5,
            cosmo=None,
            sample_size=100,
    ):
        if file_path is None:
            file_path = 'C:/Users/TXZ27/OneDrive/Documents/GitHub/slsim/notebooks/GGL_los_8_0_3_3_3_N_4096_ang_4_SA_galaxies_on_plane_27_to_63.images.txt'

        self.df = pd.read_csv(file_path, sep='\t')
        self.selecting_area = selecting_area
        if cosmo is None:
            from astropy.cosmology import default_cosmology
            cosmo = default_cosmology.get()
        self.cosmo = cosmo
        self.selection_radius = self.selection_radius_rad()  # Define selection_radius before using it
        (self.adjusted_max_pos_0,
         self.adjusted_min_pos_0,
         self.adjusted_max_pos_1,
         self.adjusted_min_pos_1) = self.seletion_range_of_data()
        self.sample_size = sample_size

    def sky_range_of_data(self):
        max_pos_0 = self.df['pos_0[rad]'].max()
        min_pos_0 = self.df['pos_0[rad]'].min()
        max_pos_1 = self.df['pos_1[rad]'].max()
        min_pos_1 = self.df['pos_1[rad]'].min()
        return max_pos_0, min_pos_0, max_pos_1, min_pos_1

    def seletion_range_of_data(self):
        # Get the sky range and radius
        max_pos_0, min_pos_0, max_pos_1, min_pos_1 = self.sky_range_of_data()
        (adjusted_max_pos_0,
         adjusted_min_pos_0,
         adjusted_max_pos_1,
         adjusted_min_pos_1) = (max_pos_0 - self.selection_radius,
                                min_pos_0 + self.selection_radius,
                                max_pos_1 - self.selection_radius,
                                min_pos_1 + self.selection_radius)
        return adjusted_max_pos_0, adjusted_min_pos_0, adjusted_max_pos_1, adjusted_min_pos_1

    def selection_radius_rad(self):
        area_in_sq_rad = sq_deg_to_sq_rad(self.selecting_area)

        # Calculate the radius of the circle
        radius = radius_from_area(area_in_sq_rad)
        # print(f"Radius: {radius} rad")
        return radius

    def select_center_point_at_random(self):
        center_pos_0 = np.random.uniform(self.adjusted_min_pos_0, self.adjusted_max_pos_0)
        center_pos_1 = np.random.uniform(self.adjusted_min_pos_1, self.adjusted_max_pos_1)

        return center_pos_0, center_pos_1

    def select_object(self, center_pos_0=None, center_pos_1=None):
        if (center_pos_0 is None) and (center_pos_0 is None):
            center_pos_0, center_pos_1 = self.select_center_point_at_random()

        selected_objects = self.df[self.df.apply(
            lambda row: np.sqrt((row['pos_0[rad]'] - center_pos_0) ** 2 + (
                        row['pos_1[rad]'] - center_pos_1) ** 2) <= self.selection_radius,
            axis=1)].copy()  # Create a copy here

        return selected_objects, center_pos_0, center_pos_1

    def selected_to_astro_table(self, selected_objects, center_pos_0, center_pos_1):
        selected_objects['pos_0[rad]'] = rad_to_arcsec(selected_objects['pos_0[rad]'] - center_pos_0)
        selected_objects['pos_1[rad]'] = rad_to_arcsec(selected_objects['pos_1[rad]'] - center_pos_1)

        selected_objects['M_Halo[M_sol/h]'] *= self.cosmo.h

        filtered_data = selected_objects[['z_spec', 'M_Halo[M_sol/h]', 'pos_0[rad]', 'pos_1[rad]']]

        # Convert to Astropy Table
        astropy_table = Table.from_pandas(filtered_data)

        astropy_table.rename_column('z_spec', 'z')
        astropy_table.rename_column('M_Halo[M_sol/h]', 'mass')  # M_sun
        astropy_table.rename_column('pos_0[rad]', 'px')
        astropy_table.rename_column('pos_1[rad]', 'py')  # arcsec
        center_point = (center_pos_0, center_pos_1)
        astropy_table.meta['Center_Point'] = center_point

        return astropy_table

    def get_tables(self):
        tables = []
        for i in range(self.sample_size):
            center_pos_0, center_pos_1 = self.select_center_point_at_random()
            selected_objects, _, _ = self.select_object(center_pos_0, center_pos_1)
            table = self.selected_to_astro_table(selected_objects, center_pos_0, center_pos_1)
            tables.append(table)
        return tables
