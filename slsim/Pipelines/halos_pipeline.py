import os
from skypy.pipeline import Pipeline
import slsim
import tempfile


class HalosSkyPyPipeline:
    def __init__(self, skypy_config=None, sky_area=None, m_min=None, m_max=None, z_max=None):
        """
        Initialize the class with the given parameters.

        Parameters
        ----------
        skypy_config : str or None, optional
            Path to SkyPy configuration yaml file. If None, the default SkyPy configuration file is used.
        sky_area : `~astropy.units.Quantity`, optional
            Sky area over which Halos are sampled. Must be in units of solid angle.
        m_min : float, optional
            Minimum halo mass.
        m_max : float, optional
            Maximum halo mass.
        z_max : float, optional
            Maximum redshift value in z_range.

        """
        path = os.path.dirname(slsim.__file__)
        module_path, _ = os.path.split(path)
        if skypy_config is None:
            skypy_config = os.path.join(module_path, 'data/SkyPy/halo.yml')

        if sky_area is None and m_min is None and m_max is None and z_max is None:
            self._pipeline = Pipeline.read(skypy_config)
            self._pipeline.execute()
        else:
            with open(skypy_config, 'r') as file:
                content = file.read()

            if sky_area is not None:
                old_fsky = "fsky: 0.0001 deg2"
                new_fsky = f"fsky: {sky_area} deg2"
                content = content.replace(old_fsky, new_fsky)

            if m_min is not None:
                old_m_min = "m_min: 1.0E+12"
                new_m_min = f"m_min: {str(m_min)}"
                content = content.replace(old_m_min, new_m_min)

            if m_max is not None:
                old_m_max = "m_max: 1.0E+16"
                new_m_max = f"m_max: {str(m_max)}"
                content = content.replace(old_m_max, new_m_max)

            if z_max is not None:
                old_z_max = "z_max: 5.00"
                new_z_max = f"z_max: {z_max}"
                content = content.replace(old_z_max, new_z_max)

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yml') as tmp_file:
                tmp_file.write(content)

            self._pipeline = Pipeline.read(tmp_file.name)
            self._pipeline.execute()
            # TODO:Add cosmo as an input
            # Remove the temporary file after the pipeline has been executed
            os.remove(tmp_file.name)

    @property
    def halos(self):
        """
        SkyPy pipeline for Halos.

        Returns
        -------
        list of dict
            List of halos.

        """
        return self._pipeline['halos']

    @property
    def mass_sheet_correction(self):
        """
        SkyPy pipeline for mass sheet correction.

        Returns
        -------
        list of dict
            List of sheet of mass for correction.

        """
        return self._pipeline['mass_sheet_correction']
