import os
from skypy.pipeline import Pipeline
import slsim
import tempfile
from astropy.cosmology import default_cosmology


class HalosSkyPyPipeline:
    """Class for halos configuration."""

    def __init__(
        self,
        skypy_config=None,
        sky_area=None,
        m_min=None,
        m_max=None,
        z_max=None,
        cosmo=None,
        sigma_8=0.81,
        n_s=0.96,
        omega_m=None,
    ):
        """Initialize the class with the given parameters.

        :param skypy_config: Path to SkyPy configuration yaml file. If None, the default
                SkyPy configuration file is used.
        :type skypy_config: str or None, optional
        :param sky_area: Sky area over which Halos are sampled. Must be in units of solid angle.
        :type sky_area: `~astropy.units.Quantity`, optional, defaults to 0.0001 deg2
        :param m_min: Minimum halo mass, defaults to 1.0E+12
        :type m_min: float, optional
        :param m_max: Maximum halo mass, defaults to 1.0E+16
        :type m_max: float, optional
        :param z_max: Maximum redshift value in z_range, defaults to 5.00
        :type z_max: float, optional
        :type sigma_8: float, optional, defaults to 0.81
        :type n_s: float, optional, defaults to 0.96
        :type omega_m: float, optional, defaults to 0.30966
        :param sigma_8: matter density fluctuations on a (8 h-1 Mpc), defaults to 0.81
                if not             specified.
        :param n_s: Spectral index, defaults to 0.96 if not specified.
        :param omega_m: Omega_m in Cosnmology, defaults to none which will lead to the
                same             in Cosmology setting.
        """

        path = os.path.dirname(slsim.__file__)
        module_path, _ = os.path.split(path)
        if skypy_config is None:
            skypy_config = os.path.join(module_path, "data/SkyPy/halo.yml")

        if (
            sky_area is None
            and m_min is None
            and m_max is None
            and z_max is None
            and cosmo is None
            and sigma_8 == 0.81
            and n_s == 0.96
            and omega_m is None
        ):
            self._pipeline = Pipeline.read(skypy_config)
            self._pipeline.execute()
        else:
            with open(skypy_config, "r") as file:
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

            old_sigma8 = "sigma8: 0.81"
            new_sigma8 = f"sigma8: {sigma_8}"
            content = content.replace(old_sigma8, new_sigma8)

            old_ns = "ns: 0.96"
            new_ns = f"ns: {n_s}"
            content = content.replace(old_ns, new_ns)

            if omega_m is not None:
                old_omega_m = "omega_m: 0.30996"
                new_omega_m = f"omega_m: {omega_m}"
                content = content.replace(old_omega_m, new_omega_m)

            if cosmo is not None:
                if cosmo is default_cosmology.get():
                    pass
                else:
                    cosmology_dict = cosmo.to_format("mapping")

                    cosmology_class = str(cosmology_dict.pop("cosmology", None))
                    cosmology_class_str = cosmology_class.replace(
                        "<class '", ""
                    ).replace("'>", "")

                    cosmology_dict.pop("cosmology", None)

                    if "meta" in cosmology_dict and cosmology_dict["meta"] not in [
                        "mapping",
                        None,
                    ]:
                        cosmology_dict.pop("meta", None)
                    # Reason: From Astropy:'meta:mapping or None (optional, keyword-only)'
                    # However, the dict will read out as meta: OrderedDict()
                    # which may raised error.

                    cosmology_dict = {
                        k: v for k, v in cosmology_dict.items() if v is not None
                    }

                    cosmology_params_list = []
                    for key, value in cosmology_dict.items():
                        if hasattr(value, "value"):
                            value = value.value
                        if key == "m_nu":
                            # Reason: From Astropy: m_nu will be out like [0.06, 0.06, 0.06]
                            value = value[0]
                        cosmology_params_list.append(f"    {key}: {value}")

                    cosmology_params_str = "\n".join(cosmology_params_list)

                    old_cosmo = "cosmology: !astropy.cosmology.default_cosmology.get []"
                    new_cosmo = (
                        f"cosmology: !{cosmology_class_str}\n{cosmology_params_str}"
                    )
                    content = content.replace(old_cosmo, new_cosmo)

            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".yml"
            ) as tmp_file:
                tmp_file.write(content)

            self._pipeline = Pipeline.read(tmp_file.name)
            self._pipeline.execute()
            # TODO:Add cosmo as an input
            # Remove the temporary file after the pipeline has been executed
            os.remove(tmp_file.name)

    @property
    def halos(self):
        """SkyPy pipeline for Halos.

        :returns: List of halos.
        :rtype: list of dict
        """

        return self._pipeline["halos"]

    @property
    def mass_sheet_correction(self):
        """SkyPy pipeline for mass sheet correction.

        :returns: List of sheet of mass for correction.
        :rtype: list of dict
        """

        return self._pipeline["mass_sheet_correction"]
