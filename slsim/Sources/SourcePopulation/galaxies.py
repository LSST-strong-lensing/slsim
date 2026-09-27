import numpy as np
from slsim.Util import param_util
from slsim.Sources.SourcePopulation.source_pop_base import SourcePopBase
from astropy.table import vstack
from slsim.Util.param_util import (
    average_angular_size,
    axis_ratio,
    eccentricity,
    downsample_galaxies,
    galaxy_size_redshift_evolution,
    galaxy_size,
)
from astropy import units as u
from astropy.table import Table
from slsim.Sources.source import Source
import os


class Galaxies(SourcePopBase):
    """Class describing elliptical galaxies."""

    def __init__(
        self,
        galaxy_list,
        cosmo,
        sky_area,
        kwargs_cut=None,
        catalog_type="skypy",
        size_model=None,
        extended_source_type="single_sersic",
        downsample_to_dc2=False,
        extended_source_kwargs=None,
    ):
        """

        :param galaxy_list: An astropy table with galaxy parameters.
         The minimal requirement for the catalog is "z" for redshift and the magnitudes in case cuts are
         being made on it.
        :type galaxy_list: astropy Table object or list of objects.
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param cosmo: astropy.cosmology instance
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
         solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param catalog_type: type of the catalog. If someone wants to use scotch
         catalog, they need to specify it. Default will be "skypy"
        :type catalog_type: str. eg: "scotch" or None
        :param downsample_to_dc2: Boolean. If True, downsamples the given galaxy
         population at redshift greater than 1.5 to DC2 galaxy population.
        :param size_model: If "Bernardi", computes galaxy size using g-band
         magnitude otherwise rescales skypy source size to Shibuya et al. (2015):
         https://iopscience.iop.org/article/10.1088/0067-0049/219/2/15/pdf
        :param extended_source_type: Keyword to specify type of the extended source.
         Supported extended source types are "single_sersic", "double_sersic", "interpolated".
        :type extended_source_type: str or None
        :param extended_source_kwargs: dictionary of keyword arguments for ExtendedSource.
         Please see documentation of ExtendedSource() class as well as specific extended source classes.
        """

        self._size_model = size_model
        self._catalog_type = catalog_type
        if isinstance(galaxy_list, Table):
            self._astropy_table = True
        else:
            self._astropy_table = False
        if self._astropy_table:
            if extended_source_type == "double_sersic":
                column_names = galaxy_list.colnames
                tuples = [
                    ("n0", "n_sersic_0"),
                    ("n1", "n_sersic_1"),
                    ("e0", "ellipticity0"),
                    ("e1", "ellipticity1"),
                    ("angular_size0", "angular_size_0"),
                    ("angular_size1", "angular_size_1"),
                ]

                for old, new in tuples:
                    if old in column_names:
                        galaxy_list.rename_column(old, new)
        if extended_source_kwargs is None:
            extended_source_kwargs = {}
        self._extended_source_kwargs = extended_source_kwargs
        if self._catalog_type == "scotch":
            galaxy_list = _convert_scotch_catalog(galaxy_list)
        if downsample_to_dc2 is True:
            samp1, samp2, samp3, samp4, samp5, samp6 = down_sample_to_dc2(
                galaxy_pop=galaxy_list, sky_area=sky_area
            )
            samp_low = galaxy_list[galaxy_list["z"] <= 2]
            galaxy_list = vstack([samp_low, samp1, samp2, samp3, samp4, samp5, samp6])
        self._num_galaxies_full = len(galaxy_list)

        super().__init__(
            cosmo=cosmo,
            object_list=galaxy_list,
            kwargs_cut=kwargs_cut,
            sky_area=sky_area,
            extended_source_type=extended_source_type,
            point_source_type=None,
        )

    def draw_source_dict(
        self, z_max=None, z_min=None, galaxy_index=None, include_all_keywords=False
    ):
        """Choose source at random. For speed, it is recommended not to use
        additional redshift selection here. Instead use the kwargs_cut of the
        class initialization.

        :param z_max: maximum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :param z_min: minimum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :param galaxy_index: index of galaxy to pic (if provided)
        :param include_all_keywords: if True, includes all keywords and
            not just the ones required
        :type include_all_keywords: bool
        :return: dictionary of source in the form of compatible with
            Source() class
        """
        galaxy = self.draw_object(z_max=z_max, z_min=z_min, galaxy_index=galaxy_index)
        if galaxy is None:
            return None

        kwargs_source = convert_catalog_to_source(
            galaxy=galaxy,
            extended_source_type=self._extended_source_type,
            catalog_type=self._catalog_type,
            size_model=self._size_model,
            cosmo=self._cosmo,
            include_all_keywords=include_all_keywords,
        )
        return kwargs_source

    def draw_source(self, z_max=None, z_min=None, galaxy_index=None):
        """Choose source at random.

        :param z_max: maximum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :param z_min: minimum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :param galaxy_index: index of galaxy to pic (if provided)
        :return: instance of Source class
        """
        kwargs_source = self.draw_source_dict(
            z_max=z_max, z_min=z_min, galaxy_index=galaxy_index
        )
        if kwargs_source is None:
            return None

        source_class = Source(
            cosmo=self._cosmo, **kwargs_source, **self._extended_source_kwargs
        )
        return source_class

    def draw_galaxies(self, area, z_max=None):
        """Draw galaxies within a specified area and redshift limit.

        :param area: Area in which to draw galaxies
        :type area: ~astropy.units.Quantity
        :param z_max: Maximum redshift for the galaxies. If None, no
            redshift cut is applied.
        :return: List of drawn galaxy instances.
        """

        total_sources = self.source_number_selected

        pop_sky_area_arcsec2 = self.sky_area.to_value("arcsec2")
        area_arcsec2 = area.to_value("arcsec2")
        mean_sources = (total_sources / pop_sky_area_arcsec2) * area_arcsec2

        # draw from Poisson Distribution
        number_of_sources = np.random.poisson(lam=mean_sources)

        galaxies_list = []
        for _ in range(number_of_sources):
            galaxy = self.draw_source(z_max=z_max)
            if galaxy is not None:
                galaxy.update_center(area=area_arcsec2)
                galaxies_list.append(galaxy)

        return galaxies_list


def galaxy_projected_eccentricity(ellipticity, rotation_angle=None):
    """Projected eccentricity of elliptical galaxies as a function of other
    deflector parameters.

    :param ellipticity: eccentricity amplitude
    :type ellipticity: float [0,1)
    :param rotation_angle: rotation angle of the major axis of
        elliptical galaxy in radian. The reference of this rotation
        angle is +Ra axis i.e towards the East direction and it goes
        from East to North. If it is not provided, it will be drawn
        randomly.
    :return: e1, e2 eccentricity components
    """
    if rotation_angle is None:
        phi = np.random.uniform(0, np.pi)
    else:
        phi = rotation_angle
    e = param_util.epsilon2e(ellipticity)
    e1 = e * np.cos(2 * phi)
    e2 = e * np.sin(2 * phi)
    return e1, e2


def _convert_scotch_catalog(galaxy_catalog):
    """

    :param galaxy_catalog: scotch catalog
    :type galaxy_catalog: ~astropy.Table
    :return:
    """

    # scotch catalog has _host naming that we do not need
    column_names = galaxy_catalog.colnames
    for col_name in column_names:
        if "_host" in col_name:
            # Remove '_host' from the column name
            new_col_name = col_name.replace("_host", "")
            # Rename the column
            galaxy_catalog.rename_column(col_name, new_col_name)
    return galaxy_catalog


def _galaxy_size(galaxy, size_model, catalog_type, cosmo):
    """Converts or adjusts galaxy size.

    :param galaxy: galaxy parameter keyword arguments
    :type galaxy: dict or Table entry
    :param size_model: galaxy size model
    :type size_model: str
    :param catalog_type: type of catalog, only relevant when conventions
        of catalog are not in default SLSim conventions
    :param cosmo: astrop cosmology
    :return: angular_size [arcsec], physical_size [kpc]
    """

    z = galaxy["z"]
    if isinstance(galaxy, dict):
        col_names = list(galaxy.keys())
    else:
        col_names = galaxy.colnames
    if size_model == "Bernardi" and catalog_type == "skypy":
        # TODO: enable this scaling also for other catalogs. Currently not done to be backwards compatible
        # compute angular size from g-band magnitude.
        if "mag_g" not in col_names:
            raise ValueError(
                "mag_g needs to be in the arguments to use the Bernardi et al. g-band magnitude to "
                "size conversion."
            )
        source_size = galaxy_size(float(galaxy["mag_g"]), z, cosmo)
        physical_size = source_size[0] * u.kpc
        angular_size = source_size[1] * u.arcsec
        return angular_size, physical_size

    if "physical_size" in col_names:
        physical_size = galaxy["physical_size"]
        if isinstance(physical_size, u.Quantity):
            physical_size = physical_size.to(u.kpc)
        else:
            physical_size *= u.kpc
    else:
        physical_size = None
    if "angular_size" in col_names:
        angular_size = galaxy["angular_size"]
        if isinstance(angular_size, u.Quantity):
            angular_size = angular_size.to(u.arcsec)
        else:
            angular_size *= u.arcsec
    else:
        angular_size = None

    if angular_size is None and physical_size is not None:
        angular_size_rad = physical_size / cosmo.angular_diameter_distance(z).to(u.kpc)
        angular_size = (angular_size_rad * u.rad).to(u.arcsec)
    elif physical_size is None and angular_size is not None:
        physical_size = angular_size.to(u.rad) * cosmo.angular_diameter_distance(z).to(
            u.kpc
        )
    elif angular_size is None and physical_size is None:
        raise ValueError("Either angular_size or physical_size need to be provided.")

    if catalog_type == "skypy":
        # skypy model comes with physical sizes in kpc

        # rescales skypy source size to Shibuya et al. (2015).
        # compute the rescaled physical size. The resulted value is divided by 2.5 to
        #  match the best-fit model given in https://iopscience.iop.org/article/10.1088/0067-0049/219/2/15/pdf
        angular_size *= galaxy_size_redshift_evolution(z) / 2.5
        physical_size *= galaxy_size_redshift_evolution(z) / 2.5

    return angular_size.value, physical_size.value


def convert_catalog_to_source(
    galaxy,
    extended_source_type,
    catalog_type,
    size_model=None,
    cosmo=None,
    include_all_keywords=False,
):
    """Converts input table entries into the quantities used in slsim Source()
    class.

    :param galaxy: entry in galaxy_list
    :type galaxy: dictionary or table entry
    :param extended_source_type: type of extended source compatible with
        Source() class
    :type extended_source_type: None or str
    :param size_model: galaxy size model
    :type size_model: str
    :param catalog_type: type of catalog, only relevant when conventions
        of catalog are not in default SLSim conventions
    :param cosmo: astrop cosmology
    :param include_all_keywords: if True, includes all keywords and not
        just the ones required
    :type include_all_keywords: bool
    :return: dictionary compatible with Source() class
    """
    if isinstance(galaxy, dict):
        colnames = list(galaxy.keys())
    else:
        colnames = galaxy.colnames
    kwargs_source = {
        "z": float(galaxy["z"]),
        "extended_source_type": extended_source_type,
    }
    for key in colnames:
        if key.startswith("ps_mag_") or key.startswith("mag_"):
            kwargs_source[key] = galaxy[key]

    if "a_rot" in colnames:
        phi_rot = galaxy["a_rot"]
        if catalog_type == "scotch":
            phi_rot = np.deg2rad(phi_rot)
    elif "p_g" in colnames:
        # SL_hammock pipeline defines the position angle in degrees
        phi_rot = np.deg2rad(galaxy["p_g"])
    else:
        phi_rot = np.random.uniform(0, np.pi)

    if extended_source_type in ["single_sersic", "hernquist", "catalog_source"]:
        # get angular_size
        angular_size, physical_size = _galaxy_size(
            galaxy, size_model=size_model, catalog_type=catalog_type, cosmo=cosmo
        )
        kwargs_source["angular_size"] = angular_size
        kwargs_source["physical_size"] = physical_size

    if extended_source_type in ["single_sersic", "hernquist", "catalog_source"]:
        if "e1" in colnames and "e2" in colnames:
            e1, e2 = galaxy["e1"], galaxy["e2"]
        elif "e1_light" in colnames and "e2_light" in colnames:
            e1, e2 = galaxy["e1_light"], galaxy["e2_light"]
        else:
            if "ellipticity" in colnames:
                ellipticity = galaxy["ellipticity"]
            elif "e" in colnames:
                ellipticity = galaxy["e"]
            else:
                raise ValueError(
                    "Ellipticity either as ('e1', 'e2'), 'e' or as 'ellipticity' is required."
                )
            e1, e2 = galaxy_projected_eccentricity(
                float(ellipticity), rotation_angle=phi_rot
            )

        kwargs_source["e1"], kwargs_source["e2"] = e1, e2
    if extended_source_type in ["single_sersic", "catalog_source"]:
        if "n_sersic" not in colnames:
            if "galaxy_type" in colnames and galaxy["galaxy_type"] == "red":
                kwargs_source["n_sersic"] = 4
            else:
                kwargs_source["n_sersic"] = 1
            # TODO make a better estimate with scatter and distinguish between blue and red galaxies
        else:
            kwargs_source["n_sersic"] = float(galaxy["n_sersic"])

    if extended_source_type == "double_sersic":
        kwargs_source.update(
            _double_sersic_source_kwargs(
                galaxy=galaxy,
                colnames=colnames,
                phi_rot=phi_rot,
                size_model=size_model,
                catalog_type=catalog_type,
                cosmo=cosmo,
            )
        )
    if "vel_disp" in colnames:
        kwargs_source["vel_disp"] = float(galaxy["vel_disp"])
    if "stellar_mass" in colnames:
        kwargs_source["stellar_mass"] = float(galaxy["stellar_mass"])
    if include_all_keywords is True:
        for key in colnames:
            if key not in kwargs_source:
                kwargs_source[key] = galaxy[key]
    return kwargs_source


def _double_sersic_source_kwargs(
    galaxy,
    colnames,
    phi_rot,
    size_model=None,
    catalog_type=None,
    cosmo=None,
):
    """Build the DoubleSersic and colour-gradient Source dictionary.

    The input ``galaxy`` dictionary may provide component-specific values or
    single-component catalog values that are shared by both components:

    .. code-block:: python

        galaxy = {
            # Required reference-band flux fractions
            "w0": 0.4,
            "w1": 0.6,
            # Shape: component-specific, shared Cartesian, or shared scalar
            "e1_0": 0.1,
            "e2_0": 0.0,
            "e1_1": 0.1,
            "e2_1": 0.0,
            # Size and Sersic-index component values
            "angular_size_0": 0.3,
            "angular_size_1": 0.9,
            "n_sersic_0": 1.0,
            "n_sersic_1": 4.0,
            # Optional chromatic configuration
            "color_gradient": {
                "component_spectral_slopes": [0.5, -0.5],
                "reference_band": "i",
                "component_radius_factors": [0.5, 1.5],
                "component_sersic_indices": [1.0, 4.0],
                "min_weight": 1e-3,
            },
        }

    When component sizes are absent, ``angular_size`` is multiplied by
    ``color_gradient['component_radius_factors']``. When component Sersic
    indices are absent, ``n_sersic`` and
    ``color_gradient['component_sersic_indices']`` provide the defaults.

    :param galaxy: Catalog row or dictionary containing light-profile fields.
    :param colnames: Available keys/column names in ``galaxy``.
    :param phi_rot: Position angle in radians used for scalar ellipticities.
    :param size_model: Galaxy size model passed to :func:`_galaxy_size`.
    :param catalog_type: Optional catalog convention identifier.
    :param cosmo: Astropy cosmology used to convert physical/angular size.
    :return: Dictionary containing ``e1_0``, ``e2_0``, ``e1_1``, ``e2_1``,
        ``angular_size_0``, ``angular_size_1``, ``n_sersic_0``,
        ``n_sersic_1``, ``w0``, ``w1``, and optional ``color_gradient``.
    """
    kwargs_double_sersic = {}
    color_gradient = galaxy["color_gradient"] if "color_gradient" in colnames else {}
    if not isinstance(color_gradient, dict):
        raise ValueError(
            "galaxy['color_gradient'] must be a dictionary when constructing "
            f"a DoubleSersic source; received {color_gradient!r}."
        )

    for component in (0, 1):
        e1_key, e2_key = f"e1_{component}", f"e2_{component}"
        if e1_key in colnames and e2_key in colnames:
            e1, e2 = galaxy[e1_key], galaxy[e2_key]
        elif "e1" in colnames and "e2" in colnames:
            e1, e2 = galaxy["e1"], galaxy["e2"]
        elif "e1_light" in colnames and "e2_light" in colnames:
            e1, e2 = galaxy["e1_light"], galaxy["e2_light"]
        else:
            ellipticity_key = f"ellipticity{component}"
            a_key, b_key = f"a{component}", f"b{component}"
            if ellipticity_key in colnames:
                ellipticity = galaxy[ellipticity_key]
            elif a_key in colnames and b_key in colnames:
                ellipticity = eccentricity(
                    q=axis_ratio(a=galaxy[a_key], b=galaxy[b_key])
                )
            elif "ellipticity" in colnames or "e" in colnames:
                ellipticity = (
                    galaxy["ellipticity"] if "ellipticity" in colnames else galaxy["e"]
                )
            else:
                raise ValueError(
                    f"Cannot determine ellipticity for DoubleSersic component "
                    f"{component}; available galaxy_list columns are {colnames}."
                )
            e1, e2 = galaxy_projected_eccentricity(
                float(ellipticity), rotation_angle=phi_rot
            )
        kwargs_double_sersic[e1_key] = e1
        kwargs_double_sersic[e2_key] = e2

    if "angular_size_0" in colnames and "angular_size_1" in colnames:
        kwargs_double_sersic["angular_size_0"] = galaxy["angular_size_0"]
        kwargs_double_sersic["angular_size_1"] = galaxy["angular_size_1"]
    elif all(key in colnames for key in ("a0", "b0", "a1", "b1")):
        kwargs_double_sersic["angular_size_0"] = average_angular_size(
            a=galaxy["a0"], b=galaxy["b0"]
        )
        kwargs_double_sersic["angular_size_1"] = average_angular_size(
            a=galaxy["a1"], b=galaxy["b1"]
        )
    elif "angular_size" in colnames or catalog_type is not None:
        angular_size, _ = _galaxy_size(
            galaxy,
            size_model=size_model,
            catalog_type=catalog_type,
            cosmo=cosmo,
        )
        radius_factors = color_gradient.get("component_radius_factors", (0.5, 1.5))
        if len(radius_factors) != 2:
            raise ValueError(
                "color_gradient['component_radius_factors'] must contain two values."
            )
        kwargs_double_sersic["angular_size_0"] = angular_size * float(radius_factors[0])
        kwargs_double_sersic["angular_size_1"] = angular_size * float(radius_factors[1])
    else:
        raise ValueError(
            "Cannot determine DoubleSersic component sizes. Provide one of: "
            "(1) both 'angular_size_0' and 'angular_size_1'; "
            "(2) all four axis fields 'a0', 'b0', 'a1', and 'b1'; "
            "(3) a shared 'angular_size'; or "
            "(4) a catalog_type from which the shared size can be inferred. "
            f"Available galaxy_list columns are {colnames}."
        )

    if "n_sersic_0" in colnames and "n_sersic_1" in colnames:
        indices = (galaxy["n_sersic_0"], galaxy["n_sersic_1"])
    else:
        if "n_sersic" in colnames:
            n_sersic = float(galaxy["n_sersic"])
        elif "galaxy_type" in colnames and galaxy["galaxy_type"] == "red":
            n_sersic = 4.0
        else:
            n_sersic = 1.0
        indices = color_gradient.get("component_sersic_indices", (n_sersic, n_sersic))
    if len(indices) != 2:
        raise ValueError(
            "color_gradient['component_sersic_indices'] must contain two values."
        )
    kwargs_double_sersic["n_sersic_0"] = float(indices[0])
    kwargs_double_sersic["n_sersic_1"] = float(indices[1])

    missing_weights = [key for key in ("w0", "w1") if key not in colnames]
    if missing_weights:
        raise ValueError(
            "DoubleSersic source dictionary is missing reference-band flux "
            f"weights {missing_weights}; provide both 'w0' and 'w1'."
        )
    kwargs_double_sersic["w0"] = galaxy["w0"]
    kwargs_double_sersic["w1"] = galaxy["w1"]
    if "color_gradient" in colnames:
        kwargs_double_sersic["color_gradient"] = color_gradient
    return kwargs_double_sersic


def down_sample_to_dc2(galaxy_pop, sky_area):
    """Downsamples given galaxy pop above redshift 1.5 to DC2 galaxy
    population.

    :param galaxy_pop: Astropy table of galaxy population.
    :param sky_area: Sky area over which galaxies are sampled. Must be in units of
     solid angle, and it should be astropy unit object.
    :return: Astropy tables of downsampled galaxy population in different bins.
     Redshift bins for returned populations are: (2-2.5), (2.5-3), (3-3.5),
     (3.5-4), (4-4.5), (4.5-5)
    """
    path = os.path.dirname(__file__)
    new_path = path[: path.rfind("slsim/")]
    module_path = os.path.dirname(new_path)
    # path1 = os.path.join(
    #    module_path, "data/DC2_data/dc2_galaxy_count_1.5_2.npy"
    # )
    path2 = os.path.join(module_path, "data/DC2_data/dc2_galaxy_count_2_2.5.npy")
    path3 = os.path.join(module_path, "data/DC2_data/dc2_galaxy_count_2.5_3.npy")
    # DC2 galaxy counts in 3 different redshift bins: (2-2.5), (2.5-3). Beyond 3
    # , we use the same count as 3rd bin because DC2 only reach up to redshift 3.
    # dN1 = np.load(path1)
    dN2 = int(sky_area.value) * np.load(path2)
    dN3 = int(sky_area.value) * np.load(path3)

    # M_min1=21.531229
    # M_max1=29.999994
    # dM1=0.2920263882341056
    M_min2 = 22.084414
    M_max2 = 29.999998
    dM2 = 0.2729511918692753
    M_min3 = 22.654068
    M_max3 = 29.999996
    dM3 = 0.25330786869443694
    # slsim_sample_15_2=downsample_galaxies(galaxy_pop, dN1, dM1, M_min1,
    #                                        M_max1, 1.5, 2)
    slsim_sample_2_25 = downsample_galaxies(
        galaxy_pop, dN2, dM2, M_min2, M_max2, 2, 2.5
    )
    slsim_sample_25_3 = downsample_galaxies(
        galaxy_pop, dN3, dM3, M_min3, M_max3, 2.5, 3
    )
    slsim_sample_3_35 = downsample_galaxies(
        galaxy_pop, dN3, dM3, M_min3, M_max3, 3, 3.5
    )
    slsim_sample_35_4 = downsample_galaxies(
        galaxy_pop, dN3, dM3, M_min3, M_max3, 3.5, 4
    )
    slsim_sample_4_45 = downsample_galaxies(
        galaxy_pop, dN3, dM3, M_min3, M_max3, 4, 4.5
    )
    slsim_sample_45_5 = downsample_galaxies(
        galaxy_pop, dN3, dM3, M_min3, M_max3, 4.5, 5
    )
    return (
        slsim_sample_2_25,
        slsim_sample_25_3,
        slsim_sample_3_35,
        slsim_sample_35_4,
        slsim_sample_4_45,
        slsim_sample_45_5,
    )
