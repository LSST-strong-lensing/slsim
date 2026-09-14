from typing import Optional
from astropy.units import Quantity
from astropy.cosmology import Cosmology
import random
from slsim.Lenses.lens import Lens
from slsim.Sources.source import Source
from slsim.Deflectors.deflector import Deflector
from slsim.LOS.los_individual import LOSIndividual


class LensPopCatalog(object):
    """Class to deal with pre-established set of lenses.

    The catalog should contain parameters for the deflector ending with
    <_deflector>, parameters for the source ending with <_source> and
    for the line of sight ending with <_los>.
    Source parameters should match slsim.Sources.source.Source() class initialization with
    Source(extended_source_type, point_source_type, **<..._source>).
    Deflector parameters should match kwargs_mass, z, center_x, center_y and kwargs_light of Deflector().
    Parameters in kwargs_light of the deflector should be stated as _light_deflector.
    """

    def __init__(
        self,
        lens_catalog,
        cosmo: Optional[Cosmology] = None,
        sky_area: Optional[float or Quantity] = None,
        catalog_type=None,
        use_jax=True,
        deflector_mass_type="EPL",
        deflector_light_type="single_sersic",
        extended_source_type="single_sersic",
        point_source_type=None,
    ):
        """

        :param lens_catalog: catalog with all deflector and source parameters with <_source>, <_deflector> and <_los>
         added
        :param cosmo: astropy.cosmology instance
        :param sky_area: Sky area (solid angle) over which Lens population is sampled.
        :type sky_area: `~astropy.units.Quantity`
        :param catalog_type: catalog type for special conversions of catalog entries ot SLSim conventions
        :type catalog_type: str
        :param use_jax: if True, will use JAX version of lenstronomy to do lensing calculations for models that are
         supported in JAXtronomy
        :type use_jax: bool
        :param deflector_mass_type: deflector mass type consistent with Deflector() class and catalog input
        :param deflector_light_type: deflector light model type consistent with Source() class and catalog input
        """
        self._lens_catalog = lens_catalog
        self._num_lenses = len(lens_catalog)
        self._catalog_type = catalog_type
        self._use_jax = use_jax
        self._cosmo = cosmo
        self._sky_area = sky_area
        self._deflector_mass_type = deflector_mass_type
        self._deflector_light_type = deflector_light_type
        self._point_source_type = point_source_type
        self._extended_source_type = extended_source_type

    def lens_from_table(self, index=None):
        """

        :param index: index of catalog of the class initialization
        :return: Lens() class
        """
        if index is None:
            index = random.randint(0, self._num_lenses - 1)
        lens_object = self._lens_catalog[index]

        deflector_dict, source_dict, los_dict = _catalog_deflector_source_split(
            lens_object
        )
        z, center_x, center_y, kwargs_mass, kwargs_light = _deflector_dict_split(
            deflector_dict
        )
        kwargs_mass["mass_type"] = self._deflector_mass_type
        kwargs_light["extended_source_type"] = self._deflector_light_type

        deflector = Deflector(
            z,
            center_x=center_x,
            center_y=center_y,
            kwargs_mass=kwargs_mass,
            kwargs_light=kwargs_light,
        )

        source = Source(
            extended_source_type=self._extended_source_type,
            point_source_type=self._point_source_type,
            **source_dict
        )
        los_class = LOSIndividual(**los_dict)
        lens_class = Lens(
            source_class=source,
            deflector_class=deflector,
            cosmo=self._cosmo,
            los_class=los_class,
        )
        return lens_class


def _deflector_dict_split(deflector_dict):
    """Split single dictionary into the components required in the Deflector()
    class.

    :param deflector_dict: single dictionary containing all the
        deflector quantities
    :type deflector_dict: dict
    :return: z, center_x, center_y, kwargs_mass, kwargs_light
    """

    z = deflector_dict.pop("z")
    center_x = deflector_dict.pop("center_x")
    center_y = deflector_dict.pop("center_y")
    colnames = list(deflector_dict.keys())
    kwargs_mass = {}
    kwargs_light = {}
    for item in colnames:
        if item.endswith("_light"):
            clean_item = item.removesuffix("_light")
            kwargs_light[clean_item] = deflector_dict[item]
        else:
            kwargs_mass[item] = deflector_dict[item]
    return z, center_x, center_y, kwargs_mass, kwargs_light


def _catalog_deflector_source_split(lens_object):
    """Split catalog with <_source> and <_deflector>

    :param lens_object: single column of catalog
    :return: deflector_dict, source_dict, los_dict
    """
    if isinstance(lens_object, dict):
        colnames = list(lens_object.keys())
    else:
        colnames = lens_object.colnames

    deflector_dict = {}
    source_dict = {}
    los_dict = {}
    for item in colnames:
        if item.endswith("_deflector"):
            clean_item = item.removesuffix("_deflector")
            deflector_dict[clean_item] = lens_object[item]
        elif item.endswith("_source"):
            clean_item = item.removesuffix("_source")
            source_dict[clean_item] = lens_object[item]
        elif item.endswith("_los"):
            clean_item = item.removesuffix("_los")
            los_dict[clean_item] = lens_object[item]
        else:
            raise ValueError(
                "key %s is not supported. Key needs to end in _deflector, _source or _los."
            )

    return deflector_dict, source_dict, los_dict
