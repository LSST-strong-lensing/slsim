from slsim.ParamDistributions.los_config import LOSConfig  # Importa la configuración de la línea de visión.
import os  # Importa el módulo os para interactuar con el sistema operativo.
import pickle  # Importa pickle para serializar y deserializar objetos de Python.

import numpy as np  # Importa NumPy, que se usa para operaciones numéricas.
from astropy.table import Table  # Importa Table de Astropy para manejar tablas de datos astronómicos.

from slsim.lens import Lens  # Importa la clase Lens que representa un sistema de lentes.
from slsim.lens import theta_e_when_source_infinity  # Importa una función que calcula el radio de Einstein.
from slsim.lensed_population_base import LensedPopulationBase  # Importa la clase base para poblaciones de lentes.
from slsim.Pipelines.skypy_pipeline import \
    SkyPyPipeline  # Importa el pipeline de SkyPy para simular objetos astronómicos.


class LensPop(LensedPopulationBase):
    """Clase para realizar muestras de una población de lentes gravitacionales."""

    def __init__(
            self,
            deflector_type="elliptical",  # Tipo de deflector por defecto: "elliptical".
            source_type="galaxies",  # Tipo de fuente por defecto: "galaxies".
            kwargs_deflector_cut=None,  # Parámetros para definir cortes en los deflectores.
            kwargs_source_cut=None,  # Parámetros para definir cortes en las fuentes.
            kwargs_quasars=None,  # Parámetros específicos para cuásares.
            kwargs_quasars_galaxies=None,  # Parámetros para cuásares más galaxias.
            variability_model=None,  # Modelo de variabilidad para la fuente.
            kwargs_variability=None,  # Parámetros adicionales para la variabilidad.
            kwargs_mass2light=None,  # Parámetros para la relación masa-luz.
            skypy_config=None,  # Ruta del archivo de configuración de SkyPy.
            slhammocks_config=None,  # Configuración específica para "halo-models".
            sky_area=None,  # Área del cielo en la que se simulará la población de lentes.
            source_sky_area=None,  # Área del cielo donde se muestrean las fuentes.
            deflector_sky_area=None,  # Área del cielo donde se muestrean los deflectores.
            filters=None,  # Filtros para la integración de SED.
            cosmo=None,  # Objeto de cosmología.
            source_light_profile="single_sersic",  # Perfil de luz de la fuente.
            catalog_type="skypy",  # Tipo de catálogo que se va a usar.
            catalog_path=None,  # Ruta al catálogo de fuentes.
            lightcurve_time=None,  # Tiempo de observación para la curva de luz.
            sn_type=None,  # Tipo de supernova.
            sn_absolute_mag_band=None,  # Banda utilizada para normalizar la magnitud absoluta.
            sn_absolute_zpsys=None,  # Sistema de magnitud cero.
            los_config=None,  # Configuración de la línea de visión.
            sn_modeldir=None,  # Directorio para los archivos del modelo de supernova.
    ):
        """
        Inicializa la clase LensPop con los parámetros proporcionados.
        """
        # Llamada al constructor de la clase base o padre (LensedPopulationBase) para inicializar los parámetros
        # comunes como sky_area, cosmo, lightcurve_time, sn_type,sn_absolute_mag_band, sn_absolute_zpsys, sn_modeldir,
        super().__init__(
            sky_area,
            cosmo,
            lightcurve_time,
            sn_type,
            sn_absolute_mag_band,
            sn_absolute_zpsys,
            sn_modeldir,
        )

        # Asignación del objeto de cosmología a un atributo de la instancia. OJO ASIGNANCION REDUNDANTE######################
        self.cosmo = cosmo

        # Configuración del área del cielo para las fuentes. Si no se especifica, se
        # utiliza el área de cielo definida en sky_area y asignada como self.f_sky en la clase base (LensedPopulationBase).
        if source_sky_area is None:
            # Si source_sky_area no se proporciona, usa self.f_sky de la clase base LensedPopulationBase
            self.source_sky_area = self.f_sky
        else:
            # Si el valor de source_sky_area se proporciona, úsalo directamente.
            self.source_sky_area = source_sky_area

        # Configuración del área del cielo para los deflectores. Similar al caso de las fuentes.
        if deflector_sky_area is None:
            self.deflector_sky_area = self.f_sky
        else:
            self.deflector_sky_area = deflector_sky_area

        # Validación de la combinación de tipo de fuente y modelo de variabilidad.
        # Por ejemplo, no tiene sentido aplicar un modelo de variabilidad a una galaxia.
        if source_type == "galaxies" and kwargs_variability is not None:
            raise ValueError(
                "Galaxies cannot have variability. Either choose"
                "point source (e.g., quasars) or do not provide kwargs_variability."
            )

        # Configuración del pipeline para deflectores y fuentes, según el tipo de deflector
        # y fuente seleccionados.
        if deflector_type in ["elliptical", "all-galaxies"] or source_type in [
            "galaxies"
        ]:
            pipeline_deflector = SkyPyPipeline(
                skypy_config=skypy_config,
                sky_area=self.deflector_sky_area,
                filters=filters,
                cosmo=cosmo,
            )

            # Si el área del cielo de las fuentes y los deflectores es la misma,
            # se utiliza el mismo pipeline para ambos.
            if self.source_sky_area == self.deflector_sky_area:
                pipeline_source = pipeline_deflector
            else:
                pipeline_source = SkyPyPipeline(
                    skypy_config=skypy_config,
                    sky_area=self.source_sky_area,
                    filters=filters,
                    cosmo=cosmo,
                )

        # Inicialización de diccionarios vacíos si no se proporcionan cortes para deflectores o la relación masa-luz.
        if kwargs_deflector_cut is None:
            kwargs_deflector_cut = {}
        if kwargs_mass2light is None:
            kwargs_mass2light = {}

        # Configuración del tipo de deflector. Dependiendo del tipo, se inicializa
        # la clase correspondiente de deflectores.
        if deflector_type == "elliptical":
            from slsim.Deflectors.elliptical_lens_galaxies import EllipticalLensGalaxies

            self._lens_galaxies = EllipticalLensGalaxies(
                pipeline_deflector.red_galaxies,
                kwargs_cut=kwargs_deflector_cut,
                kwargs_mass2light=kwargs_mass2light,
                cosmo=cosmo,
                sky_area=self.deflector_sky_area,
            )

        elif deflector_type == "all-galaxies":
            from slsim.Deflectors.all_lens_galaxies import AllLensGalaxies

            red_galaxy_list = pipeline_deflector.red_galaxies
            blue_galaxy_list = pipeline_deflector.blue_galaxies

            self._lens_galaxies = AllLensGalaxies(
                red_galaxy_list=red_galaxy_list,
                blue_galaxy_list=blue_galaxy_list,
                kwargs_cut=kwargs_deflector_cut,
                kwargs_mass2light=kwargs_mass2light,
                cosmo=cosmo,
                sky_area=self.deflector_sky_area,
            )

        elif deflector_type == "halo-models":
            from slsim.Deflectors.compound_lens_halos_galaxies import (
                CompoundLensHalosGalaxies,
            )
            from slsim.Pipelines.sl_hammocks_pipeline import SLHammocksPipeline

            halo_galaxy_list = SLHammocksPipeline(
                slhammocks_config=slhammocks_config,
                sky_area=self.deflector_sky_area,
                cosmo=cosmo,
            )

            self._lens_galaxies = CompoundLensHalosGalaxies(
                halo_galaxy_list=halo_galaxy_list._pipeline,
                kwargs_cut=kwargs_deflector_cut,
                kwargs_mass2light=kwargs_mass2light,
                cosmo=cosmo,
                sky_area=self.deflector_sky_area,
            )

        else:
            # Si se proporciona un tipo de deflector no soportado, se lanza una excepción.
            raise ValueError("deflector_type %s is not supported" % deflector_type)

        # Configuración del tipo de fuente. Similar al proceso de deflectores,
        # dependiendo del tipo de fuente se inicializa la clase correspondiente.
        if kwargs_source_cut is None:
            kwargs_source_cut = {}
        if source_type == "galaxies":
            from slsim.Sources.galaxies import Galaxies

            self._sources = Galaxies(
                pipeline_source.blue_galaxies,
                kwargs_cut=kwargs_source_cut,
                cosmo=cosmo,
                sky_area=self.source_sky_area,
                light_profile=source_light_profile,
                catalog_type=catalog_type,
            )
            self._source_model_type = "extended"
        elif source_type == "quasars":
            from slsim.Sources.point_sources import PointSources
            from slsim.Sources.QuasarCatalog.quasar_pop import QuasarRate

            if kwargs_quasars is None:
                kwargs_quasars = {}
            quasar_class = QuasarRate(
                cosmo=cosmo,
                sky_area=self.source_sky_area,
                noise=True,
                redshifts=np.linspace(0.001, 5.01, 100),
            )
            quasar_source = quasar_class.quasar_sample(m_min=15, m_max=30)
            self._sources = PointSources(
                quasar_source,
                cosmo=cosmo,
                sky_area=self.source_sky_area,
                kwargs_cut=kwargs_source_cut,
                variability_model=variability_model,
                kwargs_variability_model=kwargs_variability,
                light_profile=source_light_profile,
            )
            self._source_model_type = "point_source"
        elif source_type == "quasar_plus_galaxies":
            from slsim.Sources.point_plus_extended_sources import (
                PointPlusExtendedSources,
            )
            from slsim.Sources.QuasarCatalog.quasar_plus_galaxies import (
                quasar_galaxies_simple,
            )

            if kwargs_quasars_galaxies is None:
                kwargs_quasars_galaxies = {}
            quasar_galaxy_source = quasar_galaxies_simple(**kwargs_quasars_galaxies)
            self._sources = PointPlusExtendedSources(
                quasar_galaxy_source,
                cosmo=cosmo,
                sky_area=self.source_sky_area,
                kwargs_cut=kwargs_source_cut,
                variability_model=variability_model,
                kwargs_variability_model=kwargs_variability,
                light_profile=source_light_profile,
                catalog_type=catalog_type,
            )
            self._source_model_type = "point_plus_extended"
        elif source_type in ["supernovae_plus_galaxies", "supernovae"]:
            from slsim.Sources.point_plus_extended_sources import (
                PointPlusExtendedSources,
            )
            from slsim.Sources.point_sources import PointSources

            self.path = os.path.dirname(__file__)
            if catalog_type == "scotch":
                if catalog_path is not None:
                    new_path = catalog_path
                else:
                    new_path = (
                            self.path + "/Sources/SupernovaeCatalog/scotch_host_data.fits"
                    )
                load_supernovae_data = Table.read(
                    new_path,
                    format="fits",
                )
                self._sources = PointPlusExtendedSources(
                    load_supernovae_data,
                    cosmo=cosmo,
                    sky_area=self.source_sky_area,
                    kwargs_cut=kwargs_source_cut,
                    variability_model=variability_model,
                    kwargs_variability_model=kwargs_variability,
                    list_type="astropy_table",
                    light_profile=source_light_profile,
                    catalog_type=catalog_type,
                )
            elif catalog_type == "supernovae_sample":
                new_path = self.path + "/Sources/SupernovaeCatalog/supernovae_data.pkl"
                with open(new_path, "rb") as f:
                    load_supernovae_data = pickle.load(f)
                self._sources = PointPlusExtendedSources(
                    load_supernovae_data,
                    cosmo=cosmo,
                    sky_area=self.source_sky_area,
                    kwargs_cut=kwargs_source_cut,
                    variability_model=variability_model,
                    kwargs_variability_model=kwargs_variability,
                    list_type="list",
                    light_profile=source_light_profile,
                )
            else:
                from slsim.Sources.SupernovaeCatalog.supernovae_sample import (
                    SupernovaeCatalog,
                )

                suffixes = []
                for key in kwargs_variability:
                    if key.startswith("ps_mag_"):
                        suffixes.append(key.split("ps_mag_")[1])
                    elif len(key) == 1:
                        suffixes.append(key)
                supernovae_catalog_class = SupernovaeCatalog(
                    sn_type=sn_type,
                    band_list=suffixes,
                    lightcurve_time=lightcurve_time,
                    absolute_mag=None,
                    absolute_mag_band=sn_absolute_mag_band,
                    mag_zpsys=sn_absolute_zpsys,
                    cosmo=cosmo,
                    skypy_config=skypy_config,
                    sky_area=self.source_sky_area,
                    sn_modeldir=sn_modeldir,
                )
                if source_type == "supernovae":
                    supernovae_sample = supernovae_catalog_class.supernovae_catalog(
                        host_galaxy=False, lightcurve=False
                    )
                    self._sources = PointSources(
                        supernovae_sample,
                        cosmo=cosmo,
                        sky_area=self.source_sky_area,
                        kwargs_cut=kwargs_source_cut,
                        variability_model=variability_model,
                        kwargs_variability_model=kwargs_variability,
                        light_profile=source_light_profile,
                    )
                else:
                    supernovae_sample = supernovae_catalog_class.supernovae_catalog(
                        host_galaxy=True, lightcurve=False
                    )
                    self._sources = PointPlusExtendedSources(
                        supernovae_sample,
                        cosmo=cosmo,
                        sky_area=self.source_sky_area,
                        kwargs_cut=kwargs_source_cut,
                        variability_model=variability_model,
                        kwargs_variability_model=kwargs_variability,
                        list_type="astropy_table",
                        light_profile=source_light_profile,
                    )
            if source_type == "supernovae":
                self._source_model_type = "point_source"
            else:
                self._source_model_type = "point_plus_extended"
        else:
            raise ValueError("source_type %s is not supported" % source_type)

        # Cálculo de factores de escala para las áreas del cielo de las fuentes y los deflectores.
        self._factor_source = self.f_sky.to_value(
            "deg2"
        ) / self.source_sky_area.to_value("deg2")
        self._factor_deflector = self.f_sky.to_value(
            "deg2"
        ) / self.deflector_sky_area.to_value("deg2")

        # Configuración de la línea de visión (los_config).
        self.los_config = los_config
        if self.los_config is None:
            self.los_config = LOSConfig()

    def select_lens_at_random(self, **kwargs_lens_cut):
        """Selecciona un lente aleatorio que cumpla con los criterios especificados.

        Este método sigue seleccionando una fuente y un deflector al azar hasta encontrar
        una combinación que pase las pruebas de validez especificadas en kwargs_lens_cut.

        :return: Una instancia de Lens con los parámetros del deflector, fuente y lente.
        """
        while True:
            source = self._sources.draw_source()  # Selecciona una fuente al azar.
            lens = self._lens_galaxies.draw_deflector()  # Selecciona un deflector al azar.
            gg_lens = Lens(
                deflector_dict=lens,
                source_dict=source,
                variability_model=self._sources.variability_model,
                kwargs_variability=self._sources.kwargs_variability,
                sn_type=self.sn_type,
                sn_absolute_mag_band=self.sn_absolute_mag_band,
                sn_absolute_zpsys=self.sn_absolute_zpsys,
                cosmo=self.cosmo,
                source_type=self._source_model_type,
                light_profile=self._sources.light_profile,
                lightcurve_time=self.lightcurve_time,
                los_config=self.los_config,
                sn_modeldir=self.sn_modeldir,
            )
            # Si el sistema de lentes es válido según los criterios, lo devuelve.
            if gg_lens.validity_test(**kwargs_lens_cut):
                return gg_lens

    #El decorador @property se utiliza para crear propiedades en una clase.
    #Una propiedad es un método que se comporta como un atributo. Ejem: 'objeto.deflector_number' en lugar de 'objeto.deflector_number()'
    @property
    def deflector_number(self):
        """Calcula el número de deflectores potenciales.

        :return: El número de deflectores, escalado según el área del cielo.
        # Calcula el número de deflectores potenciales en función del área del cielo y el número de deflectores.
        """
        # _factor_deflector escala el número de deflectores al área del cielo relevante.
        # self._lens_galaxies.deflector_number() obtiene el número de deflectores desde el objeto de lentes.
        return round(self._factor_deflector * self._lens_galaxies.deflector_number())
        #El guion bajo (_) es una convención que indica que un atributo o método es "privado"
        #ya que su utilizacion esta reservada para uso dentro de la clase.

    @property
    def source_number(self):
        """Calcula el número de fuentes consideradas para la simulación.

        :return: El número de fuentes, escalado según el área del cielo.
        """
        return round(self._factor_source * self._sources.source_number_selected)

    def get_num_sources_tested_mean(self, testarea):
        """Calcula el número promedio de fuentes que deben ser probadas en un área de prueba.

        Utiliza la fórmula:
        num_sources_tested_mean/ testarea = num_sources/ f_sky
        testarea es en arcsec^2, f_sky es en deg^2.

        :param testarea: El área de prueba en unidades de arcsec^2.
        :return: El número promedio de fuentes probadas.
        """
        num_sources = self.source_number # Obtiene el número total de fuentes escaladas según el área del cielo.
        # Calcula el número promedio de fuentes que serán probadas en el área de prueba proporcionada.
        num_sources_tested_mean = (testarea * num_sources) / (
                12960000 * self._factor_source * self.source_sky_area.to_value("deg2")
        )
        return num_sources_tested_mean # Devuelve el número promedio de fuentes probadas en el área de prueba.

    def get_num_sources_tested(self, testarea=None, num_sources_tested_mean=None):
        """Genera una realización de la distribución esperada para el número de fuentes probadas.

        Si no se proporciona un número promedio de fuentes probadas, se calcula usando el área de prueba.
        Se usa una distribución de Poisson para obtener el número real de fuentes probadas.

        :param testarea: Área de prueba en arcsec^2.
        :param num_sources_tested_mean: Número promedio de fuentes probadas.
        :return: Número de fuentes probadas en esta realización.
        """
        if num_sources_tested_mean is None:
            num_sources_tested_mean = self.get_num_sources_tested_mean(testarea)
        num_sources_range = np.random.poisson(lam=num_sources_tested_mean)
        return num_sources_range

    def draw_population(self, kwargs_lens_cuts, speed_factor=1):
        """Dibuja una población completa de lentes dentro del área especificada.

        Este método genera una lista de instancias de Lens que cumplen con los criterios
        especificados en kwargs_lens_cuts.

        :param kwargs_lens_cuts: Criterios de validación para los lentes.
        :param speed_factor: Factor para reducir el número de deflectores y acelerar el cálculo.
        :return: Lista de instancias de Lens.
        """
        gg_lens_population = []  # Inicializa una lista vacía para almacenar las instancias de Lens.
        num_lenses = self.deflector_number  # Calcula el número de lentes esperados.

        # Bucle para generar la población de lentes.
        for _ in range(int(num_lenses / speed_factor)):
            lens = self._lens_galaxies.draw_deflector()  # Selecciona un deflector.
            test_area = draw_test_area(deflector=lens)  # Calcula el área de prueba alrededor del deflector.
            num_sources_tested = self.get_num_sources_tested(
                testarea=test_area * speed_factor
            )

            # Si hay fuentes para probar, verificar cada una.
            if num_sources_tested > 0:
                n = 0
                while n < num_sources_tested:
                    source = self._sources.draw_source()  # Selecciona una fuente.
                    gg_lens = Lens(
                        deflector_dict=lens,
                        source_dict=source,
                        variability_model=self._sources.variability_model,
                        kwargs_variability=self._sources.kwargs_variability,
                        sn_type=self.sn_type,
                        sn_absolute_mag_band=self.sn_absolute_mag_band,
                        sn_absolute_zpsys=self.sn_absolute_zpsys,
                        cosmo=self.cosmo,
                        test_area=test_area,
                        source_type=self._source_model_type,
                        los_config=self.los_config,
                        light_profile=self._sources.light_profile,
                        lightcurve_time=self.lightcurve_time,
                        sn_modeldir=self.sn_modeldir,
                    )
                    # Si el sistema de lentes pasa la prueba de validez, se añade a la población.
                    if gg_lens.validity_test(**kwargs_lens_cuts):
                        gg_lens_population.append(gg_lens)
                        n = num_sources_tested  # Salir del bucle si se encuentra un lente válido.
                    else:
                        n += 1
        return gg_lens_population  # Devuelve la población de lentes generada.


def draw_test_area(deflector):
    """Calcula un área de prueba alrededor del deflector.

    :param deflector: Diccionario que contiene los parámetros del deflector.
    :return: Área de prueba en arcsec^2.
    """
    theta_e_infinity = theta_e_when_source_infinity(
        deflector)  # Calcula el radio de Einstein para una fuente a z=infinito.
    test_area = np.pi * (theta_e_infinity * 2.5) ** 2  # Calcula el área circular basada en este radio.
    return test_area  # Devuelve el área de prueba.
