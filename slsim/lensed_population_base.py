from abc import ABC, abstractmethod
import warnings

# Se importa ABC para crear una clase base abstracta y abstractmethod para definir métodos abstractos
# que deben ser implementados en las clases derivadas.


# Definición de la clase abstracta LensedPopulationBase que hereda de ABC (Abstract Base Class)
class LensedPopulationBase(ABC):
    """Clase base abstracta que proporciona la estructura necesaria para cualquier clase
    que desee crear sistemas con lentes gravitacionales.

    Es abstracta porque no se puede instanciar directamente; requiere que las clases
    derivadas implementen los métodos.
    """

    # Constructor de la clase
    # Inicializa una instancia de la clase con parámetros que definen las condiciones del área del cielo,
    # el modelo cosmológico, y otras configuraciones relacionadas con las curvas de luz.
    def __init__(
        self,
        sky_area=None,  # Área del cielo (ángulo sólido) sobre la cual se muestrean las galaxias.
        cosmo=None,  # Instancia de cosmología que describe el modelo cosmológico utilizado.
        lightcurve_time=None,  # Array de tiempo de observación para la curva de luz en días.
        sn_type=None,  # Tipo de supernova (Ia, Ib, Ic, IIP, etc.).
        sn_absolute_mag_band=None,  # Banda utilizada para normalizar la magnitud absoluta.
        sn_absolute_zpsys=None,  # Sistema de magnitud cero, puede ser AB o Vega (AB por defecto).
        sn_modeldir=None,  # Directorio que contiene los archivos necesarios para inicializar la clase sncosmo.model.
    ):
        """Inicializa la clase LensedPopulationBase con los parámetros proporcionados.

        :param sky_area: Área del cielo (ángulo sólido) sobre la cual se muestrean las galaxias.
        :type sky_area: `~astropy.units.Quantity`
        :param cosmo: Modelo cosmológico utilizado.
        :type cosmo: ~astropy.cosmology instance
        :param lightcurve_time: Array de tiempo de observación para la curva de luz en días.
        :type lightcurve_time: array
        :param sn_type: Tipo de supernova (Ia, Ib, Ic, IIP, etc.).
        :type sn_type: str
        :param sn_absolute_mag_band: Banda utilizada para normalizar la magnitud absoluta.
        :type sn_absolute_mag_band: str or `~sncosmo.Bandpass`
        :param sn_absolute_zpsys: Sistema de magnitud cero, puede ser AB o Vega (AB por defecto).
        :type sn_absolute_zpsys: str
        :param sn_modeldir: Directorio que contiene los archivos necesarios para inicializar la clase sncosmo.model.
        :type sn_modeldir: str
        """

        # Asigna los parámetros proporcionados a los atributos de la instancia
        self.lightcurve_time = lightcurve_time
        self.sn_type = sn_type
        self.sn_absolute_mag_band = sn_absolute_mag_band
        self.sn_absolute_zpsys = sn_absolute_zpsys
        self.sn_modeldir = sn_modeldir

        # Si no se proporciona sky_area, se establece un valor por defecto y se emite una advertencia
        if sky_area is None:
            from astropy.units import (
                Quantity,
            )  # Importa la clase Quantity de astropy para manejar unidades físicas

            sky_area = Quantity(
                value=0.1, unit="deg2"
            )  # Establece el valor por defecto del área del cielo a 0.1 grados cuadrados
            warnings.warn(
                "No sky area provided, instead uses 0.1 deg2"
            )  # Emite una advertencia indicando que se ha utilizado el valor por defecto
        self.f_sky = (
            sky_area  # Asigna el valor del área del cielo al atributo de la instancia
        )

        # Si no se proporciona cosmo, se establece un modelo cosmológico por defecto y se emite una advertencia
        if cosmo is None:
            warnings.warn(
                "No cosmology provided, instead uses flat LCDM with default parameters"
            )  # Emite una advertencia indicando que se ha utilizado un modelo cosmológico por defecto
            from astropy.cosmology import (
                FlatLambdaCDM,
            )  # Importa la clase FlatLambdaCDM de astropy.cosmology

            cosmo = FlatLambdaCDM(
                H0=70, Om0=0.3
            )  # Establece un modelo cosmológico plano con parámetros por defecto
        self.cosmo = cosmo  # Asigna el modelo cosmológico al atributo de la instancia

    # Método abstracto que debe ser implementado en las clases derivadas
    @abstractmethod
    def select_lens_at_random(self):
        """Selecciona un lente aleatorio dentro de los cortes del lente y la fuente, con
        posibles cortes adicionales en la configuración de lentes.

        :return: Una instancia de Lens() con los parámetros del deflector y la luz de la
            lente y la fuente.
        """
        pass  # Método no implementado, debe ser sobrescrito por clases derivadas

    # Método abstracto que debe ser implementado en las clases derivadas
    @abstractmethod
    def deflector_number(self):
        """Número de posibles deflectores (objetos con masa que se consideran como
        posibles fuentes de lentes gravitacionales).

        :return: Número de posibles deflectores.
        """
        pass  # Método no implementado, debe ser sobrescrito por clases derivadas

    # Método abstracto que debe ser implementado en las clases derivadas
    @abstractmethod
    def source_number(self):
        """Número de fuentes que se están considerando para ser colocadas en el área del
        cielo, potencialmente alineadas detrás de los deflectores.

        :return: Número de fuentes.
        """
        pass  # Método no implementado, debe ser sobrescrito por clases derivadas

    # Método abstracto que debe ser implementado en las clases derivadas
    @abstractmethod
    def draw_population(self, **kwargs):
        """Devuelve la lista completa de todas las lentes dentro del área.

        :return: Lista de instancias de LensedSystemBase con los parámetros de los
            deflectores y la fuente.
        :rtype: list
        """
        pass  # Método no implementado, debe ser sobrescrito por clases derivadas
