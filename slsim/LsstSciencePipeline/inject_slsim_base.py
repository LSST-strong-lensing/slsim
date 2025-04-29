from __future__ import annotations

__all__ = ["BaseInjectSLSimConnections", "BaseInjectSLSimConfig", "BaseInjectSLSimTask"]

from typing import cast

import numpy as np

import lsst.geom as geom
from astropy.table import Table, hstack, vstack
from lsst.pex.config import Field
from lsst.pipe.base.connectionTypes import PrerequisiteInput
from lsst.pex.exceptions import InvalidParameterError
from lsst.pipe.base import Struct

from astropy.cosmology import FlatLambdaCDM

from lsst.source.injection.inject_base import (
    BaseInjectConnections,
    BaseInjectConfig,
    BaseInjectTask,
)

from slsim.Sources.source import Source
from slsim.Deflectors.deflector import Deflector
from slsim.LOS.los_pop import LOSPop
from slsim.lens import Lens
from slsim.image_simulation import lens_image


class BaseInjectSLSimConnections(
    BaseInjectConnections,
    dimensions=("instrument",),
    defaultTemplates={
        "injection_prefix": "injection_slsim_",
        "injected_prefix": "injected_slsim_",
    },
):
    """Base connections for strong lensing source injection tasks.

    This class extends the BaseInjectConnections class from LSST's source injection
    framework. For general information on how to use the injection framework, refer
    to the LSST source injection classes documentation at:
    https://github.com/lsst/source-injection
    """

    injection_catalogs = PrerequisiteInput(
        doc="Set of catalogs of sources to draw inputs from.",
        # name="{injection_prefix}catalog",
        name="injection_slsim",
        dimensions=("htm7", "band"),
        storageClass="ArrowAstropy",
        minimum=0,
        multiple=True,
    )


class BaseInjectSLSimConfig(
    BaseInjectConfig, pipelineConnections=BaseInjectSLSimConnections
):
    """Base configuration for strong lensing source injection tasks.

    This class extends the BaseInjectConfig class from LSST's source injection
    framework and customizes it for strong lensing injection. For general
    information on how to use the injection framework's configuration options,
    refer to the LSST source injection documentation at:
    https://github.com/lsst/source-injection
    """

    # Catalog manipulation options.
    process_all_data_ids = Field[bool](
        doc="If True, all input data IDs will be processed, even those where no synthetic sources were "
        "identified for injection. In such an eventuality this returns a clone of the input image, renamed "
        "to the *output_exposure* connection name and with an empty *mask_plane_name* mask plane attached.",
        default=False,
    )
    trim_padding = Field[int](
        doc="Size of the pixel padding surrounding the image. Only those synthetic sources with a centroid "
        "falling within the ``image + trim_padding`` region will be considered for source injection.",
        default=100,
        optional=True,
    )
    selection = Field[str](
        doc="A string that can be evaluated as a boolean expression to select rows in the input injection "
        "catalog. To make use of this configuration option, the internal object name ``injection_catalog`` "
        "must be used. For example, to select all sources with a magnitude in the range 20.0 < mag < 25.0, "
        "set ``selection=\"(injection_catalog['mag'] > 20.0) & (injection_catalog['mag'] < 25.0)\"``. "
        "The ``{visit}`` field will be substituted for the current visit ID of the exposure being processed. "
        "For example, to select only visits that match a user-supplied visit column in the input injection "
        "catalog, set ``selection=\"np.isin(injection_catalog['visit'], {visit})\"``.",
        optional=True,
    )
    # General configuration options.
    mask_plane_name = Field[str](
        doc="Name assigned to the injected mask plane which is attached to the output exposure.",
        default="SLSIM_INJECTED",
    )
    # Size of injected pixels, same for all
    stamp_size = Field[int](
        doc="Size of stamp for each injected source",
        default=60,
        optional=True,
    )

    def setDefaults(self):
        """Set defaults for configuration parameters.

        :return: None
        """
        super().setDefaults()


class BaseInjectSLSimTask(BaseInjectTask):
    """Base class for injecting strong lensing sources into images.

    This class extends the BaseInjectTask class from LSST's source injection
    framework to handle the specific requirements of strong lensing image simulation
    using the SLSim package. It uses the same overall workflow as the base class,
    but adds specialized handling for lens-source galaxy pairs and their images.

    For information on the general source injection framework and how to use it,
    refer to the LSST source injection documentation at:
    https://github.com/lsst/source-injection

    For details on the SLSim strong lensing simulation package:
    https://github.com/LSST-strong-lensing/slsim
    """

    _DefaultName = "baseInjectSLSimTask"
    ConfigClass = BaseInjectSLSimConfig

    def run(self, injection_catalogs, input_exposure, psf, photo_calib, wcs):
        """Inject strong lensing sources into an image.

        :param injection_catalogs: Tract level injection catalogs that potentially cover the named input exposure
        :type injection_catalogs: `list` [`astropy.table.Table`]
        :param input_exposure: The exposure sources will be injected into
        :type input_exposure: `lsst.afw.image.ExposureF`
        :param psf: PSF model
        :type psf: `lsst.meas.algorithms.ImagePsf`
        :param photo_calib: Photometric calibration used to calibrate injected sources
        :type photo_calib: `lsst.afw.image.PhotoCalib`
        :param wcs: WCS used to calibrate injected sources
        :type wcs: `lsst.afw.geom.SkyWcs`
        :return: Struct containing output_exposure and output_catalog
        :rtype: `lsst.pipe.base.Struct` with output_exposure (`lsst.afw.image.ExposureF`) and output_catalog (`lsst.afw.table.SourceCatalog`)
        """
        self.config = cast(BaseInjectSLSimConfig, self.config)

        # Make empty table if none supplied to support process_all_data_ids.
        if len(injection_catalogs) == 0:
            if self.config.process_all_data_ids:
                injection_catalogs = [Table(names=["ra", "dec"])]
            else:
                raise RuntimeError(
                    "No injection sources overlap the data query. Check injection catalog coverage."
                )

        # Consolidate injection catalogs and compose main injection catalog.
        injection_catalog = self._compose_injection_catalog(injection_catalogs)

        # Clean the injection catalog of sources which are not injectable.
        injection_catalog = self._clean_sources(injection_catalog, input_exposure)

        # Injection binary flag lookup dictionary.
        binary_flags = {
            "SLSIM_FAILURE": 0,
            "NOT_FULL_OVERLAP": 1,
            "PSF_COMPUTE_ERROR": 2,
        }

        # Check that sources in the injection catalog are able to be injected.
        injection_catalog = self._check_sources(injection_catalog, binary_flags)

        # Inject sources into input_exposure.
        good_injections: list[bool] = injection_catalog["injection_flag"] == 0
        good_injections_index = [i for i, val in enumerate(good_injections) if val]
        num_injection_sources = np.sum(good_injections)
        calib_flux_radius = None
        num_pix = self.config.stamp_size
        
        if num_injection_sources > 0:
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

            exp_bbox = input_exposure.getBBox()
            zeropoint = 2.5 * np.log10(photo_calib.getInstFluxAtZeroMagnitude())
            band = input_exposure.getInfo().getFilter().bandLabel
            exp_time = input_exposure.getInfo().getVisitInfo().getExposureTime()

            lens_mask = injection_catalog[good_injections]["type"] == "lens"
            source_mask = injection_catalog[good_injections]["type"] == "source"
            lens_index = injection_catalog[good_injections][lens_mask]["id"].data
            for li in lens_index:
                lens_sel = injection_catalog[good_injections][lens_mask]["id"] == li
                source_sel = injection_catalog[good_injections][source_mask]["id"] == li
                lens_dict = Table(
                    injection_catalog[good_injections][lens_mask][lens_sel]
                )
                source_dict = Table(
                    injection_catalog[good_injections][source_mask][source_sel]
                )

                sky = geom.SpherePoint(
                    lens_dict["ra"][0], lens_dict["dec"][0], geom.degrees
                )
                xy = wcs.skyToPixel(sky)

                try:
                    psf_xy = psf.computeKernelImage(xy).array
                except InvalidParameterError:
                    injection_catalog[good_injections][lens_mask]["injection_flag"] += (
                        2 ** binary_flags["PSF_COMPUTE_ERROR"]
                    )
                    injection_catalog[good_injections][source_mask][
                        "injection_flag"
                    ] += (2 ** binary_flags["PSF_COMPUTE_ERROR"])
                    continue

                pixscale = wcs.getPixelScale(xy).asArcseconds()
                bbox = geom.Box2I(
                    geom.Point2I(xy.getX() - num_pix / 2, xy.getY() - num_pix / 2),
                    geom.Extent2I(num_pix, num_pix),
                )

                if exp_bbox.contains(bbox) == False:
                    injection_catalog[good_injections][lens_mask]["injection_flag"] += (
                        2 ** binary_flags["NOT_FULL_OVERLAP"]
                    )
                    injection_catalog[good_injections][source_mask][
                        "injection_flag"
                    ] += (2 ** binary_flags["NOT_FULL_OVERLAP"])
                    continue

                if self.config.calib_flux_radius is not None:
                    apCorr = psf.computeApertureFlux(self.config.calib_flux_radius, xy)
                    psf_ker = psf_xy / apCorr
                else:
                    psf_ker = psf_xy

                matrix = (
                    wcs.linearizePixelToSky(sky, geom.arcseconds)
                    .getLinear()
                    .getMatrix()
                )

                # SLSim has problems with matrix that include reflections like
                # because it assumes det is positive, need to recreate matrix
                scale = np.sqrt(abs(np.linalg.det(matrix)))
                matrix = np.array([[scale, 0], [0, scale]])

                # Consolidate vector objects because butler stored them separately
                source_dict["coeff"] = np.array(
                    [
                        [
                            source_dict["coeff0"],
                            source_dict["coeff1"],
                            source_dict["coeff2"],
                            source_dict["coeff3"],
                            source_dict["coeff4"],
                        ]
                    ]
                )
                lens_dict["coeff"] = np.array(
                    [
                        [
                            lens_dict["coeff0"],
                            lens_dict["coeff1"],
                            lens_dict["coeff2"],
                            lens_dict["coeff3"],
                            lens_dict["coeff4"],
                        ]
                    ]
                )

                try:
                    source = Source(
                        source_dict=source_dict,
                        cosmo=cosmo,
                        source_type=source_dict["source_type"][0],
                        light_profile=source_dict["light_profile"][0],
                    )
                    lens = Deflector(
                        deflector_type=lens_dict["deflector_type"][0],
                        deflector_dict=lens_dict,
                    )
                    los_pop = LOSPop()
                    los = los_pop.draw_los(
                        source_redshift=source.redshift,
                        deflector_redshift=lens.redshift,
                    )
                    
                    lens_class = Lens(
                        source_class=source,
                        deflector_class=lens,
                        cosmo=cosmo,
                        los_class=los,
                    )
                    
                    lens_im = lens_image(
                        lens_class=lens_class,
                        band=band,
                        mag_zero_point=zeropoint,
                        num_pix=num_pix,
                        psf_kernel=psf_ker,
                        transform_pix2angle=matrix,
                        exposure_time=exp_time,
                    )
                except InvalidParameterError:
                    injection_catalog[good_injections][lens_mask]["injection_flag"] += (
                        2 ** binary_flags["SLSIM_FAILURE"]
                    )
                    injection_catalog[good_injections][source_mask][
                        "injection_flag"
                    ] += (2 ** binary_flags["SLSIM_FAILURE"])
                except Exception as e:
                    print("Failed: ", e)
                    continue

                if np.sum(lens_im) > 0:
                    input_exposure.mask.addMaskPlane(self.config.mask_plane_name)
                    bitvalue = input_exposure.mask.getPlaneBitMask(
                        self.config.mask_plane_name
                    )
                    input_exposure[bbox].mask.array |= bitvalue

                    input_exposure[bbox].image.array += lens_im

        # Add injection provenance and injection flags metadata.
        metadata = input_exposure.getMetadata()
        input_dataset_type = self.config.connections.input_exposure.format(
            **self.config.connections.toDict()
        )
        metadata.set(
            "SLSIM_INJECTED",
            input_dataset_type,
            "Initial source injection dataset type",
        )
        for flag, value in sorted(binary_flags.items(), key=lambda item: item[1]):
            injection_catalog.meta[flag] = value

        output_struct = Struct(
            output_exposure=input_exposure, output_catalog=injection_catalog
        )
        return output_struct

    def _compose_injection_catalog(self, injection_catalogs):
        """Consolidate injection catalogs and compose main injection catalog.

        If multiple injection catalogs are input, all catalogs are
        concatenated together.

        A running injection_id, specific to this dataset ref, is assigned to
        each source in the output injection catalog if not provided.

        :param injection_catalogs: Set of synthetic source catalogs to concatenate
        :type injection_catalogs: `list` [`astropy.table.Table`]
        :return: Catalog of sources to be injected
        :rtype: `astropy.table.Table`
        """
        self.config = cast(BaseInjectConfig, self.config)

        # Generate injection IDs (if not provided) and injection flag column.
        injection_data = vstack(injection_catalogs)
        if "injection_id" in injection_data.columns:
            injection_id = injection_data["injection_id"]
            injection_data.remove_column("injection_id")
        else:
            injection_id = range(len(injection_data))
        injection_header = Table(
            {
                "injection_id": injection_id,
                "injection_flag": np.zeros(len(injection_data), dtype=int),
            }
        )

        # Construct final injection catalog.
        injection_catalog = hstack([injection_header, injection_data])

        # Log and return.
        num_injection_catalogs = np.sum(
            [len(table) > 0 for table in injection_catalogs]
        )
        grammar1 = "source" if len(injection_catalog) == 1 else "sources"
        grammar2 = "trixel" if num_injection_catalogs == 1 else "trixels"
        self.log.info(
            "Retrieved %d injection %s from %d HTM %s.",
            len(injection_catalog),
            grammar1,
            num_injection_catalogs,
            grammar2,
        )
        return injection_catalog

    def _check_sources(self, injection_catalog, binary_flags):
        """Check that sources in the injection catalog are able to be injected.

        This method will check that sources in the injection catalog are able
        to be injected, and will flag them if not. Checks will be made on a
        number of parameters, including magnitude, source type and Sérsic index
        (where relevant).

        Legacy profile types will be renamed to their standardized GalSim
        equivalents; any source profile types that are not GalSim classes will
        be flagged.

        Note: Unlike the cleaning method, no sources are actually removed here.
        Instead, a binary flag is set in the *injection_flag* column for each
        source. Only unflagged sources will be generated for source injection.

        :param injection_catalog: Catalog of sources to be injected
        :type injection_catalog: `astropy.table.Table`
        :param binary_flags: Dictionary of binary flags to be used in the injection_flag column
        :type binary_flags: `dict` [`str`, `int`]
        :return: The cleaned catalog of sources to be injected
        :rtype: `astropy.table.Table`
        """
        self.config = cast(BaseInjectConfig, self.config)

        # Exit early if there are no sources to inject.
        if len(injection_catalog) == 0:
            self.log.info("Catalog checking not applied to empty injection catalog.")
            return injection_catalog

        num_flagged_total = np.sum(injection_catalog["injection_flag"] != 0)
        grammar = "source" if len(injection_catalog) == 1 else "sources"
        self.log.info(
            "Catalog checking flagged %d of %d %s; %d remaining for source generation.",
            num_flagged_total,
            len(injection_catalog),
            grammar,
            np.sum(injection_catalog["injection_flag"] == 0),
        )
        return injection_catalog
