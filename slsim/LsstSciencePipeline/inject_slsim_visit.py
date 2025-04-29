from __future__ import annotations

__all__ = [
    "VisitInjectSLSimConnections",
    "VisitInjectSLSimConfig",
    "VisitInjectSLSimTask",
]

from typing import cast

from lsst.pex.config import Field
from lsst.pipe.base.connectionTypes import Input, Output

from .inject_slsim_base import (
    BaseInjectSLSimConfig,
    BaseInjectSLSimConnections,
    BaseInjectSLSimTask,
)


class VisitInjectSLSimConnections(  # type: ignore [call-arg]
    BaseInjectSLSimConnections,
    dimensions=("instrument", "visit", "detector"),
):
    """Visit-level connections for strong lensing source injection tasks.

    This class extends BaseInjectSLSimConnections to handle visit-level injections.
    It configures the appropriate input and output connections for the visit-level
    injection tasks.

    For information on the LSST source injection framework this is built upon:
    https://github.com/lsst/source-injection
    """

    visit_summary = Input(
        doc="A visit summary table containing PSF, PhotoCalib and WCS information.",
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=("visit",),
        deferLoad=True,
    )
    input_exposure = Input(
        doc="Exposure to inject synthetic sources into.",
        name="calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )
    output_exposure = Output(
        doc="Injected Exposure.",
        name="{injected_prefix}calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )
    output_catalog = Output(
        doc="Catalog of injected sources.",
        name="{injected_prefix}calexp_catalog",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit", "detector"),
    )

    def __init__(self, *, config=None):
        """Initialize visit-level connection parameters.

        :param config: Configuration for the connection
        :type config: `VisitInjectSLSimConfig`
        :return: None
        """
        config = cast(VisitInjectSLSimConfig, config)

        super().__init__(config=config)
        if (
            not config.external_psf
            and not config.external_photo_calib
            and not config.external_wcs
        ):
            self.inputs.remove("visit_summary")


class VisitInjectSLSimConfig(  # type: ignore [call-arg]
    BaseInjectSLSimConfig,
    pipelineConnections=VisitInjectSLSimConnections,
):
    """Visit-level configuration for strong lensing source injection tasks.

    This class extends BaseInjectSLSimConfig to provide visit-level configuration
    options for strong lensing source injection. For information on the LSST
    source injection framework this is built upon:
    https://github.com/lsst/source-injection
    """

    # Calibrated data options.
    external_psf = Field[bool](
        doc="If True, use the PSF model from a visit summary table. "
        "If False (default), use the PSF model attached to the input exposure.",
        dtype=bool,
        default=False,
    )
    external_photo_calib = Field[bool](
        doc="If True, use the photometric calibration from a visit summary table. "
        "If False (default), use the photometric calibration attached to the input exposure.",
        dtype=bool,
        default=False,
    )
    external_wcs = Field[bool](
        doc="If True, use the astrometric calibration from a visit summary table. "
        "If False (default), use the astrometric calibration attached to the input exposure.",
        dtype=bool,
        default=False,
    )


class VisitInjectSLSimTask(BaseInjectSLSimTask):
    """Visit-level class for injecting strong lensing sources into images.

    This task extends BaseInjectSLSimTask to operate at the visit level, handling
    visit-specific data structures and metadata. It provides functionality to
    inject strong lensing sources into visit-level exposures.

    For information on the LSST source injection framework this is built upon:
    https://github.com/lsst/source-injection
    """

    _DefaultName = "visitInjectSLSimTask"
    ConfigClass = VisitInjectSLSimConfig

    def runQuantum(self, butler_quantum_context, input_refs, output_refs):
        """Run the task on a quantum of data.

        :param butler_quantum_context: Butler quantum context
        :type butler_quantum_context: `lsst.daf.butler.QuantumContext`
        :param input_refs: Input dataset references
        :type input_refs: `dict`
        :param output_refs: Output dataset references
        :type output_refs: `dict`
        :return: None
        """
        inputs = butler_quantum_context.get(input_refs)
        detector_id = inputs["input_exposure"].getDetector().getId()

        try:
            visit_summary = inputs["visit_summary"].get()
        except KeyError:
            # Use internal PSF, PhotoCalib and WCS.
            inputs["psf"] = inputs["input_exposure"].getPsf()
            inputs["photo_calib"] = inputs["input_exposure"].getPhotoCalib()
            inputs["wcs"] = inputs["input_exposure"].getWcs()
        else:
            # Use external PSF, PhotoCalib and WCS.
            detector_summary = visit_summary.find(detector_id)
            if detector_summary:
                inputs["psf"] = detector_summary.getPsf()
                inputs["photo_calib"] = detector_summary.getPhotoCalib()
                inputs["wcs"] = detector_summary.getWcs()
            else:
                raise RuntimeError(
                    f"No record for detector {detector_id} found in visit summary table."
                )

        input_keys = [
            "injection_catalogs",
            "input_exposure",
            "sky_map",
            "psf",
            "photo_calib",
            "wcs",
        ]
        
        outputs = self.run(
            **{key: value for (key, value) in inputs.items() if key in input_keys}
        )
        butler_quantum_context.put(outputs, output_refs)
