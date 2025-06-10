__author__ = "Paras Sharma"

from slsim.Microlensing.source_morphology.source_morphology import (
    SourceMorphology,
)


class SupernovaeSourceMorphology(SourceMorphology):
    """Class for Supernovae source morphology."""

    def __init__(self, *args, **kwargs):
        """Initializes the Supernovae source morphology."""
        super().__init__(*args, **kwargs)
        raise NotImplementedError(
            "Supernovae source morphology is not implemented yet."
        )
