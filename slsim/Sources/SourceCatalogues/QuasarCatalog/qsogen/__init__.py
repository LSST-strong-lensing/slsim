# The code in this directory is part of QSOGen, a quasar spectrum generator.
# QSOGen is used within SLSim to simulate quasar populations with realistic SEDs.
# Here is the link to their project page: https://github.com/MJTemple/qsogen/tree/main
# The paper describing QSOGen is available at (Temple et al. 2021, https://arxiv.org/abs/2109.04472)

from slsim.Sources.SourceCatalogues.QuasarCatalog.qsogen.config import params_agile
from slsim.Sources.SourceCatalogues.QuasarCatalog.qsogen.qsosed import Quasar_sed

__all__ = ["params_agile", "Quasar_sed"]
