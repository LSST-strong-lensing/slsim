## QSOGen

The code in this directory was sourced from QSOGen, a quasar spectrum generator. QSOGen is used within SLSim to simulate quasar populations with realistic SEDs. Here is the link to their project page: https://github.com/MJTemple/qsogen/tree/main. The paper describing QSOGen is available at (Temple et al. 2021, https://arxiv.org/abs/2109.04472).

Some assumptions/information about the Quasar SEDs currently being used in SLSim:

* Emission line templates span rest-frame **980–30,000 Å**, extending down to the far-UV.
* Default AGILE parameters (Viitanen et al. 2026) are calibrated only to **$z \approx 5$**. Extrapolation to extreme redshifts ($z \sim 12$) should be used with caution.
* At high redshift ($z \sim 12$), the **912 Å Lyman-Limit cutoff** shifts into observing bands, which may result in zero emission for many standard filters.