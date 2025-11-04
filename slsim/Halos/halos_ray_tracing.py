from lenstronomy.Util.util import make_grid
import numpy as np
import math
from slsim.Util.param_util import deg2_to_cone_angle


class HalosRayTracing(object):
    """A class for performing ray tracing computations in gravitational lensing
    scenarios involving multiple halos.

    This class provides methods to compute various lensing quantities such as convergence (kappa), shear (gamma),
    and their non-linear corrections across different redshifts and sky areas.

    Attributes:
        ray_halo (list): A list to store halo-related data during computations.
        lens_kwargs (dict): Keyword arguments for the lens model that describe the lensing scenario.
        lens_model (LensModel): An instance of a lens model used for the ray tracing computations.

    Methods:
        get_convergence_shear: Computes the convergence and shear at the origin due to all Halos.
        nonlinear_correction_kappa_gamma_values: Computes various kappa and gamma values for given deflector and source redshifts.
        get_kext_gext_values: Computes the external convergence and shear for given deflector and source redshifts.
        various_halos_data: Computes various convergence and shear values for given deflector and source redshifts.
        compute_kappa: Computes the convergence values over a grid and returns both the 2D kappa image and the 1D array of kappa values.
        plot_convergence: Plots the convergence across the lensed sky area.
    """

    def __init__(self, lens_kwargs, lens_model):
        """Initializes the HalosRayTracing class with specified lens model and
        keyword arguments.

        :param lens_kwargs: Keyword arguments for the lens model.
        :type lens_kwargs: dict
        :param lens_model: An instance of a lens model.
        :type lens_model: LensModel
        """
        self.ray_halo = []
        self.lens_kwargs = lens_kwargs
        self.lens_model = lens_model

    def get_convergence_shear(
        self,
        kwargs=None,
        lens_model=None,
        same_from_class=True,
        gamma12=False,
        diff=1.0,
        diff_method="square",
        zdzs=None,
    ):
        """Computes the convergence and shear.

        :param gamma12: If True, returns gamma1 and gamma2 in addition to kappa. If False, returns total shear gamma along with kappa.
        :type gamma12: bool, optional
        :param same_from_class: If True and kwargs, lens_model is none uses the class's lens model and lens kwargs. If False, uses the provided lens model and kwargs. Specify for the when the 'None' type kwargs
        :type same_from_class: bool, optional
        :param diff: The differential used in the computation of the Hessian matrix. Default is 1.0.
        :type diff: float, optional
        :param diff_method: The method used to compute the differential. Default is "square".
        :type diff_method: str, optional
        :param kwargs: Keyword arguments for the lens model. If None, uses the class method to generate them.
        :type kwargs: dict, optional
        :param lens_model: The lens model instance to use. If None, uses the class's lens model.
        :type lens_model: LensModel, optional
        :param zdzs: A tuple of deflector and source redshifts (zd, zs). If provided, uses `hessian_z1z2` method of the lens model.
        :type zdzs: tuple, optional
        :returns: Depending on `gamma12`, either (kappa, gamma) or (kappa, gamma1, gamma2). Kappa is the convergence, gamma is the total shear, and gamma1 and gamma2 are the shear components.
        :rtype: tuple
        """
        if same_from_class:
            if lens_model is None:
                lens_model = self.lens_model
            if kwargs is None:
                kwargs = self.lens_kwargs
        if kwargs is None:
            if gamma12:
                return 0.0, 0.0, 0.0
            else:
                return 0.0, 0.0
        if zdzs is not None:
            f_xx, f_xy, f_yx, f_yy = lens_model.hessian_z1z2(
                z1=zdzs[0],
                z2=zdzs[1],
                theta_x=0,
                theta_y=0,
                kwargs_lens=kwargs,
                diff=diff,
            )
        else:
            f_xx, f_xy, f_yx, f_yy = lens_model.hessian(
                x=0.0, y=0.0, kwargs=kwargs, diff=diff, diff_method=diff_method
            )
        kappa = 0.5 * (f_xx + f_yy)
        if gamma12:
            gamma1 = 1.0 / 2 * (f_xx - f_yy)
            gamma2 = f_xy
            return kappa, gamma1, gamma2
        else:
            gamma = np.sqrt(f_xy**2 + 0.25 * (f_xx - f_yy) ** 2)
            return kappa, gamma

    def nonlinear_correction_kappa_gamma_values(self, lens_data, zd, zs):
        """Computes various kappa (convergence) and gamma (shear) values for
        given deflector and source redshifts.

        This function retrieves the lens data based on the input redshifts and computes the convergence
        and shear for three categories: `od`, `os`, and `ds`. The gamma values are computed for
        both components, gamma1 and gamma2.

        :param lens_data: A tuple containing lens data for three different conditions:
                  1. Between deflector and source redshift (ds).
                  2. From zero to deflector redshift (od).
                  3. From zero to source redshift (os).
        :type lens_data: dict
        :param zd: The deflector redshift.
        :type zd: float
        :param zs: The source redshift.
        :type zs: float
        :returns: A tuple containing the convergence and shear values for `od`, `os`, and `ds` categories.
        :rtype: (float, float, float, float, float, float, float, float, float)
        """

        # Obtain the lens data for each redshift using the get_lens_data_by_redshift function
        # Extracting lens model and lens_kwargs for 'od' and 'os'
        lens_model_od = lens_data["od"]["param_lens_model"]
        kwargs_lens_od = lens_data["od"]["kwargs_lens"]

        lens_model_os = lens_data["os"]["param_lens_model"]
        kwargs_lens_os = lens_data["os"]["kwargs_lens"]

        lens_model_ds = lens_data["ds"]["param_lens_model"]
        kwargs_lens_ds = lens_data["ds"]["kwargs_lens"]

        kappa_od, gamma_od1, gamma_od2 = self.get_convergence_shear(
            gamma12=True,
            kwargs=kwargs_lens_od,
            lens_model=lens_model_od,
            same_from_class=False,
        )

        kappa_os, gamma_os1, gamma_os2 = self.get_convergence_shear(
            gamma12=True,
            kwargs=kwargs_lens_os,
            lens_model=lens_model_os,
            same_from_class=False,
        )
        kappa_ds, gamma_ds1, gamma_ds2 = self.get_convergence_shear(
            gamma12=True,
            kwargs=kwargs_lens_ds,
            lens_model=lens_model_ds,
            zdzs=(zd, zs),
            same_from_class=False,
        )

        return (
            kappa_od,
            kappa_os,
            gamma_od1,
            gamma_od2,
            gamma_os1,
            gamma_os2,
            kappa_ds,
            gamma_ds1,
            gamma_ds2,
        )

    def get_kext_gext_values(self, lens_data, zd, zs):
        r"""Computes the external convergence (kappa_ext) and external shear
        (gamma_ext) for given deflector and source redshifts.

        :param lens_data: A dict containing lens data for three different conditions:
                  1. Between deflector and source redshift (ds).
                  2. From zero to deflector redshift (od).
                  3. From zero to source redshift (os).
        :type lens_data: dict
        :param zd: The deflector redshift.
        :type zd: float
        :param zs: The source redshift.
        :type zs: float
        :returns: A tuple containing the computed external convergence value (kext) and the computed external shear magnitude (gext).
        :rtype: (float, float)

        .. note::
            The function implements the following formula:

            .. math::
                1 - \kappa_{\text{ext}} = \frac{(1-\kappa_{\text{od}})(1-\kappa_{\text{os}})}{1-\kappa_{\text{ds}}}

            and

            .. math::
                \gamma_{\text{ext}} = \sqrt{(\gamma_{\text{od}1}+\gamma_{\text{os}1}-\gamma_{\text{ds}1})^2+(\gamma_{\text{od}2}+\gamma_{\text{os}2}-\gamma_{\text{ds}2})^2}
        """
        (
            kappa_od,
            kappa_os,
            gamma_od1,
            gamma_od2,
            gamma_os1,
            gamma_os2,
            kappa_ds,
            gamma_ds1,
            gamma_ds2,
        ) = self.nonlinear_correction_kappa_gamma_values(lens_data, zd, zs)

        kext = 1 - (1 - kappa_od) * (1 - kappa_os) / (1 - kappa_ds)
        gext = math.sqrt(
            (gamma_od1 + gamma_os1 - gamma_ds1) ** 2
            + (gamma_od2 + gamma_os2 - gamma_ds2) ** 2
        )

        return kext, gext

    def various_halos_data(self, lens_data, zd, zs):
        r"""Computes (kappa_od, kappa_os, gamma_od1, gamma_od2, gamma_os1,
        gamma_os2, kappa_ds, gamma_ds1, gamma_ds2, kappa_os2, gamma_os12,
        gamma_os22, kext, gext,), (kwargs_lens_os, lens_model_os)  convergence
        (kappa) and shear (gamma) values for given deflector and source
        redshifts.

        This function extracts the lens model and its keyword arguments for different redshift combinations
        ('od`, `os`, and `ds`). It then computes the convergence and shear values for each of these combinations.

        :param lens_data: A dict tuple containing lens data for three different conditions:
                  1. Between deflector and source redshift (ds).
                  2. From zero to deflector redshift (od).
                  3. From zero to source redshift (os).
        :type lens_data: dict
        :param zd: The deflector redshift.
        :type zd: float
        :param zs: The source redshift.
        :type zs: float
        :returns:
            A tuple containing:
                - A tuple of computed values for kappa and gamma for the different redshift combinations and the external convergence and shear.
                - A tuple containing the lens model and its keyword arguments for the `os` redshift combination.
        :rtype: tuple

        .. note::
            This function is utilized by the `self.get_all_pars_distib()` method. The mathematical formulations behind
            the calculations, especially for `kext` and `gext`, can be referenced from the documentation of
            `get_kext_gext_values`, is applied with the line of sight non-linear correction.

            The function implements the following formulae for the external convergence and shear with LOS correction:

            .. math::
                1 - \kappa_{\text{ext}} = \frac{(1-\kappa_{\text{od}})(1-\kappa_{\text{os}})}{1-\kappa_{\text{ds}}}

            and

            .. math::
                \gamma_{\text{ext}} = \sqrt{(\gamma_{\text{od}1}+\gamma_{\text{os}1}-\gamma_{\text{ds}1})^2+(\gamma_{\text{od}2}+\gamma_{\text{os}2}-\gamma_{\text{ds}2})^2}
        """

        # Obtain the lens data for each redshift using the get_lens_data_by_redshift function
        # Extracting lens model and lens_kwargs for 'od' and 'os'
        lens_model_od = lens_data["od"]["param_lens_model"]
        kwargs_lens_od = lens_data["od"]["kwargs_lens"]

        lens_model_os = lens_data["os"]["param_lens_model"]
        kwargs_lens_os = lens_data["os"]["kwargs_lens"]

        lens_model_ds = lens_data["ds"]["param_lens_model"]
        kwargs_lens_ds = lens_data["ds"]["kwargs_lens"]

        kappa_od, gamma_od1, gamma_od2 = self.get_convergence_shear(
            gamma12=True,
            kwargs=kwargs_lens_od,
            lens_model=lens_model_od,
            same_from_class=False,
        )

        kappa_os, gamma_os1, gamma_os2 = self.get_convergence_shear(
            gamma12=True,
            kwargs=kwargs_lens_os,
            lens_model=lens_model_os,
            same_from_class=False,
        )

        kappa_os2, gamma_os12, gamma_os22 = self.get_convergence_shear(
            gamma12=True,
            kwargs=kwargs_lens_os,
            lens_model=lens_model_os,
            zdzs=(0, zs),
            same_from_class=False,
        )

        kappa_ds, gamma_ds1, gamma_ds2 = self.get_convergence_shear(
            gamma12=True,
            kwargs=kwargs_lens_ds,
            lens_model=lens_model_ds,
            zdzs=(zd, zs),
            same_from_class=False,
        )

        kext = 1 - (1 - kappa_od) * (1 - kappa_os) / (1 - kappa_ds)
        gext = math.sqrt(
            (gamma_od1 + gamma_os1 - gamma_ds1) ** 2
            + (gamma_od2 + gamma_os2 - gamma_ds2) ** 2
        )

        results_dict = {
            "kappa_od": kappa_od,
            "kappa_os": kappa_os,
            "gamma_od1": gamma_od1,
            "gamma_od2": gamma_od2,
            "gamma_os1": gamma_os1,
            "gamma_os2": gamma_os2,
            "kappa_ds": kappa_ds,
            "gamma_ds1": gamma_ds1,
            "gamma_ds2": gamma_ds2,
            "kappa_os2": kappa_os2,
            "gamma_os12": gamma_os12,
            "gamma_os22": gamma_os22,
            "kext": kext,
            "gext": gext,
        }

        lens_model_data = {
            "kwargs_lens_os": kwargs_lens_os,
            "lens_model_os": lens_model_os,
        }

        return results_dict, lens_model_data

    def compute_kappa(
        self,
        sky_area,
        diff=0.0000001,
        num_points=500,
        diff_method="square",
        kwargs=None,
        lens_model=None,
    ):
        # can out as single
        """Computes the convergence (kappa) values over a grid and returns both
        the 2D kappa image and the 1D array of kappa values.

        :param sky_area: Total sky area in steradians over which halos are distributed. Defaults to full sky (4π steradians).
        :type sky_area: float, optional
        :param diff: The differential used in the computation of the convergence. Defaults to 0.0000001.
        :type diff: float, optional
        :param num_points: The number of points along each axis of the grid. Defaults to 500.
        :type num_points: int, optional
        :param diff_method: The method used to compute the differential. Defaults to "square".
        :type diff_method: str, optional
        :param kwargs: The keyword arguments for the lens model. If None, it will use the class method `get_halos_lens_kwargs` to generate them. Defaults to None.
        :type kwargs: dict, optional
        :param lens_model: The lens model to use. If None, it will use the class attribute `param_lens_model`. Defaults to None.
        :type lens_model: LensModel, optional
        :return: A tuple containing the 2D kappa image and the 1D array of kappa values.
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """

        if kwargs is None:
            kwargs = self.lens_kwargs
        if lens_model is None:
            lens_model = self.lens_model

        radius_arcsec = deg2_to_cone_angle(sky_area) * 206264.806
        x = np.linspace(-radius_arcsec, radius_arcsec, num_points)
        y = np.linspace(-radius_arcsec, radius_arcsec, num_points)
        X, Y = np.meshgrid(x, y)
        mask_2D = X**2 + Y**2 <= radius_arcsec**2
        mask_1D = mask_2D.ravel()

        # Use lenstronomy utility to make grid
        x_grid, y_grid = make_grid(
            numPix=num_points, deltapix=2 * radius_arcsec / num_points
        )
        x_grid, y_grid = x_grid[mask_1D], y_grid[mask_1D]

        # Calculate the kappa values
        kappa_values = lens_model.kappa(
            x_grid, y_grid, kwargs, diff=diff, diff_method=diff_method
        )
        kappa_image = np.ones((num_points, num_points)) * np.nan
        kappa_image[mask_2D] = kappa_values
        return kappa_image, kappa_values

    def plot_convergence(
        self,
        sky_area,
        diff=0.0000001,
        num_points=500,
        diff_method="square",
        kwargs=None,
        lens_model=None,
    ):
        # can out
        r"""Plots the convegence (:math:`\kappa`) across the lensed sky area.

        :param sky_area: Total sky area in steradians. Defaults to full sky (4π steradians).
        :type sky_area: float, optional
        :param diff: The differentiation value used for computing the hessian. Default is 1e-7.
        :type diff: float, optional
        :param num_points: Number of points along each axis for which convergence is computed. Default is 500.
        :type num_points: int, optional
        :param diff_method: The method to use for differentiation when computing the hessian. Default is "square".
        :type diff_method: str, optional
        :param kwargs: Keyword arguments for the lens model. If not provided, the halos lens lens_kwargs of the instance are used.
        :type kwargs: dict, optional
        :param lens_model: The lens model to use. If not provided, the lens model from the class instance is utilized.
        :type lens_model: LensModel instance, optional
        :return: None. The function will display a plot of the computed convergence with plot.
        :rtype: None

        .. note::
            The function computes the convergence for a grid defined by `num_points` and plots the result using matplotlib.
            The computed sky area is determined by the instance's sky area, converted from square degrees to arcseconds.
            Overlaying on the convergence plot are positions of halos represented by yellow `x` markers.
        """

        import matplotlib.pyplot as plt

        radius_arcsec = deg2_to_cone_angle(sky_area) * 206264.806

        if kwargs is None:
            kwargs = self.lens_kwargs
        if lens_model is None:
            lens_model = self.lens_model

        kappa_image, _ = self.compute_kappa(
            sky_area=sky_area,
            diff=diff,
            num_points=num_points,
            diff_method=diff_method,
            kwargs=kwargs,
            lens_model=lens_model,
        )

        plt.imshow(
            kappa_image,
            extent=[-radius_arcsec, radius_arcsec, -radius_arcsec, radius_arcsec],
        )
        plt.colorbar(label=r"$\kappa$")

        halos_x = [k.get("center_x", None) for k in kwargs]
        halos_y = [
            -k.get("center_y") if k.get("center_y") is not None else None
            for k in kwargs
        ]
        plt.scatter(halos_x, halos_y, color="yellow", marker="x", label="Halos")
        plt.title(f"Convergence Plot, radius is {radius_arcsec} arcsec")
        plt.xlabel("x-coordinate (arcsec)")
        plt.ylabel("y-coordinate (arcsec)")
        plt.legend()
        plt.show()
