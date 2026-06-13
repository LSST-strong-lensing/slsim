import numpy as np
import matplotlib.pyplot as plt

from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle


class SLSimLogo(object):
    """Generates the SLSim logo.

    :param rubin_color: color of the logo text and marker dots
    :param arc_color: color of the strong-lensing arcs
    :param galaxy_color: color of the central galaxy
    :param symbol_scale: overall scaling applied to the lensing symbol
    :type symbol_scale: float
    :param symbol_rotation: rotation angle of the lensing symbol in
        degrees
    :type symbol_rotation: float
    :param flip_x: mirror the lensing symbol about the y-axis
    :type flip_x: bool
    :param fontsize: font size of the logo text
    :type fontsize: float
    :param galaxy_ellipticity: ellipticity of the source galaxy
    :type galaxy_ellipticity: float
    :param galaxy_angle: major-axis position angle of the galaxy in
        degrees
    :type galaxy_angle: float
    :param galaxy_seed: random seed used for galaxy realization
    :type galaxy_seed: int
    """

    def __init__(
        self,
        rubin_color="#58B4B8",
        arc_color="#00D9FF",
        galaxy_color="#4B2E83",
        symbol_scale=1.1,
        symbol_rotation=-30,
        flip_x=False,
        fontsize=170,
        galaxy_ellipticity=0.4,
        galaxy_angle=150,
        galaxy_seed=3,
    ):

        self.rubin_color = rubin_color
        self.arc_color = arc_color
        self.galaxy_color = galaxy_color

        self.symbol_scale = symbol_scale
        self.symbol_rotation = symbol_rotation
        self.flip_x = flip_x

        self.fontsize = fontsize

        self.galaxy_ellipticity = galaxy_ellipticity
        self.galaxy_angle = galaxy_angle
        self.galaxy_seed = galaxy_seed

    def _transform(
        self,
        x,
        y,
        scale=1.0,
        rotation_deg=0.0,
        flip_x=False,
    ):
        """Apply scaling, reflection, and rotation.

        :param x: x coordinates
        :param y: y coordinates
        :param scale: scaling factor
        :type scale: float
        :param rotation_deg: rotation angle in degrees
        :type rotation_deg: float
        :param flip_x: mirror coordinates about the y-axis
        :type flip_x: bool
        :return: transformed coordinates
        """

        x = np.asarray(x) * scale
        y = np.asarray(y) * scale

        if flip_x:
            x = -x

        theta = np.deg2rad(rotation_deg)

        xr = x * np.cos(theta) - y * np.sin(theta)
        yr = x * np.sin(theta) + y * np.cos(theta)

        return xr, yr

    def _rotate_about_point(
        self,
        x,
        y,
        center,
        angle_deg,
    ):
        """Rotate coordinates about a pivot point.

        :param x: x coordinates
        :param y: y coordinates
        :param center: rotation center [x0, y0]
        :param angle_deg: rotation angle in degrees
        :type angle_deg: float
        :return: rotated coordinates
        """

        theta = np.deg2rad(angle_deg)

        xc = x - center[0]
        yc = y - center[1]

        xr = xc * np.cos(theta) - yc * np.sin(theta)
        yr = xc * np.sin(theta) + yc * np.cos(theta)

        return xr + center[0], yr + center[1]

    def _bezier_curve(self, P0, P1, P2, P3, n=800):
        """Generate a cubic Bezier curve.

        :param P0: first control point
        :param P1: second control point
        :param P2: third control point
        :param P3: fourth control point
        :param n: number of sampled points
        :type n: int
        :return: x and y coordinates of the curve
        """

        t = np.linspace(0, 1, n)[:, None]

        B = (
            ((1 - t) ** 3) * P0
            + 3 * ((1 - t) ** 2) * t * P1
            + 3 * (1 - t) * (t**2) * P2
            + (t**3) * P3
        )

        return B[:, 0], B[:, 1]

    def _draw_curve_in_S(self, ax, x, y, max_width, color):
        """Draw a curve in S.

        :param ax: matplotlib axes
        :param x: x coordinates of the curve
        :param y: y coordinates of the curve
        :param max_width: maximum ribbon width
        :type max_width: float
        :param color: curve color
        :return: curve in S
        """

        s = np.linspace(0, np.pi, len(x))
        width = max_width * np.sin(s) ** 2.5

        dx = np.gradient(x)
        dy = np.gradient(y)

        norm = np.sqrt(dx**2 + dy**2)

        nx = -dy / norm
        ny = dx / norm

        xu = x + 0.5 * width * nx
        yu = y + 0.5 * width * ny

        xl = x - 0.5 * width * nx
        yl = y - 0.5 * width * ny

        verts = np.vstack(
            [
                np.column_stack([xu, yu]),
                np.column_stack([xl[::-1], yl[::-1]]),
            ]
        )

        patch = PathPatch(
            Path(verts, closed=True),
            facecolor=color,
            edgecolor="none",
        )

        ax.add_patch(patch)

    def _draw_fake_galaxy(
        self,
        ax,
        x0,
        y0,
        scale,
    ):
        """Draw a stylized elliptical galaxy.

        :param ax: matplotlib axes
        :param x0: x coordinate of galaxy center
        :type x0: float
        :param y0: y coordinate of galaxy center
        :type y0: float
        :param scale: overall galaxy size
        :type scale: float
        """

        rng = np.random.default_rng(self.galaxy_seed)

        q = 1 - self.galaxy_ellipticity

        theta = np.linspace(0, 2 * np.pi, 220)

        cos_a = np.cos(np.deg2rad(self.galaxy_angle))
        sin_a = np.sin(np.deg2rad(self.galaxy_angle))

        n_shells = 10

        for i in range(n_shells):

            r = (0.02 + i * 0.02) * scale

            x = r * np.cos(theta)
            y = r * np.sin(theta) * q

            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a

            alpha = np.exp(-i / 2.2)

            ax.fill(
                x0 + xr,
                y0 + yr,
                color=self.galaxy_color,
                alpha=alpha,
                edgecolor="none",
            )

        ax.add_patch(
            Circle(
                (x0, y0),
                0.045 * scale,
                color=self.galaxy_color,
                alpha=0.95,
            )
        )

        ax.add_patch(
            Circle(
                (x0 + 0.015 * scale, y0 - 0.01 * scale),
                0.02 * scale,
                color=self.galaxy_color,
                alpha=0.6,
            )
        )

        nstars = 60

        xs = rng.normal(0, 0.04 * scale, nstars)
        ys = rng.normal(0, 0.04 * scale, nstars)

        xsr = xs * cos_a - ys * sin_a
        ysr = xs * sin_a + ys * cos_a

        ax.scatter(
            x0 + xsr,
            y0 + ysr,
            s=2,
            color="white",
            alpha=0.12,
        )

    def create(
        self,
        output_file=None,
        figsize=(14, 6),
        save=False,
        show=True,
    ):
        """Create the SLSim logo.

        :param output_file: output filename
        :param figsize: figure size for the matplotlib plot
        :param save: save figure to disk
        :type save: bool
        :param show: display figure
        :type show: bool
        :return: matplotlib figure and axes
        """

        fig, ax = plt.subplots(figsize=figsize)
        # Text
        ax.text(
            -1.9,
            0,
            "SL",
            fontsize=self.fontsize,
            fontweight="bold",
            color=self.rubin_color,
            ha="center",
            va="center",
        )

        ax.text(
            2.2,
            0,
            "im",
            fontsize=self.fontsize,
            fontweight="bold",
            color=self.rubin_color,
            ha="center",
            va="center",
        )

        upper_dot = np.array([-0.5, -0.06])
        lower_dot = np.array([0.5, 0.06])
        # upper arcs
        P0 = np.array([1, 0.6])
        P1 = np.array([-0.01, 1.50])
        P2 = np.array([-1.05, 0.95])
        P3 = np.array([0.4, -0.3])

        x, y = self._bezier_curve(P0, P1, P2, P3)

        x, y = self._rotate_about_point(
            x,
            y,
            center=P3,
            angle_deg=40,
        )

        x, y = self._transform(
            x,
            y,
            scale=self.symbol_scale,
            rotation_deg=self.symbol_rotation,
            flip_x=self.flip_x,
        )

        self._draw_curve_in_S(
            ax,
            x,
            y,
            0.18 * self.symbol_scale,
            self.arc_color,
        )
        # Lower arc
        P0 = np.array([-1, -0.6])
        P1 = np.array([0.01, -1.50])
        P2 = np.array([1.05, -0.95])
        P3 = np.array([-0.4, 0.3])

        x, y = self._bezier_curve(P0, P1, P2, P3)

        x, y = self._rotate_about_point(
            x,
            y,
            center=P3,
            angle_deg=40,
        )

        x, y = self._transform(
            x,
            y,
            scale=self.symbol_scale,
            rotation_deg=self.symbol_rotation,
            flip_x=self.flip_x,
        )

        self._draw_curve_in_S(
            ax,
            x,
            y,
            0.22 * self.symbol_scale,
            self.arc_color,
        )
        # Galaxy
        gx, gy = self._transform(
            [0],
            [0],
            scale=self.symbol_scale,
            rotation_deg=self.symbol_rotation,
            flip_x=self.flip_x,
        )

        self._draw_fake_galaxy(
            ax,
            gx[0],
            gy[0],
            self.symbol_scale * 5,
        )
        # Dots
        dots = np.array([upper_dot, lower_dot])

        dx, dy = self._transform(
            dots[:, 0],
            dots[:, 1],
            scale=self.symbol_scale,
            rotation_deg=self.symbol_rotation,
            flip_x=self.flip_x,
        )

        for x0, y0 in zip(dx, dy):

            ax.add_patch(
                Circle(
                    (x0, y0),
                    0.055 * self.symbol_scale,
                    color=self.rubin_color,
                )
            )
        # Layout
        ax.set_xlim(-5.5, 5.5)
        ax.set_ylim(-2.5, 2.5)

        ax.set_aspect("equal")
        ax.axis("off")

        plt.tight_layout()

        if save and output_file is not None:

            plt.savefig(
                output_file,
                dpi=600,
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.02,
            )

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax
