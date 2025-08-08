import h5py
import numpy as np
import os
import warnings


class LineOfSightDistribution:
    """Class to read the joint and no nonlinear distributions from the H5
    files.

    From the H5 files, the class can retrieve kappa and gamma values
    from the H5 files based on the source and lens redshifts for the
    resample of external convergence and shear.
    """

    correction_data = None
    no_nonlinear_correction_data = None

    def __init__(self, nonlinear_correction_path=None, no_correction_path=None):
        """Initialize the Data Reader. Load data into class variables if not
        already loaded.

        :param nonlinear_correction_path: Path to the
            'joint_distributions.h5' file.
        :param no_correction_path: Path to the
            'kg_distributions_nolos.h5' file.
        """
        current_script_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_script_path)
        parent_directory = os.path.dirname(current_directory)
        parent_parent = os.path.dirname(os.path.dirname(parent_directory))

        if nonlinear_correction_path is None:
            file_path_joint = os.path.join(
                parent_parent, "data/glass/joint_distributions.h5"
            )
            if os.path.exists(file_path_joint):
                nonlinear_correction_path = file_path_joint
            else:
                nonlinear_correction_path = None

        if no_correction_path is None:
            file_path_no_nonlinear = os.path.join(
                parent_parent, "data/glass/no_nonlinear_distributions.h5"
            )
            if os.path.exists(file_path_no_nonlinear):
                no_correction_path = file_path_no_nonlinear
            else:
                no_correction_path = None

        if (
            LineOfSightDistribution.correction_data is None
            and nonlinear_correction_path is not None
        ):
            LineOfSightDistribution.correction_data = self._load_data(
                nonlinear_correction_path
            )
        elif nonlinear_correction_path is None:
            LineOfSightDistribution.correction_data = None

        if (
            LineOfSightDistribution.no_nonlinear_correction_data is None
            and no_correction_path is not None
        ):
            LineOfSightDistribution.no_nonlinear_correction_data = self._load_data(
                no_correction_path
            )
        elif no_correction_path is None:
            LineOfSightDistribution.no_nonlinear_correction_data = None

    @staticmethod
    def _load_data(file_path):
        """Load data from an H5 file into memory.

        :param file_path: Path to the H5 file.
        :return: Dictionary of datasets.
        """
        data = {}
        with h5py.File(file_path, "r") as h5_file:
            for dataset_name in h5_file:
                data[dataset_name] = h5_file[dataset_name][()]
        return data

    @staticmethod
    def _round_to_nearest_0_1(value):
        """Round the value to the nearest 0.1.

        :param value: A floating point number.
        :return: Rounded number to the nearest 0.1.
        """
        if value > 4.9:
            return 4.9
        if value < 0.1:
            return 0.1
        return round(value * 10) / 10

    def get_kappa_gamma(self, z_source, z_lens, use_nonlinear_correction=False):
        """Retrieve kappa and gamma values from the loaded data based on source
        and lens redshifts.

        :param z_source: Source redshift (zs).
        :param z_lens: Lens redshift (zd).
        :param use_nonlinear_correction: Boolean to use the nonlinear
            correction data.
        :return: Tuple of gamma and kappa values.
        """
        if z_source <= z_lens:
            return 0.0, 0.0
        z_source_rounded = self._round_to_nearest_0_1(z_source)
        z_lens_rounded = self._round_to_nearest_0_1(z_lens)
        if z_lens_rounded == z_source_rounded:
            z_lens_rounded = z_lens_rounded - 0.1
            z_lens_rounded = round(z_lens_rounded, 1)
            # todo： make z_lens_rounded == z_source_rounded worked in file
        dataset_name = (
            f"zs_{z_source_rounded}_zd_{z_lens_rounded}"
            if (use_nonlinear_correction)
            else f"zs_{z_source_rounded}"
        )
        data = (
            LineOfSightDistribution.correction_data
            if use_nonlinear_correction
            else LineOfSightDistribution.no_nonlinear_correction_data
        )
        if data is None:
            warnings.warn("No file found, provide 0 instead.")
            return 0.0, 0.0
        if dataset_name in data:
            gamma = np.random.choice(data[dataset_name][:, 1])
            kappa = np.random.choice(data[dataset_name][:, 0])
            return gamma, kappa
        else:
            raise ValueError(
                f"No data found for zs={z_source_rounded} and zd={z_lens_rounded}."
            )
