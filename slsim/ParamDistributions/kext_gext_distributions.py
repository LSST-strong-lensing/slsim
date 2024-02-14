import h5py
import numpy as np

class LineOfSightDistribution:
    correction_data = None
    no_nonlinear_correction_data = None

    def __init__(self, nonlinear_correction_path=None, no_nonlinear_correcction=None):
        """
        Initialize the Data Reader. Load data into class variables if not already loaded.

        :param nonlinear_correction_path: Path to the 'joint_distributions.h5' file.
        :param no_nonlinear_correcction: Path to the 'kg_distributions_nolos.h5' file.
        """
        if nonlinear_correction_path is None:
            nonlinear_correction_path = '/Users/tz/Documents/GitHub/slsim/data/glass/joint_distributions.h5'
        if no_nonlinear_correcction is None:
            no_nonlinear_correcction = '/Users/tz/Documents/GitHub/slsim/data/glass/no_nonlinear_distributions.h5'
        if LineOfSightDistribution.correction_data is None and nonlinear_correction_path is not None:
            LineOfSightDistribution.correction_data = self._load_data(nonlinear_correction_path)

        if LineOfSightDistribution.no_nonlinear_correction_data is None and no_nonlinear_correcction is not None:
            LineOfSightDistribution.no_nonlinear_correction_data = self._load_data(no_nonlinear_correcction)

    @staticmethod
    def _load_data(file_path):
        """
        Load data from an H5 file into memory.

        :param file_path: Path to the H5 file.
        :return: Dictionary of datasets.
        """
        data = {}
        with h5py.File(file_path, 'r') as h5_file:
            for dataset_name in h5_file:
                data[dataset_name] = h5_file[dataset_name][()]
        return data

    @staticmethod
    def _round_to_nearest_0_1(value):
        """
        Round the value to the nearest 0.1.

        :param value: A floating point number.
        :return: Rounded number to the nearest 0.1.
        """
        if value>4.9:
            return 4.9
        if value<0.1:
            return 0.1
        return round(value * 10) / 10

    def get_kappa_gamma(self, z_source, z_lens, use_kg_nolos=False):
        """
        Retrieve kappa and gamma values from the loaded data based on source and lens redshifts.

        :param z_source: Source redshift (zs).
        :param z_lens: Lens redshift (zd).
        :param use_kg_nolos: Boolean to use 'kg_distributions_nolos.h5' file.
        :return: Tuple of gamma and kappa values.
        """

        z_source_rounded = self._round_to_nearest_0_1(z_source)
        z_lens_rounded = self._round_to_nearest_0_1(z_lens)
        if z_lens_rounded == z_source_rounded:
            z_lens_rounded = z_lens_rounded - 0.1
            z_lens_rounded = round(z_lens_rounded, 1)
            #todo： make z_lens_rounded == z_source_rounded worked in file
        dataset_name = f'zs_{z_source_rounded}' if use_kg_nolos else f'zs_{z_source_rounded}_zd_{z_lens_rounded}'
        data = LineOfSightDistribution.no_nonlinear_correction_data if use_kg_nolos else LineOfSightDistribution.correction_data
        if dataset_name in data:
            gamma = np.random.choice(data[dataset_name][:, 1])
            kappa = np.random.choice(data[dataset_name][:, 0])
            return gamma, kappa
        else:
            raise ValueError(f'No data found for zs={z_source_rounded} and zd={z_lens_rounded}.')