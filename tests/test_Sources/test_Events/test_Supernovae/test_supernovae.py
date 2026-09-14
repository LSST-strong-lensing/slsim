from slsim.Sources.Events.Supernovae.supernova import Supernova
import numpy as np
import numpy.testing as npt
import pytest


@pytest.fixture
def Supernova_class():
    SN = Supernova(
        source="salt3-nir",
        redshift=1.2,
        sn_type="Ia",
        absolute_mag=-19.3,
        absolute_mag_band="bessellb",
        mag_zpsys="AB",
    )

    return SN


def test_supernova_mag(Supernova_class):
    mag = Supernova_class.get_apparent_magnitude(time=0, band="lsstr")
    assert mag > 0

    npt.assert_warns(
        UserWarning, Supernova_class.get_apparent_magnitude, time=0, band="lsstg"
    )


def test_supernova_mag_outside_spectral_range(Supernova_class):
    # salt3-nir covers rest frame 2000 to 20000 Angstrom, so at z = 1.2 the u
    # band falls blueward of the model and no flux can be assigned.
    with pytest.warns(UserWarning, match="no flux assigned"):
        mag = Supernova_class.get_apparent_magnitude(
            time=np.linspace(-20, 50, 30), band="lsstu"
        )
    assert np.isinf(mag).all()
    assert not np.isnan(mag).any()


def test_supernova_mag_never_nan(Supernova_class):
    # The time range deliberately extends well past the phase coverage of the
    # model, where the supernova has no flux at all.
    mag = Supernova_class.get_apparent_magnitude(
        time=np.linspace(-100, 300, 200), band="lsstr"
    )
    assert not np.isnan(mag).any()
    assert np.isfinite(mag).any()


def test_supernova_amplitude_arguments(capsys):
    kwargs = dict(source="salt3-nir", redshift=1.2, sn_type="Ia")

    # An absolute magnitude cannot be normalized without the band it refers to.
    with pytest.raises(Exception, match="Must set absolute_mag_band"):
        Supernova(absolute_mag=-19.3, **kwargs)

    # Given both normalizations, the absolute magnitude is the one applied.
    Supernova(
        absolute_mag=-19.3,
        absolute_mag_band="bessellb",
        peak_apparent_mag=22.0,
        peak_apparent_mag_band="lsstr",
        **kwargs
    )
    assert "choosing absolute_mag" in capsys.readouterr().out

    # Given only an apparent magnitude, the source is normalized to it.
    peak = Supernova(peak_apparent_mag=22.0, peak_apparent_mag_band="lsstr", **kwargs)
    npt.assert_almost_equal(peak.source_peakmag("lsstr", "AB"), 22.0, decimal=3)

    # Given neither, the amplitude is left for the user to set.
    with pytest.warns(UserWarning, match="set the amplitude"):
        Supernova(**kwargs)


if __name__ == "__main__":
    pytest.main()
