from lenstronomy.Util.coolest_interface import (
    update_coolest_from_lenstronomy,
    create_lenstronomy_from_coolest,
)
from slsim.Util.param_util import magnitude_to_amplitude, amplitude_to_magnitude


def update_coolest_from_slsim(
    lens_class, path, file_name, band=None, mag_zero_point=27, ending="_update"
):
    """This function updates given coolest format .json file using lenstronomy kwargs of
    Lens class. This function needs a .json file of coolest format. So, to generate
    required template file please use a notebook given in our notebooks folder.

    :param lens_class: Lens() object. This contains all the lensing parameters.
    :param path: path to the .json file that need to be updated
    :param file_name: name of the .json file without .json extension
    :param band: imaging band
    :param mag_zero_point: magnitude zero point for the exposure
    :param ending: extention to the original .json file that distinguish between
        original coolest file and updated coolest file.
    :returns: saves updated .json file in a given path
    """
    kwargs_result_slsim = lens_class.lenstronomy_kwargs(band=band)[1]

    # Iterate over sources to extract magnitudes and convert to amplitudes
    source_amps = []
    for source in kwargs_result_slsim["kwargs_source"]:
        source_amp = magnitude_to_amplitude(
            source["magnitude"], mag_zero_point=mag_zero_point
        )
        source_amps.append(source_amp)

    # Convert magnitudes to amplitudes for lens light and point source
    lens_amp = magnitude_to_amplitude(
        kwargs_result_slsim["kwargs_lens_light"][0]["magnitude"],
        mag_zero_point=mag_zero_point,
    )
    ps_amp = magnitude_to_amplitude(
        kwargs_result_slsim["kwargs_ps"][0]["magnitude"],
        mag_zero_point=mag_zero_point,
    )

    # Replace magnitudes with amplitudes in lenstronomy kwargs
    replacement_mappings = {
        "kwargs_source": {"magnitude": "amp", "value": source_amps},
        "kwargs_lens_light": {"magnitude": "amp", "value": lens_amp},
        "kwargs_ps": {"magnitude": "point_amp", "value": ps_amp},
    }

    for key, replacement_info in replacement_mappings.items():
        if key == "kwargs_source":
            for index, item in enumerate(kwargs_result_slsim[key]):
                item[replacement_info["magnitude"]] = replacement_info["value"][index]
                del item["magnitude"]
        else:
            for item in kwargs_result_slsim[key]:
                item[replacement_info["magnitude"]] = replacement_info["value"]
                del item["magnitude"]

    update_coolest = update_coolest_from_lenstronomy(
        path + file_name, kwargs_result=kwargs_result_slsim, ending=ending
    )
    return update_coolest


def create_slsim_from_coolest(path, file_name, mag_zero_point=27):
    """This function creates an lenstronomy_kwargs from the coolest file.

    :param path: path to the .json file that need to be updated
    :param file_name: name of the .json file without .json extension
    :return: dictionary of lenstronomy_kwargs for slsim. It contains lens_light_model,
        lens_mass_model, source_light_model, point_source_model.
    """
    kwargs_out = create_lenstronomy_from_coolest(path + file_name)
    kwargs_result = kwargs_out["kwargs_result"]
    source_mags = []
    for source in kwargs_result["kwargs_source"]:
        source_mag = amplitude_to_magnitude(
            source["amp"], mag_zero_point=mag_zero_point
        )
        source_mags.append(source_mag)
    lens_mag = amplitude_to_magnitude(
        kwargs_result["kwargs_lens_light"][0]["amp"],
        mag_zero_point=mag_zero_point,
    )
    ps_mag = amplitude_to_magnitude(
        kwargs_result["kwargs_ps"][0]["point_amp"],
        mag_zero_point=mag_zero_point,
    )
    # replace amplitudes with magnitudes in lenstronomy kwargs.
    replacement_mappings = {
        "kwargs_source": {"amp": "magnitude", "value": source_mags},
        "kwargs_lens_light": {"amp": "magnitude", "value": lens_mag},
        "kwargs_ps": {"point_amp": "magnitude", "value": ps_mag},
    }
    for key, replacement_info in replacement_mappings.items():
        for keys, values in replacement_info.items():
            if values == "magnitude":
                key2 = keys
            if key == "kwargs_source":
                for index, item in enumerate(kwargs_result[key]):
                    item[replacement_info[key2]] = replacement_info["value"][index]
                    if key2 in item:
                        del item[key2]
            else:
                for item in kwargs_result[key]:
                    item[replacement_info[key2]] = replacement_info["value"]
                    if key2 in item:
                        del item[key2]
    lenstronomy_slsim = kwargs_out["kwargs_model"], kwargs_result
    return lenstronomy_slsim
