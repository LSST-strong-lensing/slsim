import speclite.filters
import os
import slsim
from slsim.Observations.roman_speclite import (
    configure_roman_filters,
    filter_names,
)


def test_roman_speclite():
    configure_roman_filters()

    path = os.path.dirname(slsim.__file__)
    module_path, _ = os.path.split(path)

    save_path = os.path.join(module_path, "data/Filters/Roman/")

    filter_name_list = filter_names()
    # print(filter_name_list[0], 'test filter_name_list')
    speclite.filters.load_filters(filter_name_list[0], filter_name_list[1])
    speclite.filters.load_filters(
        save_path + "Roman-F062.ecsv",
        save_path + "Roman-F087.ecsv",
        save_path + "Roman-F106.ecsv",
    )

    speclite.filters.load_filters("Roman-F062", "Roman-F087", "Roman-F106")
    os.remove(save_path + "Roman-F062.ecsv")
    os.remove(save_path + "Roman-F087.ecsv")
    os.remove(save_path + "Roman-F106.ecsv")
