import speclite.filters
import os
from slsim.Pipelines.roman_speclite import (
    configure_roman_filters,
    filter_names,
)
from slsim.Util.data_path import data_path


def test_roman_speclite():
    configure_roman_filters()

    save_path = data_path("Filters", "Roman")

    filter_name_list = filter_names()
    # print(filter_name_list[0], 'test filter_name_list')
    speclite.filters.load_filters(filter_name_list[0], filter_name_list[1])
    speclite.filters.load_filters(
        os.path.join(save_path, "Roman-F062.ecsv"),
        os.path.join(save_path, "Roman-F087.ecsv"),
        os.path.join(save_path, "Roman-F106.ecsv"),
    )

    speclite.filters.load_filters("Roman-F062", "Roman-F087", "Roman-F106")
    os.remove(os.path.join(save_path, "Roman-F062.ecsv"))
    os.remove(os.path.join(save_path, "Roman-F087.ecsv"))
    os.remove(os.path.join(save_path, "Roman-F106.ecsv"))
