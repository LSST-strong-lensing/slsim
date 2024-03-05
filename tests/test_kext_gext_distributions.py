from slsim.ParamDistributions.kext_gext_distributions import LineOfSightDistribution

def test_initializes_with_default_data_paths():
    line_of_sight = LineOfSightDistribution()
    assert line_of_sight.no_nonlinear_correction_data is not None

def test_round_to_nearest_0_1_multiple_decimal_places():
        los_distribution = LineOfSightDistribution()
        result = los_distribution._round_to_nearest_0_1(2.345)
        assert result == 2.3
        result2 = los_distribution._round_to_nearest_0_1(5.6)
        assert result2 == 4.9
        result3 = los_distribution._round_to_nearest_0_1(0.05)
        assert result3 == 0.1

def test_valid_inputs():
    line_of_sight = LineOfSightDistribution()
    result = line_of_sight.get_kappa_gamma(0.5, 0.3)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)