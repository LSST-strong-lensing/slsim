from sim_pipeline.Sources.source_variability.variability import sinusoidal_variability


def test_sinusoidal_variability():
    variability = sinusoidal_variability(0)
    assert variability == 0
