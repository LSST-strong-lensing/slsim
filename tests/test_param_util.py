from sim_pipeline.Util.param_util import epsilon2e, e2epsilon, random_ra_dec


def test_epsilon2e():
    e = epsilon2e(0)
    assert e == 0


def test_e2epsilon():
    ep = e2epsilon(0)
    assert ep == 0


def test_random_ra_dec():
    ra, dec = random_ra_dec(ra_min=30, ra_max=62, dec_min=-63, dec_max=-36, n=50)
    assert len(ra) == 50
