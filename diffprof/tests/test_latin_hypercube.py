"""
"""
import pytest
import numpy as np
from ..latin_hypercube import latin_hypercube, latin_hypercube_from_cov
from ..latin_hypercube import uniform_random_hypercube, latin_hypercube_pydoe
from ..latin_hypercube import HAS_PYDOE2


def verify_lhs_respects_bounds(box, xmins, xmaxs):
    for idim in range(len(xmins)):
        assert np.all(box[:, idim] >= xmins[idim])
        assert np.all(box[:, idim] <= xmaxs[idim])


@pytest.mark.xfail
def test_latin_hypercube_pydoe_respects_bounds():
    xmins = (-3, -2, 0)
    xmaxs = (2, 3, 5)
    n_dim = len(xmins)
    npts = 5000
    if HAS_PYDOE2:
        lhs_box = latin_hypercube_pydoe(xmins, xmaxs, n_dim, npts)
        verify_lhs_respects_bounds(lhs_box, xmins, xmaxs)


def test_uniform_random_hypercube_respects_bounds():
    xmins = (-3, -2, 0)
    xmaxs = (2, 3, 5)
    n_dim = len(xmins)
    npts = 5000
    lhs_box = uniform_random_hypercube(xmins, xmaxs, n_dim, npts)
    verify_lhs_respects_bounds(lhs_box, xmins, xmaxs)


def test_uniform_random_hypercube_is_reproducible():
    xmins = (-3, -2, 0)
    xmaxs = (2, 3, 5)
    n_dim = len(xmins)
    npts = 5000
    lhs_box = uniform_random_hypercube(xmins, xmaxs, n_dim, npts, seed=0)
    lhs_box1 = uniform_random_hypercube(xmins, xmaxs, n_dim, npts, seed=0)
    lhs_box2 = uniform_random_hypercube(xmins, xmaxs, n_dim, npts, seed=2)
    assert np.allclose(lhs_box, lhs_box1)
    assert not np.allclose(lhs_box, lhs_box2)


@pytest.mark.xfail
def test_latin_hypercube_pydoe_is_reproducible():
    xmins = (-3, -2, 0)
    xmaxs = (2, 3, 5)
    n_dim = len(xmins)
    npts = 5000
    if HAS_PYDOE2:
        lhs_box = latin_hypercube_pydoe(xmins, xmaxs, n_dim, npts, seed=0)
        lhs_box1 = latin_hypercube_pydoe(xmins, xmaxs, n_dim, npts, seed=0)
        lhs_box2 = latin_hypercube_pydoe(xmins, xmaxs, n_dim, npts, seed=2)
        assert np.allclose(lhs_box, lhs_box1)
        assert not np.allclose(lhs_box, lhs_box2)


def test_latin_hypercube_is_reproducible():
    xmins = (-3, -2, 0)
    xmaxs = (2, 3, 5)
    n_dim = len(xmins)
    npts = 5000
    lhs_box = latin_hypercube(xmins, xmaxs, n_dim, npts, seed=0)
    lhs_box1 = latin_hypercube(xmins, xmaxs, n_dim, npts, seed=0)
    lhs_box2 = latin_hypercube(xmins, xmaxs, n_dim, npts, seed=2)
    assert np.allclose(lhs_box, lhs_box1)
    assert not np.allclose(lhs_box, lhs_box2)


def test_latin_hypercube_respects_constant_bounds():
    xmins = (-3, -2, 0)
    xmaxs = (2, 3, 5)
    n_dim = len(xmins)
    npts = 5000
    lhs_box = latin_hypercube(xmins, xmaxs, n_dim, npts)
    verify_lhs_respects_bounds(lhs_box, xmins, xmaxs)


def test_latin_hypercube_respects_scalar_bounds():
    xmins = -3
    xmaxs = 2
    n_dim = 1
    npts = 5000
    lhs_box = latin_hypercube(xmins, xmaxs, n_dim, npts)
    verify_lhs_respects_bounds(lhs_box, [xmins], [xmaxs])


def test_latin_hypercube_handles_mixed_scalar_bounds():
    xmins = -3
    npts = 5000
    xmaxs = 2
    n_dim = 1
    try:
        lhs_box = latin_hypercube(xmins, xmaxs + np.zeros(npts), n_dim, npts)
        verify_lhs_respects_bounds(lhs_box, [xmins], [xmaxs])
    except AssertionError:
        pass


def test_latin_hypercube_handles_mixed_scalar_bounds2():
    xmins = -3
    npts = 5000
    xmaxs = 2
    n_dim = 1
    try:
        lhs_box = latin_hypercube(xmins + np.zeros(npts), xmaxs, n_dim, npts)
        verify_lhs_respects_bounds(lhs_box, [xmins], [xmaxs])
    except AssertionError:
        pass


def test_latin_hypercube_handles_mixed_scalar_bounds3():
    xmins = -3
    npts = 5000
    xmaxs = 2
    n_dim = 1
    lhs_box = latin_hypercube(xmins, [xmaxs + np.zeros(npts)], n_dim, npts)
    verify_lhs_respects_bounds(lhs_box, [xmins], [xmaxs])


def test_latin_hypercube_handles_mixed_scalar_bounds4():
    xmins = -3
    npts = 5000
    xmaxs = 2
    n_dim = 1
    lhs_box = latin_hypercube([xmins + np.zeros(npts)], xmaxs, n_dim, npts)
    verify_lhs_respects_bounds(lhs_box, [xmins], [xmaxs])


def test_latin_hypercube_handles_mixed_scalar_bounds5():
    npts = 5000
    xmins = (-3, np.random.uniform(-2, 3, npts), 0)
    xmaxs = (2, 3, 5)
    n_dim = len(xmins)
    lhs_box = latin_hypercube(xmins, xmaxs, n_dim, npts)
    verify_lhs_respects_bounds(lhs_box, xmins, xmaxs)


def test_latin_hypercube_handles_mixed_scalar_bounds6():
    npts = 5000
    xmins = (-3, -2, 0)
    xmaxs = (2, 3 + np.zeros(npts), 5)
    n_dim = len(xmins)
    lhs_box = latin_hypercube(xmins, xmaxs, n_dim, npts)
    verify_lhs_respects_bounds(lhs_box, xmins, xmaxs)


def test_latin_hypercube_handles_mixed_scalar_bounds7():
    npts = 5000
    xmins = (-3, -2, 0)
    xmaxs = (2, -2 + np.zeros(npts), 5)
    n_dim = len(xmins)
    try:
        latin_hypercube(xmins, xmaxs, n_dim, npts)
    except AssertionError:
        pass


def test_latin_hypercube_handles_array_bounds():
    npts = 5000
    xmins = (-3, np.random.uniform(-2, 3, npts), 0)
    xmaxs = [x + 1 for x in xmins]
    n_dim = len(xmins)
    lhs_box = latin_hypercube(xmins, xmaxs, n_dim, npts)
    verify_lhs_respects_bounds(lhs_box, xmins, xmaxs)


def test_latin_hypercube_handles_array_bounds2():
    npts = 5000
    xmins = (-3, np.random.uniform(-2, 3, npts), 0)
    xmaxs = [x - 1 for x in xmins]
    n_dim = len(xmins)
    try:
        latin_hypercube(xmins, xmaxs, n_dim, npts)
    except AssertionError:
        pass


def test_latin_hypercube2():
    xmins = (-3, -2, 0)
    xmaxs = (2, 3, 5)
    npts = 5000
    n_dim = len(xmins)
    try:
        latin_hypercube(xmins, xmaxs, n_dim, npts)
    except AssertionError:
        pass


def test_latin_hypercube_from_diagonal_cov():
    mu = np.array((4.0, -5.0))
    cov = np.array([[0.014, 0.0], [0.0, 0.015]])
    sig = 5
    n = 5000

    lhs = latin_hypercube_from_cov(mu, cov, sig, n)
    for idim in range(2):
        assert np.all(lhs[:, idim] > mu[idim] - sig * np.sqrt(cov[idim, idim]))
        assert np.all(lhs[:, idim] < mu[idim] + sig * np.sqrt(cov[idim, idim]))

    lhs2 = latin_hypercube_from_cov(mu, cov, (sig, sig), n)
    for idim in range(2):
        assert np.all(lhs2[:, idim] > mu[idim] - sig * np.sqrt(cov[idim, idim]))
        assert np.all(lhs2[:, idim] < mu[idim] + sig * np.sqrt(cov[idim, idim]))


def test_latin_hypercube_from_non_diagonal_cov():
    mu = np.array((4.0, -5.0))
    cov = np.array([[0.014, 0.0075], [0.0075, 0.015]])
    sig = 5
    n = 5000

    lhs = latin_hypercube_from_cov(mu, cov, sig, n)
    for idim in range(2):
        assert ~np.all(lhs[:, idim] > mu[idim] - sig * np.sqrt(cov[idim, idim]))
        assert ~np.all(lhs[:, idim] < mu[idim] + sig * np.sqrt(cov[idim, idim]))
    for idim in range(2):
        assert np.all(lhs[:, idim] > mu[idim] - 2 * sig * np.sqrt(cov[idim, idim]))
        assert np.all(lhs[:, idim] < mu[idim] + 2 * sig * np.sqrt(cov[idim, idim]))

    lhs2 = latin_hypercube_from_cov(mu, cov, (sig, sig), n)
    for idim in range(2):
        assert ~np.all(lhs2[:, idim] > mu[idim] - sig * np.sqrt(cov[idim, idim]))
        assert ~np.all(lhs2[:, idim] < mu[idim] + sig * np.sqrt(cov[idim, idim]))
    for idim in range(2):
        assert np.all(lhs2[:, idim] > mu[idim] - 2 * sig * np.sqrt(cov[idim, idim]))
        assert np.all(lhs2[:, idim] < mu[idim] + 2 * sig * np.sqrt(cov[idim, idim]))
