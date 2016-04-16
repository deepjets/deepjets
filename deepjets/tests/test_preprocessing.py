
from nose.tools import (raises, assert_raises, assert_true,
                        assert_equal, assert_not_equal, assert_almost_equal)

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from deepjets.preprocessing import zoom_image, pixel_edges


def test_zoom():
    edges = pixel_edges(jet_size=1, pixel_size=(0.1, 0.1), border_size=0.25)
    assert_equal(edges[0].shape, (26,))
    assert_equal(edges[1].shape, (26,))
    image, _, _ = np.histogram2d(
        np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000),
        bins=(edges[0], edges[1]))
    assert_true(image.sum() > 0)
    assert_equal(image.shape, (25, 25))

    # zooming with factor 1 should not change anything
    image_zoomed = zoom_image(image, 1, out_width=25)
    assert_array_almost_equal(image, image_zoomed)

    assert_raises(ValueError, zoom_image, image, 0.5)

    # test out_width
    assert_equal(zoom_image(image, 1, out_width=11).shape, (11, 11))

    image_zoomed = zoom_image(image, 2, out_width=25)
    assert_true(image.sum() < image_zoomed.sum())
