import unittest
from importlib.util import find_spec
from unittest.mock import Mock

if find_spec("timestral") is None:
    raise unittest.SkipTest("Install requirements-timestral.txt to test this optional runner")

import numpy as np

from run_timestral import detect


class TimestralTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        bounds = np.array([9000] * 116 + [100, 200, 300, 400])
        self.model.forecast.return_value.quantile.return_value = bounds.reshape(1, 1, 120)

    def test_only_first_half_enters_forecast_and_bounds_align_with_last_minute(self):
        context = list(range(120))
        context[12] = np.nan
        page, _ = detect(self.model, context + [9000] * 116 + [101, 201, 301, 401])
        self.assertTrue(page)
        args, kwargs = self.model.forecast.call_args
        np.testing.assert_equal(args[0], context)
        self.assertEqual(kwargs["prediction_length"], 120)
        self.model.forecast.return_value.quantile.assert_called_once_with(0.99)

    def test_every_observed_sample_must_strictly_exceed_its_bound(self):
        for target, expected in [
            ([101, 201, 301, 400], False),
            ([None, 201, 301, 401], True),
            ([None, None, None, 400], False),
            ([None, None, None, 401], True),
        ]:
            with self.subTest(target=target):
                page, _ = detect(self.model, [100] * 236 + target)
                self.assertEqual(page, expected)

    def test_empty_context_or_target(self):
        for values in [[None] * 120 + [100] * 120, [100] * 236 + [None] * 4]:
            self.assertFalse(detect(self.model, values)[0])
        self.model.forecast.assert_not_called()
