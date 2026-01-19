import unittest
import numpy as np
import pathlib
import sys

# Ensure project root is on sys.path so that the `experiments` package can be imported
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.distributional_alphazero.DistributionalAlphaZero import Distribution


class TestDistributionCategoricalRoundtrip(unittest.TestCase):
    def test_uniform_roundtrip_exact_or_close(self):
        """
        For a uniform distribution on [a, b], to_categorical followed by
        from_categorical should recover (approximately) the same quantile values.
        In this specific construction, the round-trip should be numerically very close.
        """
        num_quantiles = 51
        num_categories = 51
        support = (-1.0, 1.0)

        # Exact uniform quantiles on [a, b]
        a, b = support
        quantile_values = np.linspace(a, b, num_quantiles)
        dist = Distribution(quantile_values=quantile_values, quantile_function="PL")

        pdf = dist.to_categorical(num_categories=num_categories, support=support)
        self.assertEqual(pdf.shape, (num_categories,))
        self.assertAlmostEqual(pdf.sum(), 1.0, places=6)

        dist_roundtrip = Distribution.from_categorical(
            num_quantiles=num_quantiles, pdf=pdf, support=support
        )

        # Uniform case should come back essentially identical (up to tiny float noise)
        np.testing.assert_allclose(
            dist.quantile_values,
            dist_roundtrip.quantile_values,
            rtol=1e-7,
            atol=1e-7,
        )

    def test_random_distribution_roundtrip_approximate(self):
        """
        For a more irregular distribution, the round-trip should approximately
        preserve the quantile function (within a small tolerance).
        """
        rng = np.random.default_rng(seed=1234)

        # Create an arbitrary non-uniform distribution via sorted random samples
        raw_samples = rng.normal(loc=0.0, scale=1.0, size=10_000)
        num_quantiles = 51
        quantile_values = np.percentile(raw_samples, np.linspace(0, 100, num_quantiles))

        # Use the quantile endpoints as the support for the histogram approximation
        support = (float(quantile_values[0]), float(quantile_values[-1]))
        dist = Distribution(quantile_values=quantile_values, quantile_function="PL")

        num_categories = 201
        pdf = dist.to_categorical(num_categories=num_categories, support=support)
        self.assertEqual(pdf.shape, (num_categories,))
        self.assertAlmostEqual(pdf.sum(), 1.0, places=6)

        dist_roundtrip = Distribution.from_categorical(
            num_quantiles=num_quantiles, pdf=pdf, support=support
        )

        # Round-trip should preserve the quantile function up to small discretization error.
        # We check the maximum absolute deviation.
        max_abs_diff = np.max(
            np.abs(dist.quantile_values - dist_roundtrip.quantile_values)
        )
        self.assertLess(max_abs_diff, 0.05)


if __name__ == "__main__":
    unittest.main()


