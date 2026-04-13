import unittest

import numpy as np

import metrics


class TestMetrics(unittest.TestCase):
    def test_condition_number_identity(self):
        A = np.eye(3)
        self.assertEqual(metrics.condition_number(A), 1.0)

    def test_squared_residuals_and_norm(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_fit = np.array([1.0, 1.0, 5.0])

        self.assertEqual(metrics.squared_residuals(y_true, y_fit), 5.0)
        self.assertAlmostEqual(metrics.residual_norm(y_true, y_fit), np.sqrt(5.0))

    def test_compute_errors(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])

        errors = metrics.compute_errors(y_true, y_pred)

        self.assertAlmostEqual(errors["RSS"], 1.0)
        self.assertAlmostEqual(errors["MSE"], 1.0 / 3.0)
        self.assertAlmostEqual(errors["RMSE"], np.sqrt(1.0 / 3.0))
        self.assertAlmostEqual(errors["R2"], 0.5)


if __name__ == "__main__":
    unittest.main()
