import unittest

import numpy as np

import models


class TestModels(unittest.TestCase):
    def test_generate_matrices_linear(self):
        points = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]])
        A, b = models.generate_matrices(points, degree=1)

        expected_A = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
            ]
        )
        expected_b = np.array([1.0, 3.0, 5.0])

        np.testing.assert_allclose(A, expected_A)
        np.testing.assert_allclose(b, expected_b)

    def test_least_squares_matches_numpy(self):
        A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        b = np.array([1.0, 2.9, 5.2, 7.1])

        x = models.least_squares(A, b)
        x_ref, *_ = np.linalg.lstsq(A, b, rcond=None)

        np.testing.assert_allclose(x, x_ref, atol=1e-10)

    def test_gradient_descent_converges_to_least_squares(self):
        A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        b = np.array([1.0, 3.0, 5.0, 7.0])

        x_gd = models.gradient_descent(A, b, lr=0.05, tol=1e-10, max_iter=20000)
        x_ref = models.least_squares(A, b)

        np.testing.assert_allclose(x_gd, x_ref, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
