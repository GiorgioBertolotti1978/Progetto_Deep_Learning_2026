"""Test numerici e casi limite senza implementazioni esterne di K-means."""

import unittest

import numpy as np

from src.evaluation import evaluate_clustering
from src.kmeans import KMeansFromScratch, fit_best_kmeans


class TestKMeansFromScratch(unittest.TestCase):
    """Verifica invarianti matematiche e comportamenti pubblici."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.X = np.vstack([
            rng.normal(-8, 0.2, size=(20, 4)),
            rng.normal(0, 0.2, size=(20, 4)),
            rng.normal(8, 0.2, size=(20, 4)),
        ])

    def assert_consistent(self, model, X):
        """Verifica la SSE direttamente e l'assegnazione al centro più vicino."""
        distances = ((X[:, None, :] - model.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        np.testing.assert_array_equal(model.labels_, distances.argmin(axis=1))
        expected = ((X - model.cluster_centers_[model.labels_]) ** 2).sum()
        self.assertAlmostEqual(model.inertia_, float(expected), places=9)
        self.assertEqual(model.inertia_, model.inertia_history_[-1])
        np.testing.assert_array_equal(model.cluster_centers_, model.centroids_history_[-1])

    def test_shapes_labels_and_finite_centers(self):
        model = KMeansFromScratch().fit(self.X)
        self.assertEqual(model.cluster_centers_.shape, (3, 4))
        self.assertEqual(model.labels_.shape, (60,))
        self.assertTrue(np.isfinite(model.cluster_centers_).all())
        self.assertTrue(np.all((model.labels_ >= 0) & (model.labels_ < 3)))
        self.assert_consistent(model, self.X)

    def test_reproducibility(self):
        first = KMeansFromScratch(random_state=12).fit(self.X)
        second = KMeansFromScratch(random_state=12).fit(self.X)
        np.testing.assert_array_equal(first.cluster_centers_, second.cluster_centers_)
        np.testing.assert_array_equal(first.labels_, second.labels_)
        self.assertEqual(first.inertia_history_, second.inertia_history_)

    def test_history_and_monotonicity_for_multiple_seeds(self):
        for seed in range(15):
            with self.subTest(seed=seed):
                model = KMeansFromScratch(random_state=seed).fit(self.X)
                self.assertEqual(len(model.inertia_history_), model.n_iter_ + 1)
                self.assertEqual(len(model.centroids_history_), model.n_iter_ + 1)
                self.assertTrue(np.all(np.diff(model.inertia_history_) <= 1e-9))
                for centers, inertia in zip(model.centroids_history_, model.inertia_history_):
                    expected = ((self.X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).min(axis=1).sum()
                    self.assertAlmostEqual(inertia, float(expected), places=9)

    def test_predict_and_fit_predict(self):
        model = fit_best_kmeans(self.X)
        np.testing.assert_array_equal(model.predict(self.X), model.labels_)
        anchors = np.array([[-8] * 4, [0] * 4, [8] * 4], dtype=float)
        self.assertEqual(len(np.unique(model.predict(anchors))), 3)
        direct = KMeansFromScratch(random_state=8)
        np.testing.assert_array_equal(direct.fit_predict(self.X), direct.labels_)
        self.assert_consistent(direct, self.X)

    def test_k_larger_than_sample_count(self):
        with self.assertRaises(ValueError):
            KMeansFromScratch(n_clusters=61).fit(self.X)

    def test_empty_clusters_with_duplicate_points(self):
        X = np.array([[0, 0], [0, 0], [0, 0], [10, 10], [10, 10]], dtype=float)
        model = KMeansFromScratch(n_clusters=4).fit(X)
        self.assertGreater(model.empty_cluster_relocations_, 0)
        self.assertTrue(np.isfinite(model.cluster_centers_).all())
        self.assertLessEqual(len(np.unique(model.labels_)), 2)
        self.assertTrue(np.all(np.diff(model.inertia_history_) <= 1e-9))
        self.assert_consistent(model, X)

    def test_all_identical_points(self):
        X = np.ones((8, 4))
        model = KMeansFromScratch(n_clusters=3).fit(X)
        self.assertEqual(model.inertia_, 0)
        self.assertTrue(model.converged_)
        self.assert_consistent(model, X)

    def test_final_consistency_at_iteration_limit_and_large_tolerance(self):
        for kwargs in [{"max_iter": 1}, {"tol": 1e10}]:
            with self.subTest(kwargs=kwargs):
                model = KMeansFromScratch(**kwargs).fit(self.X)
                self.assertEqual(model.n_iter_, 1)
                self.assert_consistent(model, self.X)

    def test_single_cluster_is_mean(self):
        model = KMeansFromScratch(n_clusters=1).fit(self.X)
        np.testing.assert_allclose(model.cluster_centers_[0], self.X.mean(axis=0))
        self.assert_consistent(model, self.X)

    def test_converged_centers_are_means(self):
        model = fit_best_kmeans(self.X, tol=0)
        self.assertTrue(model.converged_)
        for cluster in range(3):
            np.testing.assert_allclose(model.cluster_centers_[cluster], self.X[model.labels_ == cluster].mean(axis=0))

    def test_invalid_data_and_parameters(self):
        for X in [[], [[1, 2], [3]], [1, 2], [[np.nan]], [[np.inf]], [["fiore"]], np.empty((2, 0))]:
            with self.subTest(X=X), self.assertRaises(ValueError):
                KMeansFromScratch().fit(X)
        for kwargs in [{"n_clusters": 0}, {"n_clusters": 1.5}, {"n_clusters": True},
                       {"max_iter": 0}, {"max_iter": 2.5}, {"tol": -1}, {"tol": np.nan}, {"tol": np.inf}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                KMeansFromScratch(**kwargs).fit(self.X)
        with self.assertRaises(ValueError):
            fit_best_kmeans(self.X, n_init=0)

    def test_predict_errors(self):
        with self.assertRaises(ValueError):
            KMeansFromScratch().predict(self.X)
        model = KMeansFromScratch().fit(self.X)
        for X in [np.zeros((2, 3)), [[np.nan] * 4], []]:
            with self.subTest(X=X), self.assertRaises(ValueError):
                model.predict(X)

    def test_best_of_twenty_is_minimum_and_reproducible(self):
        model = fit_best_kmeans(self.X, n_init=20)
        self.assertEqual(model.inertia_, min(model.initialization_inertias_))
        self.assertEqual(len(set(model.initialization_seeds_)), 20)
        repeated = fit_best_kmeans(self.X, n_init=20)
        np.testing.assert_array_equal(model.labels_, repeated.labels_)
        self.assert_consistent(model, self.X)

    def test_external_metrics_ignore_cluster_numbering(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        labels = np.array([2, 2, 0, 0, 1, 1])
        X = np.array([[0], [0.1], [5], [5.1], [10], [10.1]])
        metrics, table = evaluate_clustering(X, y, labels, np.array(["a", "b", "c"]), 3)
        self.assertEqual(metrics["purity"], 1)
        self.assertEqual(metrics["adjusted_rand_index"], 1)
        self.assertEqual(int(table.to_numpy().sum()), 6)
        one_cluster, _ = evaluate_clustering(X, y, np.zeros(6, dtype=int), np.array(["a", "b", "c"]), 3)
        self.assertIsNone(one_cluster["silhouette_score"])


if __name__ == "__main__":
    unittest.main()
