import numpy as np

from scripts.train_sqrl_anchor_fold import anchor_predictions


def test_anchor_predictions_and_label_knn() -> None:
    query_base = np.asarray([5.0, 9.0])
    reference_base = np.asarray([4.0, 6.0])
    reference_truth = np.asarray([5.0, 4.0])
    neighbors = [np.asarray([0, 1]), np.asarray([], dtype=int)]
    similarities = [np.asarray([0.75, 0.25]), np.asarray([], dtype=float)]
    result = anchor_predictions(
        query_base,
        reference_base,
        reference_truth,
        neighbors,
        similarities,
        np.asarray([2.0, 0.0]),
    )
    # Reference residuals are +1 and -2, so weighted correction is +0.25.
    np.testing.assert_allclose(result["weighted_train_residual_anchor"], [5.25, 9.0])
    np.testing.assert_allclose(result["nearest_train_residual_anchor"], [6.0, 9.0])
    np.testing.assert_allclose(result["weighted_train_label_knn"], [4.75, 9.0])
    np.testing.assert_allclose(result["weighted_permuted_train_residual_anchor"], [6.5, 9.0])

