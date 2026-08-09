import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.train_residual_feasibility_fold import directed_features


def test_directed_features_reverse_exactly() -> None:
    left_embedding = np.asarray([[1.0, 3.0], [2.0, -1.0]])
    right_embedding = np.asarray([[0.0, 4.0], [-2.0, 5.0]])
    left_prediction = np.asarray([2.0, 1.0])
    right_prediction = np.asarray([0.5, 3.0])
    forward = directed_features(left_embedding, right_embedding, left_prediction, right_prediction)
    reverse = directed_features(right_embedding, left_embedding, right_prediction, left_prediction)
    np.testing.assert_allclose(reverse, -forward)


def test_no_intercept_ridge_prediction_is_antisymmetric() -> None:
    features = np.asarray([[1.0, 2.0], [-2.0, 1.0], [0.5, -3.0]])
    target = np.asarray([1.5, -0.25, 2.0])
    augmented_features = np.vstack([features, -features])
    augmented_target = np.concatenate([target, -target])
    model = make_pipeline(
        StandardScaler(with_mean=False), Ridge(alpha=10.0, fit_intercept=False)
    ).fit(augmented_features, augmented_target)
    np.testing.assert_allclose(model.predict(features), -model.predict(-features), atol=1e-12)

