import numpy as np

from experiments.wan_adapter.evaluate_clip import score_prediction


def test_rgb_score_excludes_known_start_and_uses_truth_for_motion():
    truth = np.zeros((17, 288, 512, 3), dtype=np.uint8)
    truth[1:, :144] = 255
    prediction = truth.copy()
    prediction[0] = 255  # Known observation is explicitly excluded from scoring.
    scores = score_prediction(prediction, truth)
    assert scores["mae"] == 0 and scores["moving_region_mae"] == 0
    # Truth changes on half the pixels of exactly one of16 scored frames.
    assert scores["moving_region_fraction"] == 1 / 32
    assert scores["per_frame"][0]["moving_fraction"] == .5
    assert scores["per_frame"][1]["moving_region_mae"] is None


def test_no_motion_is_reported_as_unavailable_not_a_perfect_motion_score():
    truth = np.zeros((17, 288, 512, 3), dtype=np.uint8)
    prediction = np.full_like(truth, 255)
    scores = score_prediction(prediction, truth)
    assert scores["mae"] == 1 and scores["rmse"] == 1
    assert scores["moving_region_fraction"] == 0
    assert scores["moving_region_mae"] is None
