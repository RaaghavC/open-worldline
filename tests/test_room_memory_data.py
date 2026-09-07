# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from experiments.room_world.memory_data import (
    DEVELOPMENT_SCENES, RESERVED_TEST_SCENES, make_pair, observed_history,
    write_development,
)


def test_successful_pair_observes_cause_before_exact_return_aliasing():
    pair = make_pair(5000, 5000*1009+37, size=16)
    rgb, actions, record = pair["observations"], pair["actions"], pair["record"]
    assert rgb.shape == (2, 66, 3, 16, 16)
    assert actions.shape == (2, 65)
    np.testing.assert_array_equal(rgb[0, 0], rgb[1, 0])
    assert actions[:, 0].tolist() == [0, 5]
    assert not np.array_equal(rgb[0, 1], rgb[1, 1])
    np.testing.assert_array_equal(actions[0, 1:], actions[1, 1:])
    np.testing.assert_array_equal(observed_history(rgb[0], 41), observed_history(rgb[1], 41))
    assert record["first_return_action_index"] == 41
    assert actions[0, 40] == 0 and actions[0, 41] == 4
    assert not np.array_equal(rgb[0, -1], rgb[1, -1])


@pytest.mark.parametrize("waiting", [4, 8, 24, 32])
def test_unsuccessful_interaction_is_a_visually_identical_control(waiting):
    pair = make_pair(5001, 37, wait_steps=waiting, successful=False, size=16)
    np.testing.assert_array_equal(pair["observations"][0], pair["observations"][1])
    assert pair["actions"].shape == (2, 49+waiting)
    assert not np.asarray(pair["record"]["teacher_door_states"]).any()


def test_observed_history_cannot_read_the_target_or_later_frames():
    observations = np.arange(8, dtype=np.uint8)[:, None, None, None]*np.ones((8,3,8,8), np.uint8)
    before = observed_history(observations, 4)
    observations[5:] = 255
    np.testing.assert_array_equal(observed_history(observations, 4), before)
    np.testing.assert_allclose(before[:, 0, 0, 0], np.asarray([1,2,3,4])/127.5-1, atol=1e-7)
    np.testing.assert_array_equal(observed_history(observations, 0), np.full((4,3,8,8), -1, np.float32))


def test_reserved_test_split_is_disjoint_and_development_writer_rejects_it(tmp_path):
    train, validation, test = map(set, [DEVELOPMENT_SCENES["train"], DEVELOPMENT_SCENES["validation"], RESERVED_TEST_SCENES])
    assert not train & validation and not train & test and not validation & test
    with pytest.raises(ValueError, match="Only train and validation"):
        write_development(tmp_path/"test", "test", size=16)
    assert not (tmp_path/"test").exists()
