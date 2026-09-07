# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from experiments.room_world.memory_controls import CONTROL_ACTIONS, make_control


@pytest.mark.parametrize("kind", list(CONTROL_ACTIONS))
def test_fixed_controls_align_actions_and_verify_physical_outcomes(kind):
    case=make_control(300000,300000*1009+37,kind,size=16)
    assert case["observations"].shape==(1,17,3,16,16)
    np.testing.assert_array_equal(case["actions"][0],CONTROL_ACTIONS[kind])
    record=case["record"]
    if kind=="translation_cycle":
        assert record["translations_verified"]==16
        assert not any(record["teacher_door_states"])
    elif kind=="out_of_reach_interaction":
        assert record["translations_verified"]==12
        assert record["interaction_changed_door"]==[False]
        assert not any(record["teacher_door_states"])
    else:
        assert record["interaction_changed_door"]==[True,True]
        states=record["teacher_door_states"]
        assert not states[8] and states[9] and states[12] and not states[13]


def test_short_control_set_covers_every_command_without_changing_script_lengths():
    assert set().union(*map(set,CONTROL_ACTIONS.values()))==set(range(6))
    assert all(len(actions)==16 for actions in CONTROL_ACTIONS.values())
