"""Tests for the push-up state machine."""

from src.counting.state_machine import PushUpPhase, PushUpStateMachine


def test_initial_state():
    sm = PushUpStateMachine()
    assert sm.state == PushUpPhase.UP
    assert sm.count == 0


def test_full_rep_cycle():
    """Simulate one complete push-up rep."""
    sm = PushUpStateMachine(down_threshold=90, up_threshold=160)

    # Start UP, arm extended
    sm.update(170)
    assert sm.state == PushUpPhase.UP

    # Going down
    sm.update(150)
    assert sm.state == PushUpPhase.GOING_DOWN

    sm.update(120)
    assert sm.state == PushUpPhase.GOING_DOWN

    # Reach bottom
    sm.update(80)
    assert sm.state == PushUpPhase.DOWN

    # Coming back up
    sm.update(100)
    assert sm.state == PushUpPhase.GOING_UP

    sm.update(140)
    assert sm.state == PushUpPhase.GOING_UP

    # Full extension = 1 rep
    sm.update(165)
    assert sm.state == PushUpPhase.UP
    assert sm.count == 1


def test_multiple_reps():
    sm = PushUpStateMachine(down_threshold=90, up_threshold=160)

    angles = [170, 150, 120, 80, 100, 140, 165,  # rep 1
              150, 120, 80, 100, 140, 165]         # rep 2

    for a in angles:
        sm.update(a)

    assert sm.count == 2


def test_no_count_without_full_extension():
    """Partial rep (never reaches UP threshold) should not count."""
    sm = PushUpStateMachine(down_threshold=90, up_threshold=160)

    angles = [170, 150, 80, 100, 140, 80, 100, 140]
    for a in angles:
        sm.update(a)

    assert sm.count == 0


def test_reset():
    sm = PushUpStateMachine()
    sm.update(80)
    sm.reset()
    assert sm.state == PushUpPhase.UP
    assert sm.count == 0
    assert len(sm.phase_history) == 0
