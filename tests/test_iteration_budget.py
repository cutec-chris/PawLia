from pawlia.agents.iteration_budget import IterationBudget


def test_consume_within_budget():
    b = IterationBudget(3)
    assert b.consume() is True
    assert b.consume() is True
    assert b.consume() is True


def test_grace_call_after_exhaustion():
    b = IterationBudget(2)
    b.consume()
    b.consume()
    # One grace call allowed
    assert b.consume() is True
    # No more after grace
    assert b.consume() is False


def test_no_calls_allowed_after_grace():
    b = IterationBudget(1)
    b.consume()          # uses budget
    b.consume()          # grace
    assert b.consume() is False
    assert b.consume() is False


def test_refund_restores_budget():
    b = IterationBudget(2)
    b.consume()
    b.consume()
    b.refund()
    assert b.consume() is True   # restored one slot


def test_refund_does_not_go_below_zero():
    b = IterationBudget(2)
    b.refund()  # no-op
    assert b.used == 0


def test_remaining_counts_down():
    b = IterationBudget(5)
    assert b.remaining == 5
    b.consume()
    assert b.remaining == 4
    b.consume()
    assert b.remaining == 3


def test_remaining_never_negative():
    b = IterationBudget(1)
    b.consume()
    b.consume()  # grace
    b.consume()  # denied
    assert b.remaining == 0


def test_used_tracks_consumed():
    b = IterationBudget(3)
    assert b.used == 0
    b.consume()
    assert b.used == 1
    b.consume()
    assert b.used == 2
