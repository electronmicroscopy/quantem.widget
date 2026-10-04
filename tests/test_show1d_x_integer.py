"""Counted x axes keep whole-number ticks."""

from quantem.widget import Show1D


def test_x_integer_is_opt_in_and_reaches_the_frontend_state():
    assert Show1D.live(["loss"]).x_integer is False
    monitor = Show1D.live(["training", "validation"], x_label="epoch", x_integer=True)
    assert monitor.x_integer is True
    # the frontend reads the synced trait; an unsynced flag would be a dead parameter
    assert monitor.trait_metadata("x_integer", "sync") is True
    assert monitor.get_state("x_integer")["x_integer"] is True
