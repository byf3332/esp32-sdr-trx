import numpy as np
import pytest

from espdr import config, txlink as tl, txmodes as tm


@pytest.mark.parametrize("mhz", [2320.0, 2335.5, 2345.0, 2350.0, 2400.1, 2449.99])
def test_the_plan_puts_the_signal_where_it_was_asked(mhz):
    p = tl.plan_lo(mhz * 1e6)
    assert tl.TX_LOW_MARGIN <= tl.lo_word(p["lo_hz"]) & 0xFF <= 255 - tl.TX_LOW_MARGIN
    assert p["lo_hz"] % 100 == 0 and abs(p["miss_hz"]) < 15           # a record unit is 28.6 Hz
    assert abs(p["static_hz"]) <= tm.MAX_Q4 * tm.Q4_HZ + 1e-9


def test_a_crystal_error_moves_the_lo_the_other_way():
    a, b = tl.plan_lo(2_350_000_000, ppm=0.0), tl.plan_lo(2_350_000_000, ppm=5.0)
    assert abs((b["lo_hz"] + b["static_hz"]) - (a["lo_hz"] + a["static_hz"]) + 11_750) < 300      # about 5 ppm of 2.35 GHz lower
    assert abs(b["miss_hz"]) < 15


def test_the_reserve_for_the_deviation_limits_the_static_offset():
    for mhz in np.arange(2320.0, 2450.0, 0.1):
        p = tl.plan_lo(round(mhz * 1e6), reserve_hz=3100.0)
        assert abs(p["static_hz"]) + 3100.0 <= tm.MAX_Q4 * tm.Q4_HZ + tm.Q4_HZ


def test_about_a_tenth_of_the_frequencies_cannot_be_reached_exactly_with_an_m17_reserve():
    misses = [abs(tl.plan_lo(round(mhz * 1e6), reserve_hz=3100.0)["miss_hz"]) for mhz in np.arange(2320.0, 2450.0, 0.0457)]
    share = np.mean(np.array(misses) > 15)
    assert 0.05 < share < 0.15                                        # the low byte within 13 steps of its ends
    assert max(misses) < 7000                                         # never more than a few kHz


def test_even_without_a_reserve_the_records_cannot_follow_a_move_beyond_their_range():
    misses = np.array([abs(tl.plan_lo(round(mhz * 1e6))["miss_hz"]) for mhz in np.arange(2320.0, 2450.0, 0.0457)])
    with_reserve = np.array([abs(tl.plan_lo(round(mhz * 1e6), reserve_hz=3100.0)["miss_hz"]) for mhz in np.arange(2320.0, 2450.0, 0.0457)])
    assert 0.03 < np.mean(misses > 15) < 0.07                         # the PLL word moves up to 48 steps, the records reach 42
    assert np.mean(with_reserve > 15) > np.mean(misses > 15)


def test_the_licence_notice_and_the_acceptance_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert not config.licence_accepted()
    config.accept_licence("1.2.3")
    assert config.licence_accepted() and "1.2.3" in (tmp_path / "espdr" / "licence-accepted").read_text()
    assert "amateur radio licence" in config.LICENCE_NOTICE


def test_the_model_of_a_board_is_found_by_its_file_name(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config.board_model() is None
    (tmp_path / "espdr").mkdir()
    (tmp_path / "espdr" / "thermal.json").write_text("{}")
    assert config.board_model() == str(tmp_path / "espdr" / "thermal.json")
